"""Read-only dataset inventory, image quality and content-identity helpers.

Uses the project's sole taxonomy mapping. No dataset loader is instantiated,
and no training caches, images, labels or split directories are written.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
try:
    from .taxonomy import load_current_label_schema
except ImportError:
    from taxonomy import load_current_label_schema

# Match the existing training decoder; no package installation is performed.
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pillow_avif = None

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif', '.avif'}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def enumerate_dataset(project_root, splits=('train', 'val')):
    """Return deterministic on-disk records and mapping without changing caches."""
    root = Path(project_root).resolve()
    id_to_class, indices, mapping_hash = load_current_label_schema(str(root))
    names = [name for name, index in sorted(indices.items(), key=lambda x: int(x[1]))]
    records, skipped = [], []
    for split in splits:
        split_root = root / split
        for folder in sorted(split_root.iterdir(), key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name)):
            if not folder.is_dir() or not folder.name.isdigit() or folder.name not in id_to_class:
                skipped.append({'split': split, 'path': folder.relative_to(root).as_posix(), 'reason': 'not_mapped_class_directory'})
                continue
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    skipped.append({'split': split, 'path': path.relative_to(root).as_posix(), 'reason': 'not_supported_image_file'})
                    continue
                stat = path.stat(); label = id_to_class[folder.name]
                records.append({'split': split, 'relative_path': path.relative_to(root).as_posix(),
                                'split_relative_path': path.relative_to(split_root).as_posix(),
                                'raw_class_id': int(folder.name), 'class_index': int(indices[label]),
                                'class_name': label, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return records, skipped, {'names': names, 'label_mapping_hash': mapping_hash,
                              'num_classes': len(names), 'raw_classes': len(id_to_class)}


def _pixel_properties(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgb = image.convert('RGB'); width, height = rgb.size
        pixels = rgb.tobytes()
        digest = hashlib.sha256(struct.pack('>II', width, height) + pixels).hexdigest()
        extrema = rgb.getextrema()
        constant = all(low == high for low, high in extrema)
        gray = rgb.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
        tiny = np.asarray(gray, dtype=np.uint8)
        bits = np.packbits((tiny[:, 1:] > tiny[:, :-1]).ravel()).tobytes().hex()
        return {'width': width, 'height': height, 'format': image.format,
                'pixel_sha256': digest, 'dhash64': bits, 'constant_rgb': constant,
                'min_side': min(width, height), 'aspect_ratio': max(width, height)/min(width, height),
                'thumbnail_gray_std': float(tiny.std()),
                'frame_count': int(getattr(image, 'n_frames', 1)),
                'exif_orientation': image.getexif().get(274, 1)}


def inspect_record(project_root, record):
    """Decode with the caller's fixed Pillow truncation policy; preserve errors."""
    row = dict(record); path = Path(project_root) / row['relative_path']
    raw = path.read_bytes(); row['sha256'] = hashlib.sha256(raw).hexdigest()
    row['verify_error'] = ''
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
    except Exception as error:
        row['verify_error'] = f'{type(error).__name__}: {error}'
    try:
        row.update(_pixel_properties(raw)); row['decode_status'] = 'strict_ok'; row['decode_error'] = ''
    except Exception as error:
        row['decode_status'] = 'strict_failed'; row['decode_error'] = f'{type(error).__name__}: {error}'
    after = path.stat()
    if after.st_size != row['bytes'] or after.st_mtime_ns != row['mtime_ns']:
        raise RuntimeError(f'Source changed during scan: {path}')
    return row


def scan_dataset(project_root, records, workers=6, progress=None):
    """Strict parallel decode, then serial tolerant fallback matching training.

    Pillow's truncation flag is global, so the two modes never run concurrently.
    """
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    rows = []
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(lambda r: inspect_record(project_root, r), records), 1):
                rows.append(row)
                if progress and (i % 5000 == 0 or i == len(records)):
                    progress(i, len(records))
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        for row in rows:
            if row['decode_status'] == 'strict_failed':
                try:
                    row.update(_pixel_properties((Path(project_root)/row['relative_path']).read_bytes()))
                    row['decode_status'] = 'tolerant_only'
                except Exception as error:
                    row['decode_status'] = 'failed'
                    row['tolerant_error'] = f'{type(error).__name__}: {error}'
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous
    return rows


def group_content(records, key):
    """Group by byte, pixel, or perceptual identity, excluding absent values."""
    groups = defaultdict(list)
    for record in records:
        if record.get(key):
            groups[record[key]].append(record)
    return {digest: items for digest, items in groups.items() if len(items) > 1}


def technical_reasons(record):
    reasons = []
    if record['decode_status'] == 'failed':
        reasons.append('cannot_decode_in_current_loader_mode')
    if record.get('constant_rgb'):
        reasons.append('spatially_constant_RGB_image')
    return reasons


def build_clean_selection(records):
    """Conservative deterministic future-data selection; original files remain.

    Cross-label exact pixel groups are quarantined in full. Same-label copies
    retain one validation sample if present, otherwise one training sample.
    """
    reasons = {r['relative_path']: technical_reasons(r) for r in records}
    valid = [r for r in records if not reasons[r['relative_path']]]
    groups = group_content(valid, 'pixel_sha256')
    for digest, items in groups.items():
        labels = {r['class_index'] for r in items}
        if len(labels) > 1:
            for row in items:
                reasons[row['relative_path']].append('exact_pixel_content_has_conflicting_labels')
        else:
            ordered = sorted(items, key=lambda r: (r['split'] != 'val', r['relative_path']))
            for row in ordered[1:]:
                reasons[row['relative_path']].append('duplicate_content_validation_priority' if ordered[0]['split']=='val' and row['split']=='train' else 'duplicate_content_same_label')
    selected = [r for r in records if not reasons[r['relative_path']]]
    return selected, reasons


def read_manifest_samples(path, project_root, verify_content=False):
    """Public manifest reader for audited samples; no implicit training activation."""
    root = Path(project_root).resolve()
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line); absolute = (root / row['relative_path']).resolve()
            if not absolute.is_relative_to(root):
                raise ValueError('Manifest path escapes project root')
            if verify_content and hashlib.sha256(absolute.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Manifest content changed: {absolute}')
            yield str(absolute), int(row['class_index'])


def validation_quality_cohorts(records, validation_records):
    """Fixed validation masks aligned to an existing content-hashed cache.

    Overlap is checked against ALL original training samples, because the cached
    models were trained before the new manifests existed. Unique validation uses
    deterministic relative-path order and does not consult model predictions.
    """
    source = {r['relative_path']: r for r in records}
    validation = []
    for cached in validation_records:
        row = source['val/' + cached['relative_path']]
        if row['sha256'] != cached['sha256'] or row['class_index'] != cached['true_index']:
            raise ValueError(f'Validation identity mismatch: {row["relative_path"]}')
        validation.append(row)
    train = [r for r in records if r['split'] == 'train']
    byte_ids = {r['sha256'] for r in train}
    pixel_ids = {r['pixel_sha256'] for r in train if r.get('pixel_sha256')}
    valid = np.array([not technical_reasons(r) for r in validation], dtype=bool)
    byte_overlap = np.array([r['sha256'] in byte_ids for r in validation])
    pixel_overlap = np.array([r.get('pixel_sha256') in pixel_ids for r in validation])
    unique = np.zeros(len(validation), dtype=bool)
    seen = set()
    for i in sorted(range(len(validation)), key=lambda i: validation[i]['relative_path']):
        row = validation[i]
        if valid[i] and not pixel_overlap[i] and row['pixel_sha256'] not in seen:
            unique[i] = True
            seen.add(row['pixel_sha256'])
    future_selected, _ = build_clean_selection(records)
    future_paths = {r['relative_path'] for r in future_selected if r['split'] == 'val'}
    return {
        'original_val': np.ones(len(validation), dtype=bool),
        'technical_valid': valid,
        'technical_valid_no_exact_train_overlap': valid & ~byte_overlap,
        'technical_valid_no_pixel_train_overlap': valid & ~pixel_overlap,
        'technical_valid_no_pixel_train_overlap_unique_val': unique,
        'technical_valid_pixel_train_overlap_only': valid & pixel_overlap,
        'future_clean_val_existing_models': np.array([r['relative_path'] in future_paths for r in validation]),
    }
