"""Reusable explicit-seed validation, prediction cache and Grad-CAM for reports.

Training/model source files and runs/ are read-only. Uses the established
eval_confusion_matrix mapping, image transform, model loader and class metrics.
"""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import eval_confusion_matrix as evaluator
from chart_data import best_ckpt_path, summarize_run


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validation_manifest():
    """Frozen content-hashed validation order and cached training normalization."""
    from calculate import _ORIGINAL_SRC_DIR
    cache = Path(_ORIGINAL_SRC_DIR) / 'calculate.config'
    if not cache.is_file():
        raise FileNotFoundError('Normalization cache required; implicit recomputation forbidden')
    mean, std, _ = evaluator.calculate_mean_and_std(evaluator.TRAIN_ROOT, retrieve_file=True, update=False)
    fid_to_class, renew, count = evaluator.build_truth_map()
    records = []
    for fid in evaluator.val_folders():
        if fid not in fid_to_class:
            raise ValueError(f'Unmapped validation folder {fid}')
        for path in sorted((Path(evaluator.VAL_ROOT) / str(fid)).iterdir()):
            if path.suffix.lower() in evaluator.VALID_EXTS:
                records.append({'relative_path': path.relative_to(evaluator.VAL_ROOT).as_posix(),
                                'true_index': int(renew[fid_to_class[fid]]),
                                'sha256': sha256(path)})
    manifest = {'records': records, 'names': [n for n, i in sorted(renew.items(), key=lambda x:int(x[1]))],
                'num_classes': count, 'num_samples': len(records),
                'normalization_sha256': sha256(cache),
                'mean': mean.tolist(), 'std': std.tolist(),
                'preprocess': 'RGB, Resize(256 bilinear), CenterCrop(224), ToTensor, cached Normalize'}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return manifest


def preload_validation_rgb(manifest):
    """Prepare validation crops once in RAM as uint8 (~2.2 GB), never a disk tensor cache."""
    transform = evaluator._transform(manifest['mean'], manifest['std'])
    images = np.empty((manifest['num_samples'], 224, 224, 3), dtype=np.uint8)
    for index, record in enumerate(manifest['records']):
        with Image.open(Path(evaluator.VAL_ROOT) / record['relative_path']) as image:
            crop = transform.transforms[1](transform.transforms[0](image.convert('RGB')))
            images[index] = np.asarray(crop)
        if (index + 1) % 2000 == 0:
            print(f'PREPROCESS {index+1}/{len(images)}', flush=True)
    return images


def normalized_batch(rgb, mean, std, device):
    # Match training's contiguous NCHW tensors. Legacy SE uses view() after pooling.
    array = np.ascontiguousarray(rgb)
    if not array.flags.writeable:
        array = array.copy()
    batch = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().to(device=device, dtype=torch.float32) / 255.
    return (batch - torch.tensor(mean, device=device)[None,:,None,None]) / torch.tensor(std, device=device)[None,:,None,None]


def load_prediction_cache(config_id, seed, output_dir, manifest=None):
    """Shared cached-prediction reader with model/seed and optional dataset identity checks."""
    path = Path(output_dir) / f'predictions_{config_id}_seed{int(seed)}.npz'
    with np.load(path, allow_pickle=False) as cached:
        identity = json.loads(str(cached['identity']))
        if identity['model'] != config_id or identity['seed'] != int(seed):
            raise ValueError(f'Prediction cache run mismatch: {path}')
        if manifest is not None and identity['manifest_fingerprint'] != manifest['fingerprint']:
            raise ValueError(f'Prediction cache dataset mismatch: {path}')
        data = {key:cached[key].copy() for key in ('truth','predictions','probabilities','cm')}
    return data, identity


def paired_prediction_statistics(truth, prediction1, prediction2, mask=None):
    """Count two-run correctness and label disagreement on identical samples.

    A correctness flip and a changed predicted label are distinct: two wrong
    predictions can also disagree. Counts retain the original sample denominator.
    """
    truth, first, second = (np.asarray(value) for value in (truth, prediction1, prediction2))
    if truth.ndim != 1 or first.shape != truth.shape or second.shape != truth.shape:
        raise ValueError('Paired predictions require equal one-dimensional arrays')
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != truth.shape:
            raise ValueError('Sample mask must be boolean and match truth')
        truth, first, second = truth[mask], first[mask], second[mask]
    if not len(truth):
        raise ValueError('Paired comparison requires at least one sample')
    a, b = first == truth, second == truth
    return {'n': len(truth), 'both_correct': int((a & b).sum()),
            'seed1_only_correct': int((a & ~b).sum()),
            'seed2_only_correct': int((~a & b).sum()),
            'one_correct': int((a ^ b).sum()), 'both_wrong': int((~a & ~b).sum()),
            'prediction_disagreement': int((first != second).sum()),
            'both_wrong_different_labels': int((~a & ~b & (first != second)).sum())}


def predict_validation(config_id, seed, manifest, output_dir, rgb=None, device='cuda', batch_size=32, amp=True):
    """Full validation logits/probabilities with fingerprinted, per-run resumable caches."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / f'predictions_{config_id}_seed{seed}.npz'
    checkpoint = Path(best_ckpt_path(config_id, seed=seed))
    checkpoint_sha = sha256(checkpoint)
    identity = {'model': config_id, 'seed': int(seed), 'manifest_fingerprint': manifest['fingerprint'],
                'checkpoint_sha256': checkpoint_sha, 'device': device, 'amp': bool(amp),
                'torch_version': torch.__version__, 'normalization_sha256': manifest['normalization_sha256']}
    if cache.exists():
        data, cached_identity = load_prediction_cache(config_id, seed, output_dir, manifest)
        if cached_identity == identity:
            print(f'REUSE {config_id} seed{seed}', flush=True)
            return data, identity
        raise ValueError(f'Cache identity mismatch; preserve and inspect {cache}')
    if rgb is None:
        raise ValueError('Uncached prediction requires shared RGB crops')
    net, ck, renew, idx2name, _ = evaluator.build_and_load(config_id, seed=seed)
    if manifest['names'] != [idx2name[i] for i in range(len(renew))]:
        raise ValueError('Validation manifest class names are not in numeric output-index order')
    identity['checkpoint_epoch'] = int(ck['epoch'])
    # Cache key excludes epoch (already encoded by checkpoint SHA); retain in meta instead.
    epoch = identity.pop('checkpoint_epoch')
    del ck; gc.collect()
    net = net.to(device).eval()
    truth = np.asarray([r['true_index'] for r in manifest['records']], dtype=np.int64)
    probabilities = np.empty((len(truth), len(renew)), dtype=np.float32)
    started = time.time()
    with torch.inference_mode():
        for offset in range(0, len(truth), batch_size):
            batch = normalized_batch(rgb[offset:offset+batch_size], manifest['mean'], manifest['std'], device)
            with torch.cuda.amp.autocast(enabled=amp and device.startswith('cuda')):
                logits = net(batch)
            probabilities[offset:offset+len(batch)] = torch.softmax(logits.float(), dim=1).cpu().numpy()
            if offset and offset % (batch_size * 80) == 0:
                print(f'INFER {config_id} seed{seed} {offset}/{len(truth)} {time.time()-started:.1f}s', flush=True)
    predictions = probabilities.argmax(1)
    cm = np.bincount(truth * len(renew) + predictions, minlength=len(renew)**2).reshape(len(renew), len(renew))
    np.savez_compressed(cache, truth=truth, predictions=predictions, probabilities=probabilities, cm=cm,
                        identity=json.dumps(identity, sort_keys=True), checkpoint_epoch=epoch,
                        inference_seconds=time.time()-started)
    summary = summarize_run(config_id, seed)
    meta = dict(identity, checkpoint_epoch=epoch, inference_seconds=time.time()-started,
                accuracy_percent=float(np.trace(cm)/cm.sum()*100),
                training_best_accuracy_percent=summary['official_best_val_acc']['value'],
                names=manifest['names'], num_samples=int(cm.sum()), cache_sha256=sha256(cache))
    (cache.with_suffix('.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    del net, batch, logits; gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()
    print(f'COMPLETE {config_id} seed{seed} {meta["accuracy_percent"]:.4f}% {meta["inference_seconds"]:.1f}s', flush=True)
    return {'truth':truth,'predictions':predictions,'probabilities':probabilities,'cm':cm}, identity


def grad_cam(net, image, target_index, layer=None):
    """True Grad-CAM: spatial mean target-logit gradients weight backbone layer4 maps."""
    target_layer = layer if layer is not None else net.resnet_backbone[-1]
    activations = []
    hook = target_layer.register_forward_hook(lambda module, args, output: activations.append(output))
    try:
        with torch.enable_grad():
            logits = net(image.detach().requires_grad_(True))
            gradient = torch.autograd.grad(logits[0, int(target_index)], activations[0])[0]
            weights = gradient.mean(dim=(2,3), keepdim=True)
            cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=image.shape[-2:], mode='bilinear', align_corners=False)[0,0]
            minimum, maximum = cam.min(), cam.max()
            cam = (cam-minimum) / (maximum-minimum).clamp_min(1e-12)
            return cam.detach().cpu().numpy(), torch.softmax(logits.detach().float(),1)[0].cpu().numpy()
    finally:
        hook.remove()


def calibration_statistics(truth, probabilities, bins=10):
    """Equal-width reliability bins, raw ECE/NLL/Brier, no fitted calibration."""
    prediction = probabilities.argmax(1); confidence = probabilities.max(1)
    correct = prediction == truth
    rows = []
    for index in range(bins):
        lo, hi = index/bins, (index+1)/bins
        mask = (confidence >= lo) & ((confidence <= hi) if index==bins-1 else (confidence < hi))
        rows.append({'lower':lo, 'upper':hi, 'count':int(mask.sum()),
                     'confidence':float(confidence[mask].mean()) if mask.any() else None,
                     'accuracy':float(correct[mask].mean()) if mask.any() else None})
    ece = sum(r['count']/len(truth)*abs(r['accuracy']-r['confidence']) for r in rows if r['count'])
    ptrue = probabilities[np.arange(len(truth)), truth]
    brier = np.mean((probabilities**2).sum(1) - 2*ptrue + 1)
    return {'bins':rows,'ece':float(ece), 'nll':float(-np.log(np.maximum(ptrue,1e-30)).mean()),
            'brier':float(brier),'mean_confidence':float(confidence.mean())}
