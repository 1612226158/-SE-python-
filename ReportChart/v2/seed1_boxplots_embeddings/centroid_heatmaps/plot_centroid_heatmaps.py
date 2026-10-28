"""复用同样本seed1特征，绘制降维前类别中心余弦相似度。"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent
SRC = OUT.parents[3]
sys.path.insert(0, str(SRC))
import numpy as np
from ReportChart.paper_style import COLORS, canvas, configure, save


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def unit(v):
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError('存在零范数向量，不能定义余弦方向')
    return v/norms


def heatmap(matrix, xlabels, ylabels, title, name):
    fig, ax = canvas()
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=25, ha='right')
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i,j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(value)>.65 else '#222222')
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label='类别中心余弦相似度')
    save(fig, OUT, name)


def main():
    configure()
    manifest_path = SOURCE/'image_sample_manifest.json'
    metadata_path = SOURCE/'image_feature_analysis.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    labels = manifest['classes']
    records = manifest['records']
    truth = np.array([r['class_index'] for r in records])
    class_ids = [next(r['class_index'] for r in records if r['short_name']==c) for c in labels]
    assert len(labels)==8 and len(records)==240 and len(set(class_ids))==8
    assert all(np.sum(truth==c)==30 for c in class_ids)
    assert metadata['manifest_fingerprint']==manifest['fingerprint']
    models = list(COLORS)
    report = dict(created=datetime.now().astimezone().isoformat(), classes=labels, class_indices=class_ids,
                  n_per_class=30, feature_dimension=512,
                  formula='u_i=x_i/||x_i||₂; m_c=mean(u_i for true_class=c); p_c=m_c/||m_c||₂; S_cd=p_c·p_d',
                  source_layer='parent_classifier输入512维；使用原始缓存，不使用PCA/t-SNE坐标',
                  source_manifest=dict(path=str(manifest_path), sha256=sha(manifest_path),fingerprint=manifest['fingerprint']),
                  source_metadata=dict(path=str(metadata_path),sha256=sha(metadata_path)),
                  color_scale=dict(cmap='RdBu_r',vmin=-1,vmax=1), models={},
                  limitations=['固定8类、每类30张验证子集，不代表252类全体。',
                               '类中心只概括平均方向，不反映全部类内离散程度或多峰结构。',
                               '中心相似度不是混淆矩阵，也不是错误率或泛化性能。',
                               '模型各自在自身512维空间内算类间相似度，不跨模型直接相乘中心。'])
    names, pair_rows = [], []
    for model in models:
        source = SOURCE/f'features_{model}_seed1.npz'
        info = metadata['models'][model]
        with np.load(source,allow_pickle=False) as z:
            assert str(z['fingerprint'])==manifest['fingerprint']
            assert str(z['checkpoint_sha256'])==info['checkpoint_sha256']
            assert int(z['best_epoch'])==info['best_epoch']
            np.testing.assert_array_equal(z['truth'],truth)
            features = z['features'].astype(np.float64)
        assert features.shape==(240,512) and np.isfinite(features).all()
        vectors = unit(features)
        centers = np.stack([vectors[truth==c].mean(axis=0) for c in class_ids])
        directions = unit(centers)
        similarities = np.clip(directions@directions.T,-1,1)
        np.testing.assert_allclose(similarities,similarities.T,atol=1e-12)
        np.testing.assert_allclose(np.diag(similarities),1,atol=1e-12)
        name = f'centroid_similarity_{model}'
        heatmap(similarities, labels, labels, f'{model} 类别中心相似度（seed1 best，8类240图）',name)
        names.append(name)
        pair_rows.append([float(similarities[i,i+1]) for i in (0,2,4,6)])
        report['models'][model]=dict(source_npz=str(source),source_npz_sha256=sha(source),
            checkpoint_sha256=info['checkpoint_sha256'],best_epoch=info['best_epoch'],
            mean_centers=centers.tolist(),unit_centers=directions.tolist(),
            mean_direction_lengths=np.linalg.norm(centers,axis=1).tolist(),similarity_matrix=similarities.tolist())
    pair_names = [f'{labels[i]}/{labels[i+1]}' for i in (0,2,4,6)]
    pairs = np.array(pair_rows)
    heatmap(pairs,pair_names,models,'七模型易混淆类对的中心相似度（seed1 best）','centroid_pair_comparison')
    names.append('centroid_pair_comparison')
    report['pair_comparison']=dict(pairs=pair_names,model_order=models,values=pairs.tolist())
    import pymupdf
    with pymupdf.open() as atlas:
        for name in names:
            with pymupdf.open(OUT/f'{name}.pdf') as page:
                atlas.insert_pdf(page)
        atlas.set_toc([[1,name,i+1] for i,name in enumerate(names)])
        atlas.save(OUT/'class_centroid_heatmaps_atlas.pdf',deflate=True)
    report['outputs']=[dict(path=str(OUT/f'{name}.{ext}'),sha256=sha(OUT/f'{name}.{ext}'))
                       for name in names+['class_centroid_heatmaps_atlas']
                       for ext in (['pdf'] if name=='class_centroid_heatmaps_atlas' else ['pdf','png'])]
    (OUT/'centroid_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report['pair_comparison'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
