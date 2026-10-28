"""七组seed1描述性箱线图及同样本CPU特征降维；不写训练目录。"""
import gc
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
SRC = OUT.parents[2]
for directory in (SRC.parent, SRC, SRC / 'ReportChart'):
    sys.path.insert(0, str(directory))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize
from ReportChart.chart_data import load_model, summarize_run
from ReportChart.paper_style import COLORS, canvas, configure, save

MODELS = list(COLORS)
RANDOM_SEED = 20261005
CLASSES = ['锅', '电饭煲', '盒子', '纸箱', '鞋', '拖鞋', '水杯', '保温杯']
CLASS_COLORS = ['#3C6EAB', '#D78A35', '#2A8C74', '#C75664', '#7656A0', '#42A3B5', '#B29D32', '#737373']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def metrics_plots():
    snapshot = json.loads((OUT.parent / 'seed1_summary/seed1_results_summary.json').read_text(encoding='utf-8'))
    frames = {}
    for model in MODELS:
        df = load_model(model, seed=1, verbose=False)
        assert df.epoch.astype(int).tolist() == list(range(100)), model
        old = snapshot['runs'][model]['metrics_all_epochs']
        for key in ['train_acc', 'train_loss', 'val_acc', 'val_loss', 'val_macro_f1', 'time_spend_s']:
            # 固定分析快照序列化保留有限小数；容许纳数量级的表示舍入。
            np.testing.assert_allclose(df[key], [row[key] for row in old], rtol=1e-9, atol=1e-9)
        frames[model] = df
    specs = [
        ('01_accuracy_all', '全程验证准确率分布（epoch 0—99）', '验证准确率（%）', 'val_acc', 1, 'all'),
        ('02_accuracy_last25', '末25轮验证准确率分布（epoch 75—99）', '验证准确率（%）', 'val_acc', 1, 'late'),
        ('03_macro_f1_last25', '末25轮验证 Macro-F1 分布（epoch 75—99）', 'Macro-F1（%）', 'val_macro_f1', 100, 'late'),
        ('04_loss_last25', '末25轮验证损失分布（epoch 75—99）', '验证损失', 'val_loss', 1, 'late'),
        ('05_time_all', '全程单轮耗时分布（epoch 0—99）', '单轮耗时（分钟）', 'time_spend_s', 1/60, 'all'),
        ('06_time_common99', '共同99轮单轮耗时分布（排除 epoch 63）', '单轮耗时（分钟）', 'time_spend_s', 1/60, 'common99'),
    ]
    statistics = {}
    for name, title, label, column, scale, scope in specs:
        data, detail = [], {}
        for model, df in frames.items():
            sub = df[df.epoch >= 75] if scope == 'late' else df[df.epoch != 63] if scope == 'common99' else df
            values = sub[column].to_numpy(float) * scale
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            iqr = q3 - q1
            outside = (values < q1-1.5*iqr) | (values > q3+1.5*iqr)
            detail[model] = dict(n=len(values), median=med, q1=q1, q3=q3, iqr=iqr,
                                 whisker_min=float(values[~outside].min()), whisker_max=float(values[~outside].max()),
                                 fliers=[dict(epoch=int(e), value=float(v)) for e,v in zip(sub.epoch[outside], values[outside])])
            data.append(values)
        fig, ax = canvas()
        boxes = ax.boxplot(data, labels=MODELS, patch_artist=True, whis=1.5,
                           medianprops=dict(color='#222222', linewidth=1.4),
                           flierprops=dict(marker='o', markersize=3, alpha=.6))
        for box, model in zip(boxes['boxes'], MODELS):
            box.set_facecolor(COLORS[model]); box.set_alpha(.7)
        ax.set_title(title); ax.set_ylabel(label); ax.tick_params(axis='x', labelrotation=25)
        ax.grid(axis='y'); save(fig, OUT, name)
        statistics[name] = dict(column=column, scale=scale, scope=scope, models=detail)
    columns = ['train_acc', 'train_loss', 'val_acc', 'val_loss', 'val_macro_f1']
    raw = np.concatenate([frames[m][columns].to_numpy(float) for m in MODELS])
    scaler = StandardScaler(); standardized = scaler.fit_transform(raw)
    pca = PCA(n_components=2, svd_solver='full'); points = pca.fit_transform(standardized)
    fig, ax = canvas()
    for i, model in enumerate(MODELS):
        xy = points[i*100:(i+1)*100]
        ax.plot(xy[:,0], xy[:,1], color=COLORS[model], alpha=.35, linewidth=.8)
        ax.scatter(xy[:,0], xy[:,1], color=COLORS[model], s=12, alpha=.7, label=model)
        ax.scatter(*xy[-1], color=COLORS[model], marker='X', s=55)
    ax.set_title('七组模型训练状态 PCA（seed1，epoch 0—99）')
    ax.set_xlabel(f'主成分1（{pca.explained_variance_ratio_[0]*100:.1f}%）')
    ax.set_ylabel(f'主成分2（{pca.explained_variance_ratio_[1]*100:.1f}%）')
    ax.legend(ncol=2); ax.grid(); save(fig, OUT, '07_training_state_pca')
    write_json('metrics_analysis.json', dict(read_time=datetime.now().astimezone().isoformat(),
        reader='ReportChart.chart_data.load_model(config_id,seed=1)', snapshot_match=True,
        snapshot_path=str(OUT.parent / 'seed1_summary/seed1_results_summary.json'),
        snapshot_sha256=sha(OUT.parent / 'seed1_summary/seed1_results_summary.json'),
        snapshot_numeric_tolerance=dict(rtol=1e-9, atol=1e-9),
        boxplots=statistics, pca=dict(features=columns, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
        components=pca.components_.tolist(), explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        coordinates={m:points[i*100:(i+1)*100].tolist() for i,m in enumerate(MODELS)}),
        limitations=['同seed连续epoch并非独立重复实验。', '训练状态PCA不是图像特征分布。',
                     '共同99轮仅作耗时敏感性分析，全100轮保留所有原始记录。']))
    print('METRICS_PLOTS_COMPLETE', flush=True)


def image_embeddings():
    import torch
    import eval_confusion_matrix as evaluator
    from PIL import Image
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    from calculate import _ORIGINAL_SRC_DIR
    cache = Path(_ORIGINAL_SRC_DIR) / 'calculate.config'
    if not cache.exists():
        raise RuntimeError('缺少均值缓存，禁止本分析隐式全量重算')
    mean, std, _ = evaluator.calculate_mean_and_std(evaluator.TRAIN_ROOT, retrieve_file=True, update=False)
    tf = evaluator._transform(mean, std)
    fid2cls, renew, n_cls = evaluator.build_truth_map()
    assert n_cls == 252
    selected = [next(k for k in renew if k.split('-')[-1] == c) for c in CLASSES]
    rng = np.random.RandomState(RANDOM_SEED)
    records = []
    for short, full in zip(CLASSES, selected):
        candidates = []
        for fid in evaluator.val_folders():
            if fid2cls.get(fid) == full:
                folder = Path(evaluator.VAL_ROOT) / str(fid)
                candidates.extend(p for p in sorted(folder.iterdir()) if p.suffix.lower() in evaluator.VALID_EXTS)
        if len(candidates) < 30:
            raise RuntimeError(f'{short}不足30张验证图片')
        for j in rng.permutation(len(candidates))[:30]:
            p = candidates[j]
            records.append(dict(path=str(p), sha256=sha(p), class_name=full, short_name=short, class_index=int(renew[full])))
    manifest = dict(seed=RANDOM_SEED, classes=CLASSES, n_per_class=30, records=records,
                    preprocess='Resize256/CenterCrop224/ToTensor/Normalize', normalization_cache_sha256=sha(cache))
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    manifest['fingerprint'] = fingerprint
    write_json('image_sample_manifest.json', manifest)
    images = []
    for rec in records:
        with Image.open(rec['path']) as im:
            images.append(tf(im.convert('RGB')))
    images = torch.stack(images)
    truth = np.array([r['class_index'] for r in records])
    summary = {}
    for model in MODELS:
        started = datetime.now().astimezone().isoformat()
        ckpath = SRC / 'runs' / f'checkpoint_{model}_seed1_best.pth'
        checkpoint_sha = sha(ckpath)
        feature_file = OUT / f'features_{model}_seed1.npz'
        reuse = False
        if feature_file.exists():
            with np.load(feature_file, allow_pickle=False) as z:
                reuse = str(z['fingerprint']) == fingerprint and str(z['checkpoint_sha256']) == checkpoint_sha
                if reuse:
                    features = z['features'].copy(); predictions = z['predictions'].copy(); best_epoch = int(z['best_epoch'])
        if not reuse:
            # 公共加载器仍负责架构/参数加载，仅在本进程显式指定seed1路径。
            with patch.object(evaluator.cd, 'best_ckpt_path', return_value=str(ckpath)):
                net, ck, loaded_renew, _, loaded_path = evaluator.build_and_load(model)
            report = summarize_run(model, 1)
            for key, expected in [('config_id', model), ('seed', 1), ('num_classes', 252),
                                  ('training_protocol_version', 2), ('label_mapping_hash', report['protocol']['label_mapping_hash'])]:
                if ck.get(key) != expected:
                    raise RuntimeError(f'{model} checkpoint身份不符：{key}')
            best_epoch = int(ck['epoch'])
            assert best_epoch == int(report['official_best_val_acc']['epoch']) and loaded_renew == renew
            del ck; gc.collect()
            captured = []
            hook = net.parent_classifier.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().cpu().numpy().copy()))
            preds = []
            with torch.inference_mode():
                for offset in range(0, len(images), 8):
                    logits = net(images[offset:offset+8])
                    preds.extend(logits.argmax(1).tolist())
            hook.remove()
            features = np.concatenate(captured); predictions = np.asarray(preds)
            assert features.shape == (240, 512) and np.isfinite(features).all()
            np.savez_compressed(feature_file, features=features, predictions=predictions, truth=truth,
                                fingerprint=fingerprint, checkpoint_sha256=checkpoint_sha, best_epoch=best_epoch)
            del net, captured; gc.collect()
        normalized = normalize(features, norm='l2')
        pca = PCA(n_components=2, svd_solver='full'); xy_pca = pca.fit_transform(normalized)
        reduced = PCA(n_components=50, svd_solver='full').fit_transform(normalized)
        tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto',
                    n_iter=1000, random_state=RANDOM_SEED, n_jobs=2)
        xy_tsne = tsne.fit_transform(reduced)
        for method, xy in [('PCA', xy_pca), ('t-SNE', xy_tsne)]:
            fig, ax = canvas()
            for short, full, color in zip(CLASSES, selected, CLASS_COLORS):
                mask = truth == int(renew[full])
                ax.scatter(xy[mask,0], xy[mask,1], s=20, alpha=.8, color=color, label=short)
            ax.set_title(f'{model} 图像特征 {method}（seed1 best，8类240图）')
            if method == 'PCA':
                ax.set_xlabel(f'主成分1（{pca.explained_variance_ratio_[0]*100:.1f}%）')
                ax.set_ylabel(f'主成分2（{pca.explained_variance_ratio_[1]*100:.1f}%）')
            else:
                ax.set_xlabel('t-SNE 维度1'); ax.set_ylabel('t-SNE 维度2')
            ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5,-.17), frameon=False)
            ax.grid(); save(fig, OUT, f'feature_{model}_{method.lower()}')
        summary[model] = dict(checkpoint_path=str(ckpath), checkpoint_sha256=checkpoint_sha, best_epoch=best_epoch,
                              device='cpu', threads=2, batch_size=8, feature_dimension=512, cache_reused=reuse,
                              started=started, completed=datetime.now().astimezone().isoformat(),
                              subset_accuracy_percent=float((predictions==truth).mean()*100),
                              pca_explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
                              pca_coordinates=xy_pca.tolist(), tsne_coordinates=xy_tsne.tolist(),
                              tsne_kl_divergence=float(tsne.kl_divergence_))
        write_json('image_feature_analysis.json', dict(models=summary, manifest_fingerprint=fingerprint,
            feature_layer='parent_classifier输入，512维；CPU FP32 eval', sample_n=240,
            parameters=dict(l2_normalize=True, pca_per_model=True, tsne_pre_pca_components=50,
                            perplexity=30, n_iter=1000, init='pca', learning_rate='auto', random_state=RANDOM_SEED),
            limitations=['仅8个预选类别的固定验证子集，不代表252类全体或独立测试。',
                        '不同模型独立拟合投影；图上坐标、旋转、簇间全局距离不可跨模型直接比较。',
                        't-SNE的二维可分程度不是泛化或稳健性的证明。']))
        print(f'FEATURE_COMPLETE {model} best_epoch={best_epoch} n={len(features)}', flush=True)


def build_atlas():
    import pymupdf
    names = ['01_accuracy_all', '02_accuracy_last25', '03_macro_f1_last25',
             '04_loss_last25', '05_time_all', '06_time_common99', '07_training_state_pca']
    names += [f'feature_{model}_{method}' for model in MODELS for method in ['pca', 't-sne']]
    with pymupdf.open() as atlas:
        for name in names:
            with pymupdf.open(OUT / f'{name}.pdf') as page:
                atlas.insert_pdf(page)
        atlas.set_toc([[1, name, i+1] for i,name in enumerate(names)])
        atlas.save(OUT / 'seed1_comparison_atlas.pdf', deflate=True)


if __name__ == '__main__':
    configure()
    metrics_plots()
    image_embeddings()
    build_atlas()
