"""Diagnose existing bottom-class predictions through the shared cache API."""
import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
FULL = OUT.parent
ROOT = FULL.parents[3]
sys.path.insert(0, str(ROOT / 'src_v2/ReportChart'))
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from report_inference import load_prediction_cache, evaluator, sha256
from paper_style import COLORS


def short(name):
    return name.split('-', 2)[-1]


def profile(cmrow, index, names):
    counts = np.asarray(cmrow, dtype=np.int64)
    n = int(counts.sum()); tp = int(counts[index]); errors = n - tp
    off = counts.copy(); off[index] = 0
    order = np.argsort(-off, kind='stable')
    targets = [{'class_index': int(j), 'class_name': names[j], 'count': int(off[j]),
                'share_of_errors': float(off[j] / errors)} for j in order[:5] if off[j]] if errors else []
    parent = names[index].split('-', 2)[1]
    cross = sum(int(off[j]) for j in range(len(names)) if names[j].split('-', 2)[1] != parent)
    return {'prediction_instances': n, 'correct': tp, 'errors': errors, 'recall': tp/n,
            'wrong_target_count': int(np.count_nonzero(off)), 'cross_parent_errors': cross,
            'top1_error_share': targets[0]['share_of_errors'] if targets else 0.,
            'top3_error_share': sum(t['share_of_errors'] for t in targets[:3]), 'top_targets': targets}


def report(data):
    notes_path = OUT / 'interpretation.json'
    notes = json.loads(notes_path.read_text(encoding='utf-8')) if notes_path.exists() else {}
    lines = ['# 固定后20类：误判去向与识别诊断', '',
        '主比较集使用S-NoProg双seed平均召回率排名233—252，七种配置使用同一组类别。每模型自有后20类另见own_bottom20.csv。', '',
        '统计来自既有14份最佳验证模型预测缓存，复用公共report_inference.load_prediction_cache；没有重新训练或推理。', '',
        '“错误占比”的分母是该真实类别的误判次数；跨模型汇总为同一批图像的14次重复预测。验证图像数仍为每类原始support，不把重复预测当成新增独立图像。', '',
        '模式可以同时存在：近邻标签混淆、分散误判和输入/标注线索并非互斥类别。1×1纯白图可确认无有效视觉内容；其余语义问题列为人工复核候选，未据模型预测更改标签。', '',
        '## 类别总览', '', '| 基线排名 | 类别 | 验证图数 | 主要错误去向（占错误比例） | 观察与检查线索 |', '|---:|---|---:|---|---|']
    for row in data['classes']:
        targets='；'.join(f'{short(t["class_name"])} {100*t["share_of_errors"]:.1f}%' for t in row['pooled']['top_targets'][:3])
        note=notes.get(str(row['class_index']), {}).get('summary', '待结合样本检查')
        lines.append(f'| {row["rank"]} | {short(row["class_name"])} | {row["support"]} | {targets} | {note} |')
    summary=[]
    for row in data['classes']:
        info=notes.get(str(row['class_index']),{})
        summary.append({'rank':row['rank'],'class_index':row['class_index'],'class_name':row['class_name'],
                        'validation_images':row['support'],'training_images':row['training_images'],
                        'pooled_recall':row['pooled']['recall'],'error_top1_share':row['pooled']['top1_error_share'],
                        'main_wrong_targets':'; '.join(f'{short(t["class_name"])}:{t["count"]}' for t in row['pooled']['top_targets']),
                        'observed_pattern':info.get('summary',''),'review_detail':info.get('detail','')})
    pd.DataFrame(summary).to_csv(OUT/'class_diagnosis_summary.csv',index=False,encoding='utf-8-sig')
    lines += ['', '## 模型间召回率（%，seed1 / seed2）', '', '| 类别 | '+' | '.join(data['models'])+' |', '|---|'+'---:|'*len(data['models'])]
    for row in data['classes']:
        vals=[]
        for m in data['models']:
            runs=[d for d in data['runs'] if d['model']==m and d['class_index']==row['class_index']]
            vals.append(' / '.join(f'{100*d["recall"]:.1f}' for d in runs))
        lines.append('| '+short(row['class_name'])+' | '+' | '.join(vals)+' |')
    lines += ['', '## 逐类逐模型去向', '']
    for row in data['classes']:
        i=row['class_index'];info=notes.get(str(i), {})
        lines += [f'### {row["rank"]} {short(row["class_name"])}（n={row["support"]}）', '',
                  f'[全部验证输入裁剪](evidence/class_{i:03d}_crops.png) · [原图与裁剪对照](evidence/class_{i:03d}_original_crop.png)', '',
                  info.get('detail', ''), '', '| 配置 | 正确数s1/s2 | 主要错误去向（两seed合计；占错误比例） | 错误去向类数 |', '|---|---|---|---:|']
        for m in data['models']:
            pair=next(p for p in data['pairs'] if p['model']==m and p['class_index']==i)
            dest='；'.join(f'{short(t["class_name"])} {t["count"]}次/{100*t["share_of_errors"]:.1f}%' for t in pair['top_targets'][:3])
            lines.append(f'| {m} | {pair["correct_seed1"]}/{pair["correct_seed2"]} | {dest} | {pair["wrong_target_count"]} |')
    lines += ['', '## 输出完整性', '',
              f'- 20类包含{data["validation_images"]}张不同验证图，共{data["prediction_instances"]}次模型预测。',
              '- 检查14份缓存的类别顺序、真实标签、概率有限性/归一化、argmax与预测一致性及整数混淆计数，结果见diagnostics.json。',
              '- 每类训练图数由现有标签映射和训练目录统计；仅计既有文件，不改动划分。',
              f'- 无视觉内容/极小输入：{len(data.get("constant_or_tiny_images",[]))}张；跨标签相同文件内容的本组图像：{len(data["bottom20_cross_label_exact_duplicate_images"])}张，详细路径见diagnostics.json。',
              '- 图像目视观察用于定位多对象、外观多样、相近标签及裁剪差异，具体诊断见上表与interpretation.json。',
              '', '## 文件', '',
              '- run_class_profiles.csv：20类×14run的召回、Top-5命中、真实类分数、错误置信度与主要去向。',
              '- paired_model_class_profiles.csv：20类×7配置的两seed对照和错误集中度。',
              '- per_image_predictions.csv：每张图在14run下的预测、真实类排名与Top-1/Top-2分数。',
              '- image_quality.csv：455张原图的尺寸、像素方差与内容SHA256。',
              '- own_bottom20.csv：每模型按自身双seed平均召回率选出的后20类。',
              '- evidence/：验证集实际输入裁剪及原图/裁剪对照，页内编号对应image_index。',
              '- diagnostics.json：全部数值与完整性检查；diagnosis_manifest.json：来源/文件SHA256。', '']
    (OUT / '后20类误判诊断.md').write_text('\n'.join(lines), encoding='utf-8')


def key_evidence(manifest):
    indices=[7221,5930,4842,4843,7373,7378,7382,1534,1308,9101,8568,8576]
    predictions=pd.read_csv(OUT/'per_image_predictions.csv')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    canvas=Image.new('RGB',(1160,760),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,6),'左：原图    右：实际输入裁剪    预测为14次运行中的众数',font=font,fill='black')
    transform=evaluator._transform(manifest['mean'],manifest['std'])
    for k,index in enumerate(indices):
        rec=manifest['records'][index]
        x=(k%4)*290;y=35+(k//4)*240
        with Image.open(Path(evaluator.VAL_ROOT)/rec['relative_path']) as source:
            rgb=source.convert('RGB');crop=transform.transforms[1](transform.transforms[0](rgb))
            original=ImageOps.contain(rgb,(138,138));canvas.paste(original,(x+(138-original.width)//2,y+(138-original.height)//2))
            canvas.paste(crop.resize((138,138)),(x+145,y))
            dims=f'{source.width}×{source.height}'
        subset=predictions[predictions.image_index==index];counts=subset.predicted_class.value_counts()
        draw.text((x+5,y+143),f'#{index} 标注：{short(manifest["names"][rec["true_index"]])}',font=font,fill='black')
        draw.text((x+5,y+165),f'预测：{short(counts.index[0])}（{counts.iloc[0]}/14）',font=font,fill='black')
        draw.text((x+5,y+187),f'原图尺寸：{dims}',font=small,fill='#555555')
        draw.rectangle((x,y,x+138,y+138),outline='#dddddd')
        draw.rectangle((x+145,y,x+283,y+138),outline='#dddddd')
    canvas.save(OUT/'关键样本对照.png')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true');args=parser.parse_args()
    if args.report_only:
        data=json.loads((OUT/'diagnostics.json').read_text(encoding='utf-8'));report(data)
    else:
        manifest=json.loads((FULL/'validation_manifest.json').read_text(encoding='utf-8'))
        names=manifest['names'];models=list(COLORS);n=len(names)
        ranking=pd.read_csv(FULL/'fixed_baseline_class_ranking.csv').tail(20)
        chosen=ranking.class_index.tolist(); truth=np.array([r['true_index'] for r in manifest['records']])
        selected=np.flatnonzero(np.isin(truth,chosen)); runs=[];pairs=[];images=[];own=[];cms={};checks=[]
        for model in models:
            for seed in (1,2):
                d,identity=load_prediction_cache(model,seed,FULL/'predictions',manifest)
                prob=d['probabilities'];pred=d['predictions'];cm=d['cm']
                assert np.array_equal(d['truth'],truth)
                assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
                np.testing.assert_allclose(prob.sum(1),1,atol=1e-5)
                assert np.array_equal(prob.argmax(1),pred)
                reconstructed=np.bincount(truth*n+pred,minlength=n*n).reshape(n,n)
                assert np.array_equal(reconstructed,cm)
                cms[model,seed]=cm.copy()
                checks.append({'model':model,'seed':seed,'finite':True,'probability_normalized':True,'cm_matches_predictions':True,'checkpoint_sha256':identity['checkpoint_sha256']})
                for row in ranking.itertuples():
                    i=row.class_index;ix=np.flatnonzero(truth==i);p=prob[ix];pr=pred[ix];wrong=pr!=i
                    positions=1+(p>p[:,i,None]).sum(1)
                    pf=profile(cm[i],i,names)
                    pf.update({'model':model,'seed':seed,'class_index':i,'class_name':names[i],'rank':row.rank,
                               'support':int(len(ix)),'top5_recall':float((positions<=5).mean()),'mean_true_class_rank':float(positions.mean()),
                               'mean_true_probability':float(p[:,i].mean()),'mean_wrong_confidence':float(p.max(1)[wrong].mean()) if wrong.any() else None,
                               'wrong_confidence_ge_08':int((p.max(1)[wrong]>=.8).sum())})
                    runs.append(pf)
                    for k,idx in enumerate(ix):
                        top=np.argsort(-p[k],kind='stable')[:2]
                        images.append({'model':model,'seed':seed,'image_index':int(idx),'relative_path':manifest['records'][int(idx)]['relative_path'],
                                       'class_index':i,'true_class':names[i],'predicted_index':int(pr[k]),'predicted_class':names[int(pr[k])],
                                       'correct':bool(pr[k]==i),'confidence':float(p[k,top[0]]),'second_class':names[int(top[1])],
                                       'second_probability':float(p[k,top[1]]),'true_probability':float(p[k,i]),'true_class_rank':int(positions[k])})
            support=cms[model,1].sum(1);rec=(np.diag(cms[model,1])+np.diag(cms[model,2]))/(2*support)
            order=np.lexsort((np.arange(n),-support,-rec))
            for rank,idx in enumerate(order[-20:],233):own.append({'model':model,'rank_within_model':rank,'class_index':int(idx),'class_name':names[int(idx)],'support':int(support[idx]),'mean_recall':float(rec[idx]),'in_fixed_bottom20':int(idx)in chosen})
            for i in chosen:
                pf=profile(cms[model,1][i]+cms[model,2][i],i,names)
                pf.update({'model':model,'class_index':i,'class_name':names[i],'correct_seed1':int(cms[model,1][i,i]),'correct_seed2':int(cms[model,2][i,i])});pairs.append(pf)
        fid_to_class,renew,_=evaluator.build_truth_map();train_counts=collections.Counter()
        for folder in Path(evaluator.TRAIN_ROOT).iterdir():
            if folder.is_dir() and folder.name.isdigit() and int(folder.name) in fid_to_class:
                idx=int(renew[fid_to_class[int(folder.name)]])
                train_counts[idx]+=sum(p.is_file() and p.suffix.lower() in evaluator.VALID_EXTS for p in folder.iterdir())
        totals=sum(cms.values());classes=[];all_hashes=collections.defaultdict(list)
        for record in manifest['records']:all_hashes[record['sha256']].append(record['true_index'])
        cross_label_duplicates=[]
        for idx in selected:
            r=manifest['records'][int(idx)]
            if len(set(all_hashes[r['sha256']]))>1:cross_label_duplicates.append(int(idx))
        for row in ranking.itertuples():
            i=row.class_index
            classes.append({'rank':row.rank,'class_index':i,'class_name':names[i],'support':row.support,
                            'training_images':int(train_counts[i]),'pooled':profile(totals[i],i,names),
                            'image_indices':np.flatnonzero(truth==i).tolist()})
        quality=[]
        for pos in selected:
            record=manifest['records'][int(pos)]
            with Image.open(Path(evaluator.VAL_ROOT)/record['relative_path']) as image:
                pixels=np.asarray(image.convert('RGB'))
                quality.append({'image_index':int(pos),'relative_path':record['relative_path'],'true_class':names[record['true_index']],
                                'width':image.width,'height':image.height,'pixel_std':float(pixels.std()),
                                'constant_image':bool(pixels.min()==pixels.max()),'sha256':record['sha256']})
        pd.DataFrame(quality).to_csv(OUT/'image_quality.csv',index=False,encoding='utf-8-sig')
        duplicates=[]
        for pos in cross_label_duplicates:
            rec=manifest['records'][pos]
            duplicates.append({'image_index':pos,'sha256':rec['sha256'],
                'matching_records':[dict(image_index=k,relative_path=r['relative_path'],true_class=names[r['true_index']]) for k,r in enumerate(manifest['records']) if r['sha256']==rec['sha256']]})
        data={'models':models,'fixed_baseline':'S-NoProg','rank_range':[233,252],'classes':classes,'runs':runs,'pairs':pairs,
              'validation_images':int(len(selected)),'prediction_instances':int(len(images)),
              'manifest_fingerprint':manifest['fingerprint'],'integrity_checks':checks,
              'bottom20_cross_label_exact_duplicate_images':cross_label_duplicates,
              'cross_label_duplicate_details':duplicates,'constant_or_tiny_images':[q for q in quality if q['constant_image'] or min(q['width'],q['height'])<8]}
        (OUT/'diagnostics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        for values,name in [(runs,'run_class_profiles.csv'),(pairs,'paired_model_class_profiles.csv')]:
            flattened=[]
            for value in values:
                v={k:x for k,x in value.items() if k!='top_targets'}
                v['top_targets']='; '.join(f'{short(t["class_name"])}:{t["count"]}' for t in value['top_targets']);flattened.append(v)
            pd.DataFrame(flattened).to_csv(OUT/name,index=False,encoding='utf-8-sig')
        pd.DataFrame(images).to_csv(OUT/'per_image_predictions.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(own).to_csv(OUT/'own_bottom20.csv',index=False,encoding='utf-8-sig')
        evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
        transform=evaluator._transform(manifest['mean'],manifest['std'])
        for row in classes:
            idx=row['class_index'];positions=row['image_indices'];cols=6;cell_w=150;cell_h=175
            sheet=Image.new('RGB',(cols*cell_w,50+math.ceil(len(positions)/cols)*cell_h),'white');draw=ImageDraw.Draw(sheet)
            draw.text((10,8),f'{row["rank"]} {short(row["class_name"])}  n={len(positions)}  输入裁剪',font=font,fill='black')
            # Original/crop pairs cover all selected validation samples as traceable evidence.
            pair_sheet=Image.new('RGB',(4*290,50+math.ceil(len(positions)/4)*180),'white');pair_draw=ImageDraw.Draw(pair_sheet)
            pair_draw.text((10,8),f'{short(row["class_name"])}：左为原图，右为模型输入',font=font,fill='black')
            for k,pos in enumerate(positions):
                record=manifest['records'][pos];path=Path(evaluator.VAL_ROOT)/record['relative_path']
                assert sha256(path)==record['sha256']
                with Image.open(path) as source:
                    # Recreate the actual loader path, which does not apply EXIF transpose.
                    actual=source.convert('RGB');crop=transform.transforms[1](transform.transforms[0](actual))
                    x=(k%cols)*cell_w;y=50+(k//cols)*cell_h
                    sheet.paste(crop.resize((140,140)),(x+5,y));draw.text((x+5,y+143),f'#{pos}',font=small,fill='black')
                    x=(k%4)*290;y=50+(k//4)*180
                    original=ImageOps.contain(actual,(140,140));pair_sheet.paste(original,(x+(140-original.width)//2,y+(140-original.height)//2))
                    pair_sheet.paste(crop.resize((140,140)),(x+145,y));pair_draw.text((x+5,y+143),f'#{pos}',font=small,fill='black')
            sheet.save(evidence/f'class_{idx:03d}_crops.png');pair_sheet.save(evidence/f'class_{idx:03d}_original_crop.png')
        report(data)
        print('DIAGNOSED',len(selected),'images',len(images),'predictions',len(runs),'class/run rows')
    key_evidence(json.loads((FULL/'validation_manifest.json').read_text(encoding='utf-8')))
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='diagnosis_manifest.json']
    (OUT/'diagnosis_manifest.json').write_text(json.dumps({'source_manifest_sha256':sha256(FULL/'validation_manifest.json'),
        'cache_api':'report_inference.load_prediction_cache','files':[{'path':p.relative_to(OUT).as_posix(),'sha256':sha256(p)} for p in sorted(files)]},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
