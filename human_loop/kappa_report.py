# -*- coding: utf-8 -*-
"""
kappa_report.py —— 标注一致性（Cohen's κ）与模型同批对照报告（2026-09-12 建，human_loop）

输入：make_kappa_pack.py 生成的抽样包（含 _key.csv 与两张标注表 sheetA/sheetB）
输出：控制台报告 + <pack>/kappa_report.csv + <pack>/kappa_report.md（可直接摘进论文）

报告内容（每个类对 + 汇总）：
  n / 有效标注数 / 一致率 p_o / 随机期望 p_e / **Cohen's κ** / 分级
  同一标注者轮次 A、B 各自的准确率（相对真实类别）
  --with-model 时附上252类协议v2模型在**同一批图片**上的准确率（不解释为“人类上限”）

用法：
  python kappa_report.py --pack exports/20260912_xxxxxx_kappa_pack
  python kappa_report.py --pack <pack> --with-model S-NoProG
  python kappa_report.py --selftest          # κ 计算自检（已知例：p_o=0.7, p_e=0.5 → κ=0.4）

纪律：只读图片与 CSV；不写回标注表；不改数据集。
"""
import os
import sys
import csv
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_common import (cohens_kappa, kappa_level, EXPORT_ROOT, LABEL_SCHEMA_ID,
                       LABEL_SCHEMA_VERSION, EXPECTED_NUM_CLASSES, short_name,
                       merged_class_index, current_protocol_identity,
                       validate_public_pack, sha256_file, VAL_ROOT, TRAIN_ROOT,
                       configure_utf8_console)  # noqa: E402

configure_utf8_console()


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def normalize_choice(raw, option_1, option_2):
    """标注值 → 规范化标签（option_1 / option_2 的短名），无法判定返回 None。

    接受：'1'/'2'（推荐）、短名本身、'0' 或空 = 弃权。
    """
    s = (raw or '').strip()
    if s in ('', '0', 'NA', 'na', '-', '弃权'):
        return None
    if s == '1':
        return option_1
    if s == '2':
        return option_2
    if s == option_1:
        return option_1
    if s == option_2:
        return option_2
    raise ValueError(f'非法 choice={raw!r}；只允许 0/1/2、空值或当前两个候选短名')


def collect(pack, sheet_name, key_by_image):
    """读并严格校验一轮标注表；只允许 choice/notes 改变。"""
    rows = read_csv(os.path.join(pack, sheet_name))
    if rows is None:
        return None
    out = {}
    for row_num, r in enumerate(rows, 2):
        image = r.get('image', '')
        if image in out:
            raise SystemExit(f'[hl] {sheet_name} 第 {row_num} 行重复 image：{image}')
        key = key_by_image.get(image)
        if key is None:
            raise SystemExit(f'[hl] {sheet_name} 含答案键之外的 image：{image}')
        for field in ('pair_id', 'image', 'option_1', 'option_2'):
            if r.get(field, '') != key.get(field, ''):
                raise SystemExit(
                    f'[hl] {sheet_name} 的不可变字段被修改：image={image}, field={field}, '
                    f'表={r.get(field)!r}, key={key.get(field)!r}')
        try:
            label = normalize_choice(r.get('choice'), r.get('option_1', ''), r.get('option_2', ''))
        except ValueError as exc:
            raise SystemExit(f'[hl] {sheet_name} 第 {row_num} 行：{exc}') from exc
        notes = r.get('notes', '') or ''
        outlier = '[图外]' in notes
        out[image] = {'label': label, 'outlier': outlier, 'notes': notes}
    missing = sorted(set(key_by_image) - set(out))
    if missing:
        raise SystemExit(f'[hl] {sheet_name} 缺少 {len(missing)} 行，例如 {missing[:3]}')
    return out


def validate_pack_files(pack, protocol):
    """把 protocol、pairs、答案键、图片集合锁成同一个252类抽样包。"""
    try:
        public = validate_public_pack(pack, protocol)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f'[hl] 抽样包公开部分校验失败：{exc}') from exc
    fid2cls, cls2fids, _renew = merged_class_index()
    pair_rows = public['pairs']
    key_rows = read_csv(os.path.join(pack, '_key.csv'))
    if key_rows is None:
        raise SystemExit('[hl] 抽样包缺少 _key.csv')
    if not key_rows:
        raise SystemExit('[hl] _key.csv 为空，拒绝生成报告')
    pair_map = {row['pair_id']: row for row in pair_rows}
    key_by_image = {}
    counts = {pid: {'A': 0, 'B': 0} for pid in pair_map}
    source_root = os.path.abspath(VAL_ROOT if protocol['source'] == 'val' else TRAIN_ROOT)
    for row_num, r in enumerate(key_rows, 2):
        image = r.get('image', '')
        if not image or image in key_by_image:
            raise SystemExit(f'[hl] _key.csv 第 {row_num} 行 image 为空或重复：{image!r}')
        pair = pair_map.get(r.get('pair_id'))
        if pair is None:
            raise SystemExit(f'[hl] _key.csv 含未知 pair_id：{r.get("pair_id")}')
        true_class = r.get('true_class')
        if true_class == pair['class_A']:
            counts[pair['pair_id']]['A'] += 1
        elif true_class == pair['class_B']:
            counts[pair['pair_id']]['B'] += 1
        else:
            raise SystemExit(f'[hl] {image} 的 true_class 不属于协议类对')
        if true_class not in cls2fids or r.get('true_short') != short_name(true_class):
            raise SystemExit(f'[hl] {image} 的真值不属于当前252类映射或短名不一致')
        if {r.get('option_1'), r.get('option_2')} != {pair['short_A'], pair['short_B']}:
            raise SystemExit(f'[hl] {image} 的候选类与协议类对不一致')
        image_path = os.path.abspath(os.path.join(pack, 'images', image.replace('/', os.sep)))
        image_root = os.path.abspath(os.path.join(pack, 'images'))
        if os.path.commonpath([image_root, image_path]) != image_root or not os.path.isfile(image_path):
            raise SystemExit(f'[hl] 答案键图片路径无效：{image}')
        copied_hash = sha256_file(image_path)
        if (r.get('image_sha256') != copied_hash or
                public['image_hashes'].get(image) != copied_hash):
            raise SystemExit(f'[hl] {image} 的复制图哈希与答案键/公开清单不一致')
        source_path = os.path.abspath(r.get('source_path', ''))
        try:
            inside_source = os.path.normcase(os.path.commonpath([source_root, source_path])) == os.path.normcase(source_root)
        except ValueError:
            inside_source = False
        if not inside_source or not os.path.isfile(source_path):
            raise SystemExit(f'[hl] {image} 的 source_path 不在协议声明的数据源目录内')
        rel_source = os.path.relpath(source_path, source_root)
        try:
            source_fid = int(rel_source.split(os.sep, 1)[0])
        except (TypeError, ValueError):
            raise SystemExit(f'[hl] {image} 无法从 source_path 解析原始目录ID')
        if fid2cls.get(source_fid) != true_class:
            raise SystemExit(f'[hl] {image} 的 source_path 目录语义与 true_class 不一致')
        source_hash = sha256_file(source_path)
        if r.get('source_sha256') != source_hash or source_hash != copied_hash:
            raise SystemExit(f'[hl] {image} 的源图/复制图内容哈希不一致')
        key_by_image[image] = r
    source_paths = [os.path.normcase(os.path.abspath(r.get('source_path', ''))) for r in key_rows]
    if protocol.get('global_source_paths_unique') is not True or len(set(source_paths)) != len(source_paths):
        raise SystemExit('[hl] 抽样包未满足整包源图路径全局唯一约束')
    for pid, pair in pair_map.items():
        if counts[pid]['A'] != int(pair['n_A']) or counts[pid]['B'] != int(pair['n_B']):
            raise SystemExit(f'[hl] {pid} 的答案键类计数与 pairs.csv 不一致')
    if set(key_by_image) != set(public['image_hashes']):
        raise SystemExit('[hl] 答案键图片集合与公开清单不一致')
    return key_rows, key_by_image


def validate_checkpoint_identity(checkpoint, expected=None, current_renew=None):
    """纯函数式校验模型checkpoint是否严格绑定当前252类协议；便于负向单测。"""
    if expected is None:
        expected = current_protocol_identity()
    if current_renew is None:
        _fid2, _cls2fids, current_renew = merged_class_index()
    checkpoint_mismatches = {}
    for field in ('label_schema_id', 'label_schema_version', 'label_schema_definition_hash',
                  'label_mapping_hash', 'num_classes'):
        if not isinstance(checkpoint, dict) or checkpoint.get(field) != expected[field]:
            checkpoint_mismatches[field] = (
                None if not isinstance(checkpoint, dict) else checkpoint.get(field), expected[field])
    if (not isinstance(checkpoint, dict) or
            checkpoint.get('renew_class_to_index') != current_renew):
        checkpoint_mismatches['renew_class_to_index'] = '与当前252类索引不一致或缺失'
    if checkpoint_mismatches:
        raise ValueError(f'模型checkpoint标签映射不一致：{checkpoint_mismatches}')
    return True


def model_predictions(pack, key_rows, model_name):
    """可选：在同一批抽样图片上跑模型，返回 {image: true_class 或 '?'} 与模型准确率。"""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), 'ReportChart'))
    import torch
    from PIL import Image
    from calculate import calculate_mean_and_std
    import eval_confusion_matrix as ecm

    net, ck, renew, idx2name, path = ecm.build_and_load(model_name, verbose=True)
    try:
        validate_checkpoint_identity(ck)
    except ValueError as exc:
        raise SystemExit(f'[hl] {exc}，拒绝生成模型对照列') from exc
    mean, std, _skip = calculate_mean_and_std(ecm.TRAIN_ROOT, retrieve_file=True)
    tf = ecm._transform(mean.tolist() if hasattr(mean, 'tolist') else mean,
                        std.tolist() if hasattr(std, 'tolist') else std)
    preds = {}
    with torch.no_grad():
        for r in key_rows:
            img_path = os.path.join(pack, 'images', r['image'])
            try:
                out = net(tf(Image.open(img_path).convert('RGB')).unsqueeze(0))
                preds[r['image']] = idx2name.get(int(out.argmax(dim=1).item()), '?')
            except Exception as e:
                preds[r['image']] = f'ERR({e.__class__.__name__})'
    return preds


def main():
    ap = argparse.ArgumentParser(description="标注一致性（Cohen's κ）报告")
    ap.add_argument('--pack', help='抽样包目录（make_kappa_pack.py 的产物）')
    ap.add_argument('--sheetA', default='sheetA.csv')
    ap.add_argument('--sheetB', default='sheetB.csv')
    ap.add_argument('--with-model', default=None, help='附：在抽样图片上跑该模型（注册表 key，如 S-NoProG）')
    ap.add_argument('--selftest', action='store_true', help='只做 κ 计算自检')
    a = ap.parse_args()

    if a.selftest:
        # 已知例：10 例；A: 5x+5y；B: x,x,x,x,y,x,x,y,y,y
        # 一致 7/10 → p_o=0.70；B 边缘 6x/4y → p_e=0.5*0.6+0.5*0.4=0.50 → κ=0.40
        A = ['x'] * 5 + ['y'] * 5
        B = ['x'] * 4 + ['y'] + ['x'] * 2 + ['y'] * 3
        k, po, pe = cohens_kappa(A, B)
        print(f'[selftest] 用例1 p_o={po:.2f} p_e={pe:.2f} κ={k:.2f}（期望 0.70/0.50/0.40）→ {kappa_level(k)}')
        assert abs(po - 0.70) < 1e-9 and abs(pe - 0.50) < 1e-9 and abs(k - 0.40) < 1e-9
        # 用例2：完全一致 → κ=1
        k2, po2, pe2 = cohens_kappa(['a', 'b', 'a'], ['a', 'b', 'a'])
        print(f'[selftest] 用例2 完全一致 p_o={po2:.2f} κ={k2:.2f}（期望 1.00）')
        assert abs(k2 - 1.0) < 1e-9
        # 用例3：与随机无异（p_o=p_e）→ κ=0
        k3, po3, pe3 = cohens_kappa(['a', 'a', 'b', 'b'], ['a', 'b', 'a', 'b'])
        print(f'[selftest] 用例3 随机水平 p_o={po3:.2f} p_e={pe3:.2f} κ={k3:.2f}（期望 0.00）')
        assert abs(k3) < 1e-9
        print('[selftest] 通过（κ 计算与分级函数正常）')
        return

    if not a.pack:
        raise SystemExit('请用 --pack 指定抽样包目录（或 --selftest）')
    pack = a.pack if os.path.isabs(a.pack) else os.path.join(EXPORT_ROOT, a.pack)
    if not os.path.isdir(pack):
        pack = a.pack
    if not os.path.isdir(pack) and not os.path.isabs(a.pack):
        # 兜底：允许在任意 CWD 下用 "exports/<包名>" 或 "<包名>" 调用（此前只在 human_loop 目录下才行）
        cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.pack)
        if os.path.isdir(cand):
            pack = cand
    if not os.path.isdir(pack):
        raise SystemExit(f'[hl] 找不到抽样包目录：{a.pack}（已尝试 {EXPORT_ROOT} 与脚本目录）')
    protocol_path = os.path.join(pack, 'protocol.json')
    if not os.path.isfile(protocol_path):
        raise SystemExit('[hl] 抽样包缺少 protocol.json；旧254类盲标包已失效，请按252类协议重新生成')
    with open(protocol_path, encoding='utf-8') as f:
        protocol = json.load(f)
    try:
        key_rows, key_by_image = validate_pack_files(pack, protocol)
    except ValueError as exc:
        raise SystemExit(f'[hl] 抽样包协议无效：{exc}') from exc
    A = collect(pack, a.sheetA, key_by_image)
    B = collect(pack, a.sheetB, key_by_image)
    if A is None or B is None:
        raise SystemExit(f'[hl] 找不到标注表 {a.sheetA}/{a.sheetB}；生成抽样包后应先标注再跑本脚本')

    preds = model_predictions(pack, key_rows, a.with_model) if a.with_model else None

    # 按类对聚合
    per_pair = {}
    for r in key_rows:
        d = per_pair.setdefault(r['pair_id'], [])
        d.append(r)

    rows_out, tot = [], dict(n=0, a=[], b=[], ok_a=0, ok_b=0, ok_m=0, n_m=0,
                            cons=0, cons_ok=0, abstain_a=0, abstain_b=0,
                            outlier_a=0, outlier_b=0)
    for pid in sorted(per_pair):
        recs = per_pair[pid]
        la, lb, truth = [], [], []
        ok_a = ok_b = ok_m = n_m = 0
        cons = cons_ok = 0
        abstain_a = abstain_b = outlier_a = outlier_b = 0
        for r in recs:
            rec_a, rec_b = A[r['image']], B[r['image']]
            outlier_a += int(rec_a['outlier'])
            outlier_b += int(rec_b['outlier'])
            abstain_a += int(rec_a['label'] is None)
            abstain_b += int(rec_b['label'] is None)
            ca = None if rec_a['outlier'] else rec_a['label']
            cb = None if rec_b['outlier'] else rec_b['label']
            t = r['true_short']
            truth.append(t)
            if ca is not None:
                la.append(ca); ok_a += (ca == t)
            if cb is not None:
                lb.append(cb); ok_b += (cb == t)
            if ca is not None and cb is not None:
                if ca == cb:
                    cons += 1
                    cons_ok += (ca == t)
            if preds is not None:
                p = preds.get(r['image'])
                n_m += 1
                ok_m += (p == r['true_class'])
        # κ 只在两轮均有效且均未标记“图外”的图片上计算。
        both = [(A[r['image']]['label'], B[r['image']]['label']) for r in recs
                if A[r['image']]['label'] is not None and B[r['image']]['label'] is not None
                and not A[r['image']]['outlier'] and not B[r['image']]['outlier']]
        k, po, pe = cohens_kappa([x for x, _ in both], [y for _, y in both])
        na, nb = len(la), len(lb)
        row = dict(pair_id=pid, n=len(recs), n_both=len(both),
                   p_o=('' if po is None else round(po, 4)),
                   p_e=('' if pe is None else round(pe, 4)),
                   kappa=('' if k is None else round(k, 4)), kappa_macro='', level=kappa_level(k),
                   acc_A=('' if na == 0 else round(ok_a / na, 4)),
                   acc_B=('' if nb == 0 else round(ok_b / nb, 4)),
                   n_consensus=cons,
                   acc_consensus=('' if cons == 0 else round(cons_ok / cons, 4)),
                   acc_model=('' if n_m == 0 else round(ok_m / n_m, 4)),
                   abstain_A=abstain_a, abstain_B=abstain_b,
                   outlier_A=outlier_a, outlier_B=outlier_b)
        rows_out.append(row)
        tot['n'] += len(recs); tot['a'] += la; tot['b'] += lb
        tot['ok_a'] += ok_a; tot['ok_b'] += ok_b; tot['ok_m'] += ok_m; tot['n_m'] += n_m
        tot['cons'] += cons; tot['cons_ok'] += cons_ok
        tot['abstain_a'] += abstain_a; tot['abstain_b'] += abstain_b
        tot['outlier_a'] += outlier_a; tot['outlier_b'] += outlier_b

    # 汇总（所有类对合并）
    both_all = [(A[r['image']]['label'], B[r['image']]['label']) for r in key_rows
                if A[r['image']]['label'] is not None and B[r['image']]['label'] is not None
                and not A[r['image']]['outlier'] and not B[r['image']]['outlier']]
    k_all, po_all, pe_all = cohens_kappa([x for x, _ in both_all], [y for _, y in both_all])
    valid_pair_kappas = [float(row['kappa']) for row in rows_out if row['kappa'] != '']
    macro_kappa = (sum(valid_pair_kappas) / len(valid_pair_kappas)) if valid_pair_kappas else None
    summary = dict(pair_id='ALL', n=tot['n'], n_both=len(both_all),
                   p_o=('' if po_all is None else round(po_all, 4)),
                   p_e=('' if pe_all is None else round(pe_all, 4)),
                   kappa=('' if k_all is None else round(k_all, 4)),
                   kappa_macro=('' if macro_kappa is None else round(macro_kappa, 4)),
                   level=kappa_level(k_all),
                   acc_A=('' if not tot['a'] else round(tot['ok_a'] / len(tot['a']), 4)),
                   acc_B=('' if not tot['b'] else round(tot['ok_b'] / len(tot['b']), 4)),
                   n_consensus=tot['cons'],
                   acc_consensus=('' if not tot['cons'] else round(tot['cons_ok'] / tot['cons'], 4)),
                   acc_model=('' if not tot['n_m'] else round(tot['ok_m'] / tot['n_m'], 4)),
                   abstain_A=tot['abstain_a'], abstain_B=tot['abstain_b'],
                   outlier_A=tot['outlier_a'], outlier_B=tot['outlier_b'])

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(pack, f'kappa_report_{ts}.csv')
    md_path = os.path.join(pack, f'kappa_report_{ts}.md')
    fields = list(rows_out[0].keys())
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out + [summary])

    def md_table(rs):
        head = '| 类对 | n | 有效(两轮) | p_o | p_e | **池化/类对κ** | 宏平均κ | 一致性分级 | 轮A准确率 | 轮B准确率 | 一致且正确率 | 模型准确率 | A/B弃权 | A/B图外 |'
        sep = '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'
        lines = [head, sep]
        for r in rs:
            lines.append('| {} | {} | {} | {} | {} | **{}** | {} | {} | {} | {} | {} | {} | {}/{} | {}/{} |'.format(
                r['pair_id'], r['n'], r['n_both'], r['p_o'], r['p_e'], r['kappa'],
                r['kappa_macro'], r['level'], r['acc_A'], r['acc_B'],
                r['acc_consensus'], r['acc_model'], r['abstain_A'], r['abstain_B'],
                r['outlier_A'], r['outlier_B']))
        return '\n'.join(lines)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 标注一致性（Cohen\'s κ）报告\n\n')
        f.write(f'- 抽样包：`{os.path.basename(pack)}`\n- 生成时间：{ts}\n')
        f.write(f'- 研究设计：同一标注者两轮 test--retest；轮次 A = {a.sheetA}，轮次 B = {a.sheetB}；模型对照：{a.with_model or "（未跑）"}\n')
        f.write('- 纳入规则：两轮均有效且两轮均未标记“[图外]”；主结果为逐类对 κ 与有效类对宏平均 κ，ALL 行池化 κ 仅作补充。\n\n')
        f.write(md_table(rows_out + [summary]))
        f.write('\n\n> 说明：较低 κ 只能提示在当前样本与流程下标签边界复现性较弱；'
                '必须结合弃权、图外、样本质量和置信区间解释，不能单独证明“任务内在歧义”或“人类上限”。\n')

    print('=' * 96)
    print(f'标注一致性报告｜抽样包 {os.path.basename(pack)}｜图片 {tot["n"]} 张')
    print(f'轮次 A 有效标注 {len(tot["a"])}，轮次 B {len(tot["b"])}；两轮共同有效且非图外 {len(both_all)}')
    if po_all is None:
        print('[提示] 两张标注表都还没有有效选择（choice 列为空？）——请先完成盲标再运行本脚本；')
        print('       本脚本已照常导出空白报告（供你确认表结构是否正确）。')
    else:
        macro_text = '不可计算' if macro_kappa is None else f'{macro_kappa:.3f}'
        print(f'【汇总】池化 p_o={po_all:.3f} p_e={pe_all:.3f} κ={k_all:.3f}；逐类对宏平均 κ={macro_text}')
        if tot['a']:
            print(f'  两轮准确率：A={tot["ok_a"]/len(tot["a"])*100:.1f}%  B={tot["ok_b"]/len(tot["b"])*100:.1f}%')
        if tot['cons']:
            print(f'  两轮一致 {tot["cons"]} 张，其中正确 {tot["cons_ok"]} 张（{tot["cons_ok"]/tot["cons"]*100:.1f}%）')
        print(f'  弃权 A/B={tot["abstain_a"]}/{tot["abstain_b"]}；图外 A/B={tot["outlier_a"]}/{tot["outlier_b"]}')
        if tot['n_m']:
            print(f'  模型（{a.with_model}）在同一批图片上准确率 = {tot["ok_m"]/tot["n_m"]*100:.1f}%')
    print('-' * 96)
    for r in rows_out:
        print(f'  {r["pair_id"]}  n={r["n"]:>3}  κ={str(r["kappa"]):>7}  {r["level"]:<16} '
              f'轮A={str(r["acc_A"]):>6} 轮B={str(r["acc_B"]):>6} 模型={str(r["acc_model"]):>6}')
    print('=' * 96)
    print(f'[hl] 明细已存：{csv_path}')
    print(f'[hl] Markdown 表格（可摘进论文）：{md_path}')


if __name__ == '__main__':
    main()
