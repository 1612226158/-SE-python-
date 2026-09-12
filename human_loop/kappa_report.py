# -*- coding: utf-8 -*-
"""
kappa_report.py —— 标注一致性（Cohen's κ）与"人类上限 vs 模型"报告（2026-09-12 建，human_loop）

输入：make_kappa_pack.py 生成的抽样包（含 _key.csv 与两张标注表 sheetA/sheetB）
输出：控制台报告 + <pack>/kappa_report.csv + <pack>/kappa_report.md（可直接摘进论文）

报告内容（每个类对 + 汇总）：
  n / 有效标注数 / 一致率 p_o / 随机期望 p_e / **Cohen's κ** / 分级
  标注者 A、B 各自的"人类准确率"（相对真实类别）
  --with-model 时附上模型在**同一批图片**上的准确率 → 人类上限 vs 模型 的直接对比

用法：
  python kappa_report.py --pack exports/20260912_xxxxxx_kappa_pack
  python kappa_report.py --pack <pack> --with-model S-NoProG
  python kappa_report.py --selftest          # κ 计算自检（已知例：p_o=0.7, p_e=0.5 → κ=0.4）

纪律：只读图片与 CSV；不写回标注表；不改数据集。
"""
import os
import sys
import csv
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_common import cohens_kappa, kappa_level, EXPORT_ROOT  # noqa: E402


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
    return None


def collect(pack, sheet_name):
    """读一张标注表 → {image: 规范化标签或 None}。"""
    rows = read_csv(os.path.join(pack, sheet_name))
    if rows is None:
        return None
    out = {}
    for r in rows:
        out[r['image']] = normalize_choice(r.get('choice'), r.get('option_1', ''), r.get('option_2', ''))
    return out


def model_predictions(pack, key_rows, model_name):
    """可选：在同一批抽样图片上跑模型，返回 {image: true_class 或 '?'} 与模型准确率。"""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(here), 'ReportChart'))
    import torch
    from PIL import Image
    from calculate import calculate_mean_and_std
    import eval_confusion_matrix as ecm

    net, ck, renew, idx2name, path = ecm.build_and_load(model_name, verbose=True)
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
    key_rows = read_csv(os.path.join(pack, '_key.csv'))
    if key_rows is None:
        raise SystemExit(f'[hl] {pack} 下找不到 _key.csv（请确认是 make_kappa_pack 的产物）')
    A = collect(pack, a.sheetA)
    B = collect(pack, a.sheetB)
    if A is None or B is None:
        raise SystemExit(f'[hl] 找不到标注表 {a.sheetA}/{a.sheetB}；生成抽样包后应先标注再跑本脚本')

    preds = model_predictions(pack, key_rows, a.with_model) if a.with_model else None

    # 按类对聚合
    per_pair = {}
    for r in key_rows:
        d = per_pair.setdefault(r['pair_id'], [])
        d.append(r)

    rows_out, tot = [], dict(n=0, a=[], b=[], ok_a=0, ok_b=0, ok_m=0, n_m=0, cons=0, cons_ok=0)
    for pid in sorted(per_pair):
        recs = per_pair[pid]
        la, lb, truth = [], [], []
        ok_a = ok_b = ok_m = n_m = 0
        cons = cons_ok = 0
        for r in recs:
            ca, cb = A.get(r['image']), B.get(r['image'])
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
        pair_set = [r['pair_id'] for r in recs]
        # κ 只在两人都标注的图片上计算
        both = [(A.get(r['image']), B.get(r['image'])) for r in recs
                if A.get(r['image']) is not None and B.get(r['image']) is not None]
        k, po, pe = cohens_kappa([x for x, _ in both], [y for _, y in both])
        na, nb = len(la), len(lb)
        row = dict(pair_id=pid, n=len(recs), n_both=len(both),
                   p_o=('' if po is None else round(po, 4)),
                   p_e=('' if pe is None else round(pe, 4)),
                   kappa=('' if k is None else round(k, 4)), level=kappa_level(k),
                   acc_A=('' if na == 0 else round(ok_a / na, 4)),
                   acc_B=('' if nb == 0 else round(ok_b / nb, 4)),
                   n_consensus=cons,
                   acc_consensus=('' if cons == 0 else round(cons_ok / cons, 4)),
                   acc_model=('' if n_m == 0 else round(ok_m / n_m, 4)))
        rows_out.append(row)
        tot['n'] += len(recs); tot['a'] += la; tot['b'] += lb
        tot['ok_a'] += ok_a; tot['ok_b'] += ok_b; tot['ok_m'] += ok_m; tot['n_m'] += n_m
        tot['cons'] += cons; tot['cons_ok'] += cons_ok

    # 汇总（所有类对合并）
    both_all = [(A.get(r['image']), B.get(r['image'])) for r in key_rows
                if A.get(r['image']) is not None and B.get(r['image']) is not None]
    k_all, po_all, pe_all = cohens_kappa([x for x, _ in both_all], [y for _, y in both_all])
    summary = dict(pair_id='ALL', n=tot['n'], n_both=len(both_all),
                   p_o=('' if po_all is None else round(po_all, 4)),
                   p_e=('' if pe_all is None else round(pe_all, 4)),
                   kappa=('' if k_all is None else round(k_all, 4)), level=kappa_level(k_all),
                   acc_A=('' if not tot['a'] else round(tot['ok_a'] / len(tot['a']), 4)),
                   acc_B=('' if not tot['b'] else round(tot['ok_b'] / len(tot['b']), 4)),
                   n_consensus=tot['cons'],
                   acc_consensus=('' if not tot['cons'] else round(tot['cons_ok'] / tot['cons'], 4)),
                   acc_model=('' if not tot['n_m'] else round(tot['ok_m'] / tot['n_m'], 4)))

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(pack, f'kappa_report_{ts}.csv')
    md_path = os.path.join(pack, f'kappa_report_{ts}.md')
    fields = list(rows_out[0].keys())
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out + [summary])

    def md_table(rs):
        head = '| 类对 | n | 有效(两人) | p_o | p_e | **κ** | 一致性分级 | 人A准确率 | 人B准确率 | 一致且正确率 | 模型准确率 |'
        sep = '|---|---|---|---|---|---|---|---|---|---|---|'
        lines = [head, sep]
        for r in rs:
            lines.append('| {} | {} | {} | {} | {} | **{}** | {} | {} | {} | {} | {} |'.format(
                r['pair_id'], r['n'], r['n_both'], r['p_o'], r['p_e'], r['kappa'], r['level'],
                r['acc_A'], r['acc_B'], r['acc_consensus'], r['acc_model']))
        return '\n'.join(lines)

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# 标注一致性（Cohen\'s κ）报告\n\n')
        f.write(f'- 抽样包：`{os.path.basename(pack)}`\n- 生成时间：{ts}\n')
        f.write(f'- 标注者：A = {a.sheetA}，B = {a.sheetB}；模型对照：{a.with_model or "（未跑）"}\n\n')
        f.write(md_table(rows_out + [summary]))
        f.write('\n\n> 说明：κ<0.4 表明该标签边界难以为人工稳定复现（支持"任务内在歧义"）；'
                'κ>0.6 则边界较清晰，模型错误不宜归因于歧义。\n')

    print('=' * 96)
    print(f'标注一致性报告｜抽样包 {os.path.basename(pack)}｜图片 {tot["n"]} 张')
    print(f'人工 A 有效标注 {len(tot["a"])}，B {len(tot["b"])}；两人共同标注 {len(both_all)}')
    if po_all is None:
        print('[提示] 两张标注表都还没有有效选择（choice 列为空？）——请先完成盲标再运行本脚本；')
        print('       本脚本已照常导出空白报告（供你确认表结构是否正确）。')
    else:
        print(f'【汇总】p_o={po_all:.3f} p_e={pe_all:.3f}  κ={k_all:.3f}（{kappa_level(k_all)}）')
        if tot['a']:
            print(f'  人类准确率：A={tot["ok_a"]/len(tot["a"])*100:.1f}%  B={tot["ok_b"]/len(tot["b"])*100:.1f}%')
        if tot['cons']:
            print(f'  两人一致 {tot["cons"]} 张，其中正确 {tot["cons_ok"]} 张（{tot["cons_ok"]/tot["cons"]*100:.1f}%）')
        if tot['n_m']:
            print(f'  模型（{a.with_model}）在同一批图片上准确率 = {tot["ok_m"]/tot["n_m"]*100:.1f}%')
    print('-' * 96)
    for r in rows_out:
        print(f'  {r["pair_id"]}  n={r["n"]:>3}  κ={str(r["kappa"]):>7}  {r["level"]:<16} '
              f'人A={str(r["acc_A"]):>6} 人B={str(r["acc_B"]):>6} 模型={str(r["acc_model"]):>6}')
    print('=' * 96)
    print(f'[hl] 明细已存：{csv_path}')
    print(f'[hl] Markdown 表格（可摘进论文）：{md_path}')


if __name__ == '__main__':
    main()
