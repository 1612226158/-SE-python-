# -*- coding: utf-8 -*-
"""
selftest_human_loop.py —— human_loop 工具链自检（2026-09-12 建）

零依赖（不需要 pytest），直接运行：
  D:\\Python\\Python3.10.7\\python.exe src_v2\\human_loop\\selftest_human_loop.py
  ... selftest_human_loop.py --with-model S-NoProG    # 额外跑一次"模型对照"分支（较慢，加载 torch）
  ... selftest_human_loop.py --keep                   # 保留临时目录以便人工检查

覆盖：
  [单元] Cohen's κ 计算（完全一致/随机水平/已知例/空输入）+ κ 分级阈值
  [单元] 短名提取、类名解析（唯一/未知/歧义）、类对解析（含全角冒号与格式错误）
  [集成] make_kappa_pack 生成抽样包：文件齐全、图片与答案键一一对应、两张表选项一致、
         文件名不含类别线索（盲标）、平衡抽样、同种子可复现
  [集成] kappa_report：完美标注（κ=1、人类准确率 100%）、弃权(choice=0)处理、报告 CSV/MD 落盘、
         未给 --with-model 时模型列留空
  [可选] --with-model 分支：模型在同一批图片上的准确率列非空

纪律：整份测试只**读** val 数据集，全部产物写在临时目录里并在结束时删除（--keep 除外）。
      刻意声明：测试里的"标注"是**程序生成的测试夹具**，不是真实人工标注，绝不用于论文。
"""
import os
import sys
import csv
import shutil
import argparse
import tempfile
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from hl_common import (cohens_kappa, kappa_level, short_name, resolve_class, parse_pairs,
                       merged_class_index, VAL_ROOT, EXPORT_ROOT)  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    mark = 'PASS' if cond else 'FAIL'
    print(f'  [{mark}] {name}' + (f'  ← {detail}' if detail and not cond else ''))


def section(title):
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)


# ---------------------------------------------------------------- 单元测试
def test_kappa():
    section('[1] Cohen\'s κ 计算')
    k, po, pe = cohens_kappa(['a', 'b', 'a'], ['a', 'b', 'a'])
    check('完全一致 → κ=1', abs(k - 1) < 1e-12 and abs(po - 1) < 1e-12, f'k={k}')

    k, po, pe = cohens_kappa(['a', 'a', 'b', 'b'], ['a', 'b', 'a', 'b'])
    check('随机水平 → κ=0', abs(k) < 1e-12 and abs(po - 0.5) < 1e-12 and abs(pe - 0.5) < 1e-12,
          f'k={k} p_o={po} p_e={pe}')

    A = ['x'] * 5 + ['y'] * 5
    B = ['x'] * 4 + ['y'] + ['x'] * 2 + ['y'] * 3
    k, po, pe = cohens_kappa(A, B)
    check('已知例 p_o=.70 p_e=.50 → κ=.40',
          abs(po - 0.70) < 1e-12 and abs(pe - 0.50) < 1e-12 and abs(k - 0.40) < 1e-12,
          f'k={k} p_o={po} p_e={pe}')

    # 与"直接从 2×2 计数表算"的独立实现对照（防公式写错）
    a = ['x'] * 30 + ['y'] * 20
    b = ['x'] * 25 + ['y'] * 5 + ['x'] * 5 + ['y'] * 15
    k, po, pe = cohens_kappa(a, b)
    n = 50
    n_xx = sum(1 for i in range(n) if a[i] == 'x' and b[i] == 'x')
    n_xy = sum(1 for i in range(n) if a[i] == 'x' and b[i] == 'y')
    n_yx = sum(1 for i in range(n) if a[i] == 'y' and b[i] == 'x')
    n_yy = sum(1 for i in range(n) if a[i] == 'y' and b[i] == 'y')
    po_ref = (n_xx + n_yy) / n
    pe_ref = ((n_xx + n_xy) * (n_xx + n_yx) + (n_yx + n_yy) * (n_xy + n_yy)) / (n * n)
    k_ref = (po_ref - pe_ref) / (1 - pe_ref)
    check('与独立 2×2 计数实现一致',
          abs(po - po_ref) < 1e-12 and abs(pe - pe_ref) < 1e-12 and abs(k - k_ref) < 1e-12,
          f'{k:.6f} vs {k_ref:.6f}')

    k, po, pe = cohens_kappa([], [])
    check('空输入 → (None, None, None)', k is None and po is None and pe is None)


def test_kappa_level():
    section('[2] κ 分级阈值')
    cases = [(-0.1, '差于随机'), (0.0, '极弱'), (0.2, '极弱'), (0.21, '一般'), (0.4, '一般'),
             (0.41, '中等'), (0.6, '中等'), (0.61, '较好'), (0.8, '较好'),
             (0.81, '几乎完全一致'), (1.0, '几乎完全一致'), (None, '不可计算')]
    for val, expect in cases:
        got = kappa_level(val)
        check(f'κ={val} → {expect}', got.startswith(expect), f'实际 {got}')


def test_names_and_pairs():
    section('[3] 类名/类对解析')
    check('短名提取（私有类）', short_name('私有类-厨余垃圾-蔬菜') == '蔬菜')
    check('短名提取（公共/合并类）', short_name('公共类-可回收物-玻璃制品类') == '玻璃制品类')
    check('短名提取（无横线时原样返回）', short_name('foo') == 'foo')

    fake = {'私有类-厨余垃圾-蔬菜': [1], '私有类-厨余垃圾-菜根菜叶': [2],
            '私有类-可回收物-鞋': [3], '公共类-可回收物-鞋': [30], '私有类-可回收物-鞋子': [4],
            '私有类-厨余垃圾-锅': [5]}
    check('短名唯一 → 解析成功', resolve_class('蔬菜', fake) == '私有类-厨余垃圾-蔬菜')
    check('短名唯一（鞋子）→ 解析成功', resolve_class('鞋子', fake) == '私有类-可回收物-鞋子')
    check('全名直接命中', resolve_class('公共类-可回收物-鞋', fake) == '公共类-可回收物-鞋')
    try:
        resolve_class('鞋', fake)     # '鞋' 同时匹配 私有类-...-鞋 与 公共类-...-鞋
        check('歧义短名应报错', False, '未报错')
    except SystemExit:
        check('歧义短名（鞋→两个同名类）→ 报错并列出候选', True)
    try:
        resolve_class('不存在的类', fake)
        check('未知类应报错', False, '未报错')
    except SystemExit:
        check('未知类 → 报错', True)

    pairs = parse_pairs('蔬菜:菜根菜叶,锅：鞋子', fake)   # 第二组用全角冒号
    check('类对解析：pair_id 编号 P01/P02', [p[0] for p in pairs] == ['P01', 'P02'], str(pairs))
    check('类对解析：全角冒号兼容',
          pairs[1][1] == '私有类-厨余垃圾-锅' and pairs[1][2] == '私有类-可回收物-鞋子', str(pairs))
    try:
        parse_pairs('格式错误没有冒号', fake)
        check('类对格式错误应报错', False, '未报错')
    except SystemExit:
        check('类对格式错误 → 报错', True)


# ---------------------------------------------------------------- 集成测试
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def test_pack_and_report(tmp, with_model=None):
    section('[4] make_kappa_pack 生成抽样包（真实 val 只读）')
    if not os.path.isdir(VAL_ROOT):
        print('  [SKIP] 未找到 val 目录，跳过集成测试')
        return
    exe = sys.executable
    pack = os.path.join(tmp, 'pack')
    rc, out = run([exe, os.path.join(HERE, 'make_kappa_pack.py'),
                   '--pairs', '锅:电饭煲', '--n-per-class', '2', '--seed', '123', '--out', pack])
    check('生成命令退出码 0', rc == 0, out[-400:])
    if rc != 0:
        return
    need = ['images', 'sheetA.csv', 'sheetB.csv', '_key.csv', 'pairs.csv', 'README_ANNOTATOR.txt']
    check('产物齐全', all(os.path.exists(os.path.join(pack, n)) for n in need),
          str([n for n in need if not os.path.exists(os.path.join(pack, n))]))

    def rows(p):
        with open(p, encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))

    A, B, K = rows(os.path.join(pack, 'sheetA.csv')), rows(os.path.join(pack, 'sheetB.csv')), rows(os.path.join(pack, '_key.csv'))
    check('sheetA/B 行数一致且等于答案键', len(A) == len(B) == len(K) == 4, f'{len(A)}/{len(B)}/{len(K)}')
    check('图片文件数 = 行数', len(os.listdir(os.path.join(pack, 'images', 'P01'))) == len(K))
    check('图片与答案键一一对应', {r['image'] for r in K} == {r['image'] for r in A})
    check('两张表选项顺序完全一致（同一道题）',
          all(a['option_1'] == b['option_1'] and a['option_2'] == b['option_2'] for a, b in zip(A, B)))
    check('表中选项与答案键一致',
          all(k['option_1'] == a['option_1'] and k['option_2'] == a['option_2'] for k, a in zip(K, A)))
    check('盲标：文件名与路径不含类别名',
          all(('锅' not in r['image']) and ('电饭煲' not in r['image']) for r in K))
    check('平衡抽样：两个类各 2 张',
          sorted([r['true_short'] for r in K]) == ['电饭煲'] * 2 + ['锅'] * 2,
          str([r['true_short'] for r in K]))
    check('答案键记录了来源路径（便于复核）', all(r.get('source_path') for r in K))

    # 同种子可复现
    pack2 = os.path.join(tmp, 'pack_same_seed')
    run([exe, os.path.join(HERE, 'make_kappa_pack.py'),
         '--pairs', '锅:电饭煲', '--n-per-class', '2', '--seed', '123', '--out', pack2])
    K2 = rows(os.path.join(pack2, '_key.csv'))
    check('同种子 → 抽到同一批图片（可复现）',
          sorted(r['source_path'] for r in K) == sorted(r['source_path'] for r in K2))

    section('[5] kappa_report 报告生成（测试夹具：程序生成的标注，仅用于自检）')
    key = {r['image']: r for r in K}
    # 夹具1：完美标注者（两人都选真值）
    fixA = os.path.join(pack, '_fix_perfect_A.csv')
    fixB = os.path.join(pack, '_fix_perfect_B.csv')
    for path in (fixA, fixB):
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['pair_id', 'image', 'option_1', 'option_2', 'choice', 'notes'])
            w.writeheader()
            for r in A:
                truth = key[r['image']]['true_short']
                w.writerow(dict(r, choice=('1' if truth == r['option_1'] else '2')))
    rc, out = run([exe, os.path.join(HERE, 'kappa_report.py'), '--pack', pack,
                   '--sheetA', '_fix_perfect_A.csv', '--sheetB', '_fix_perfect_B.csv'])
    check('报告命令退出码 0', rc == 0, out[-500:])
    reps = [f for f in os.listdir(pack) if f.startswith('kappa_report_')]
    check('报告 CSV/MD 已落盘', any(f.endswith('.csv') for f in reps) and any(f.endswith('.md') for f in reps))
    rep = sorted(f for f in reps if f.endswith('.csv'))[-1]
    rr = rows(os.path.join(pack, rep))
    allrow = [r for r in rr if r['pair_id'] == 'ALL'][0]
    check('完美标注 → κ=1.0', abs(float(allrow['kappa']) - 1.0) < 1e-9, str(allrow))
    check('完美标注 → 人类准确率 100%', float(allrow['acc_A']) == 1.0 and float(allrow['acc_B']) == 1.0)
    check('未给 --with-model 时模型列为空', allrow['acc_model'] == '', f"acc_model={allrow['acc_model']!r}")

    # 夹具2：一人完美、一人全部反向 → 两人一致率 0 → κ 为负
    fixC = os.path.join(pack, '_fix_flip_B.csv')
    with open(fixC, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair_id', 'image', 'option_1', 'option_2', 'choice', 'notes'])
        w.writeheader()
        for r in A:
            truth = key[r['image']]['true_short']
            w.writerow(dict(r, choice=('2' if truth == r['option_1'] else '1')))
    rc, out = run([exe, os.path.join(HERE, 'kappa_report.py'), '--pack', pack,
                   '--sheetA', '_fix_perfect_A.csv', '--sheetB', '_fix_flip_B.csv'])
    rep2 = sorted(f for f in os.listdir(pack) if f.startswith('kappa_report_') and f.endswith('.csv'))[-1]
    rr2 = rows(os.path.join(pack, rep2))
    all2 = [r for r in rr2 if r['pair_id'] == 'ALL'][0]
    check('完全相反的两人 → κ<0', float(all2['kappa']) < 0, str(all2))

    # 夹具3：弃权处理（一半填 0）→ 有效标注数减半，但流程不崩
    fixD = os.path.join(pack, '_fix_abstain_A.csv')
    with open(fixD, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair_id', 'image', 'option_1', 'option_2', 'choice', 'notes'])
        w.writeheader()
        for i, r in enumerate(A):
            truth = key[r['image']]['true_short']
            ch = '0' if i % 2 == 0 else ('1' if truth == r['option_1'] else '2')
            w.writerow(dict(r, choice=ch, notes=('弃权' if ch == '0' else '')))
    rc, out = run([exe, os.path.join(HERE, 'kappa_report.py'), '--pack', pack,
                   '--sheetA', '_fix_abstain_A.csv', '--sheetB', '_fix_perfect_B.csv'])
    rep3 = sorted(f for f in os.listdir(pack) if f.startswith('kappa_report_') and f.endswith('.csv'))[-1]
    rr3 = rows(os.path.join(pack, rep3))
    all3 = [r for r in rr3 if r['pair_id'] == 'ALL'][0]
    check('弃权(choice=0) → 有效共同标注数下降', int(all3['n_both']) == 2, f"n_both={all3['n_both']}")

    if with_model:
        section(f'[6] --with-model 分支（{with_model}）')
        rc, out = run([exe, os.path.join(HERE, 'kappa_report.py'), '--pack', pack,
                       '--sheetA', '_fix_perfect_A.csv', '--sheetB', '_fix_perfect_B.csv',
                       '--with-model', with_model])
        check('带模型分支退出码 0', rc == 0, out[-600:])
        repm = sorted(f for f in os.listdir(pack) if f.startswith('kappa_report_') and f.endswith('.csv'))[-1]
        allm = [r for r in rows(os.path.join(pack, repm)) if r['pair_id'] == 'ALL'][0]
        check('模型准确率列已填充', allm['acc_model'] != '', f"acc_model={allm['acc_model']!r}")


def main():
    ap = argparse.ArgumentParser(description='human_loop 工具链自检')
    ap.add_argument('--with-model', default=None, help='额外测试模型对照分支（如 S-NoProG；会加载 torch，较慢）')
    ap.add_argument('--keep', action='store_true', help='保留临时目录')
    a = ap.parse_args()

    print('human_loop 自检开始（只读数据；产物写临时目录）')
    print(f'python = {sys.executable}')
    print(f'val    = {VAL_ROOT}（{"存在" if os.path.isdir(VAL_ROOT) else "缺失"}）')

    tmp = tempfile.mkdtemp(prefix='hl_selftest_')
    try:
        test_kappa()
        test_kappa_level()
        test_names_and_pairs()
        test_pack_and_report(tmp, with_model=a.with_model)
    finally:
        if a.keep:
            print(f'\n[--keep] 临时目录保留：{tmp}')
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print('\n' + '=' * 80)
    print(f'结果：通过 {len(PASS)} 项，失败 {len(FAIL)} 项')
    if FAIL:
        print('失败项：')
        for f in FAIL:
            print(f'  - {f}')
    print('=' * 80)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
