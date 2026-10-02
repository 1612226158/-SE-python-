# -*- coding: utf-8 -*-
"""
annotate_gui.py —— 标注一致性检验的图形标注界面（2026-09-12 建，human_loop）

做什么：把 make_kappa_pack.py 生成的盲标抽样包，做成"看图 → 二选一 → 自动存盘"的界面，
        不用逐个打开文件夹找图、也不用手工对齐 CSV。

安全/纪律：
  * **绝不读取 _key.csv**（答案键）——界面只显示图片与两个候选类，保证盲标；
  * 每次作答后**立即原子写盘**（临时文件 + os.replace），关窗/断电不丢数据；写盘失败会明确告警并阻止退出；
  * 只写 sheet<标注者>.csv；不碰图片、不改数据集、不参与训练。

用法：
  D:\\Python\\Python3.10.7\\python.exe src_v2\\human_loop\\annotate_gui.py
  ... annotate_gui.py --pack 20260929_002757_kappa_pack --annotator A
  ... annotate_gui.py --ui-selftest          # 自动构建界面并模拟作答（强制 UITEST 标注者，不碰真实表）

快捷键：1 / 2 选择左/右候选；0 无法判断；← → 上一张/下一张；Delete 清除本题（4 秒内 Ctrl+Z 可撤销）；
        Ctrl+S 保存；Ctrl+Q 或 Esc 保存并退出。
（单字母键不再做全局快捷键，以免与备注输入冲突；备注框获焦时按键一律放行给输入框。）

—— 2026-09-12 依据设计师审查（子代理 b8dd3394）整改 ——
P0：备注抢键守卫 / DPI 感知 + 像素字号 / takefocus=0 与不抖动的已答高亮 / 画布凹底色 /
    进度口径统一 + 作答 toast 确认 / 坏图状态（禁用选项、禁止作答、不留旧图）
P1：原子写异常保护与退出拦截 / Ctrl+S·Ctrl+Q·Ctrl+Z 组合键 / 边界提示 / 清除按钮仅已答可点与撤销 /
    自检强制 UITEST / 画布重绘防抖 / 下一张预解码（线程池 + JPEG draft）/ 窗口与右栏按工作区自适应
"""
import os
import sys
import csv
import argparse
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hl_common import (EXPORT_ROOT, validate_public_pack, public_class_member_hint,
                       configure_utf8_console)  # noqa: E402

configure_utf8_console()

VALID_CHOICES = frozenset({'0', '1', '2'})


def is_answered(row):
    """只有协议允许的0/1/2才算已答；空白或异常字符串都应继续提示补填。"""
    return (row.get('choice') or '').strip() in VALID_CHOICES


def first_unanswered_index(rows):
    """返回第一道未合法作答题目的索引；全部完成时返回None。"""
    return next((index for index, row in enumerate(rows) if not is_answered(row)), None)


def next_unanswered_index(rows, current_index):
    """从当前题之后寻找下一道未答题，必要时回绕；全部完成时返回None。"""
    if not rows:
        return None
    order = list(range(current_index + 1, len(rows))) + list(range(0, current_index))
    return next((index for index in order if not is_answered(rows[index])), None)


# ---------------- DPI 感知（必须在 Tk() 之前调用） ----------------
def _enable_dpi_awareness():
    """声明 DPI 感知并返回缩放系数（96dpi → 1.0）。未声明时 Windows 会位图拉伸导致界面发糊。"""
    if sys.platform != 'win32':
        return 1.0
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)      # PER_MONITOR_AWARE_V2
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)    # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return (dpi or 96) / 96.0
    except Exception:
        return 1.0


UI = _enable_dpi_awareness()


def px(n):
    """设计像素 → Tk 像素字号（负值 = 物理像素高度，避免二次缩放变糊）。"""
    return -int(round(n * UI))


def sp(n):
    """设计像素 → Tk 间距/尺寸。"""
    return int(round(n * UI))


# ---------------- 设计规范（色板/字号/间距） ----------------
# 对比度已按 WCAG AA 校核：正文/次要文字/警告色均达标（原 #6B7280、#D97706 不达标，已替换）
BG = '#EFF2F6'          # 页面底
CARD = '#FFFFFF'        # 卡片
CANVAS_BG = '#E8ECF2'   # 图片画布（凹一档，减轻 240 张连看的眩光）
BORDER = '#D8DEE7'      # 描边
TEXT = '#1F2A37'        # 主文字（14.5:1）
SUBTLE = '#5A6472'      # 次要文字（6.0:1 AA）
PRIMARY = '#2F6FED'     # 主色（选项①的身份色）
PRIMARY_DARK = '#2559C9'
CAND2 = '#7C3AED'       # 选项②的身份色
CAND2_DARK = '#6D28D9'
OK = '#16A34A'
OK_TEXT = '#15803D'          # 白底上的绿字（答案回显/保存提示）
OK_TEXT_DEEP = '#137237'     # 绿底上的绿字（"无法判断"被选中），对绿底 5.38:1
WARN = '#B45309'             # 白底上的警告色（状态栏提示）
WARN_DEEP = '#A34A08'        # 灰底上的警告色（"无法判断"/"清除"按钮），对 GHOST 5.0:1
OK_TINT = '#E7F6EC'
# "图外/异常"标记词：写进 notes 列（不改表结构、不动选项数，κ 的"固定二选一"前提不受影响）
OUTLIER_TOKEN = '[图外]'
WARN = '#B45309'
GHOST = '#E7ECF3'
GHOST_HOVER = '#DCE3EC'
EMPTY = '#9CA3AF'
BAR_TRACK = '#DDE3EC'
# 选项按钮三态配色（2026-09-12 二次整改）：
#   未选 = 各自色相的极淡底 + 同色相中灰描边（边界对比 ≈3:1，符合 WCAG 1.4.11 组件边界的可辨识要求）
#   悬停 = 同色相再深一档（**永不用实心彩色**，避免被误读成"已选"）
#   已选 = 实心彩色 + 白字 + 绿描边 + ✓ 前缀（永远比悬停醒目）
# 让选项①恒为蓝、选项②恒为紫，跨题、跨状态都保持同一身份色，便于标注者追踪"我选的是哪一个"。
OPT_STYLE = {
    '1': dict(idle='#EDF2FB', idle_hover='#E1E9F8', ring='#6F8BC4',
              solid=PRIMARY, solid_hover=PRIMARY_DARK),
    '2': dict(idle='#F4EFFC', idle_hover='#EBE2FA', ring='#9070CB',
              solid=CAND2, solid_hover=CAND2_DARK),
}

FONT_STACK = ('Microsoft YaHei UI', 'Microsoft YaHei', 'Segoe UI', 'SimSun')
FAMILY = 'Microsoft YaHei UI'      # 运行时由 _resolve_family() 按可用性覆盖

SM, MD, LG, XL = 8, 12, 16, 24     # 间距栅格（×UI）


def _resolve_family(root):
    """字体回退：取字体栈里第一个可用的，避免静默退到系统默认导致字形突变。"""
    global FAMILY
    try:
        avail = set(tkfont.families(root))
    except Exception:
        return
    for f in FONT_STACK:
        if f in avail:
            FAMILY = f
            return


def _protocol_ok(pack):
    """只接受当前252类且公开文件/图片哈希完整的抽样包；绝不读取答案键。"""
    try:
        validate_public_pack(pack)
        return True
    except (OSError, ValueError, TypeError):
        return False


def packs_available():
    """exports 下当前252类协议的 *_kappa_pack 目录，按名称倒序。"""
    if not os.path.isdir(EXPORT_ROOT):
        return []
    out = []
    for name in sorted(os.listdir(EXPORT_ROOT), reverse=True):
        p = os.path.join(EXPORT_ROOT, name)
        if os.path.isdir(p) and name.endswith('_kappa_pack') and _protocol_ok(p):
            out.append(p)
    return out


class AnnotatorApp:
    def __init__(self, root, pack=None, annotator='A'):
        self.root = root
        _resolve_family(root)
        self.root.title('人工标注 · 类对二选一（标注一致性检验）')
        self.root.configure(bg=BG)
        self._apply_geometry()
        self._init_fonts()

        self.pack = pack
        self.pack_lookup = {}
        self.annotator = annotator
        self.rows = []
        self.idx = 0
        self.sheet_path = None
        self.img_cache = None
        self.cur_pil = None
        self.img_ok = True
        self._resize_job = None
        self._toast_job = None
        self._undo = None
        self._pil_cache = {}
        self._pool = None

        self.status_var = tk.StringVar(value='')
        self.prog_var = tk.StringVar(value='')
        self.prog_pct = tk.StringVar(value='')
        self.pair_var = tk.StringVar(value='')
        self.notes_var = tk.StringVar(value='')
        self.pack_path_var = tk.StringVar(value='')
        self.answer_var = tk.StringVar(value='')

        self.container = tk.Frame(root, bg=BG)
        self.container.pack(fill='both', expand=True)
        self._bind_keys()
        self._build_start()

    # ------------------------------------------------------------------ 窗口/字体
    def _apply_geometry(self):
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w = min(sp(1440), int(sw * 0.94))
        h = min(sp(880), int(sh * 0.90))
        self.root.geometry(f'{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2 - sp(20))}')
        self.root.minsize(sp(1040), sp(660))
        self.right_w = max(sp(372), int(w * 0.29))

    def _init_fonts(self):
        self.F_TITLE = (FAMILY, px(19), 'bold')
        self.F_BIG = (FAMILY, px(22), 'bold')
        self.F_H2 = (FAMILY, px(13), 'bold')
        self.F_BODY = (FAMILY, px(12))
        self.F_SMALL = (FAMILY, px(11))
        self.F_OPT = (FAMILY, px(14), 'bold')

    # ------------------------------------------------------------------ 快捷键
    def _bind_keys(self):
        """数字/方向键走守卫分发；保存·退出·撤销用组合键（不与备注输入冲突）。"""
        plain = {'1': ('answer', 1), '2': ('answer', 2), '0': ('answer', 0),
                 'Left': ('move', -1), 'Right': ('move', 1), 'Delete': ('clear', None),
                 'x': ('flag', None), 'X': ('flag', None)}
        for k, (act, arg) in plain.items():
            self.root.bind(k, lambda e, a=act, g=arg: self._dispatch(a, g))
        self.root.bind('<Control-s>', lambda e: (self.save(), 'break')[1])
        self.root.bind('<Control-q>', lambda e: (self.quit_app(), 'break')[1])
        self.root.bind('<Control-z>', lambda e: (self.undo_clear(), 'break')[1])
        self.root.bind('<Escape>', lambda e: (self.quit_app(), 'break')[1])

    def _typing(self):
        """焦点是否在输入框里（备注框）——是则除组合键外一律放行。"""
        try:
            w = self.root.focus_get()
        except Exception:
            return False
        return isinstance(w, (tk.Entry, tk.Text))

    def _dispatch(self, act, arg):
        if self._typing():
            return None                     # 交给 Entry 自己处理
        if act == 'answer':
            self.answer(arg)
        elif act == 'move':
            self.move(arg)
        elif act == 'clear':
            self.clear_current()
        elif act == 'flag':
            self.toggle_outlier()
        return 'break'

    # ------------------------------------------------------------------ 启动页
    def _build_start(self):
        self._clear()
        card = tk.Frame(self.container, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        card.place(relx=0.5, rely=0.5, anchor='center', width=sp(640), height=sp(486))

        tk.Label(card, text='人工标注 · 类对二选一', font=(FAMILY, px(21), 'bold'), bg=CARD, fg=TEXT)\
            .pack(pady=(sp(30), sp(4)))
        tk.Label(card, text='用于“标注一致性检验（Cohen\'s κ）”：判断图中物体属于哪一类',
                 font=self.F_BODY, bg=CARD, fg=SUBTLE).pack()

        tk.Label(card, text='抽样包', font=self.F_H2, bg=CARD, fg=TEXT)\
            .pack(anchor='w', padx=sp(40), pady=(sp(24), sp(6)))
        packs = packs_available()
        if self.pack and os.path.isdir(self.pack) and _protocol_ok(self.pack):
            resolved = os.path.abspath(self.pack)
            if resolved not in [os.path.abspath(p) for p in packs]:
                packs.insert(0, resolved)
        packs = [os.path.abspath(p) for p in packs]
        basename_counts = {}
        for path in packs:
            basename_counts[os.path.basename(path)] = basename_counts.get(os.path.basename(path), 0) + 1
        self.pack_lookup = {}
        selected = ''
        explicit = os.path.abspath(self.pack) if self.pack else None
        for path in packs:
            base = os.path.basename(path)
            display = base if basename_counts[base] == 1 else f'{base}  —  {path}'
            self.pack_lookup[display] = path
            if explicit and os.path.normcase(path) == os.path.normcase(explicit):
                selected = display
        if not selected and self.pack_lookup:
            selected = next(iter(self.pack_lookup))
        self.pack_var = tk.StringVar(value=selected)
        try:
            ttk.Style().configure('TCombobox', padding=sp(6))
        except Exception:
            pass
        combo = ttk.Combobox(card, textvariable=self.pack_var, state='readonly',
                             values=list(self.pack_lookup),
                             font=(FAMILY, px(12)))
        combo.pack(fill='x', padx=sp(40))
        tk.Label(card, textvariable=self.pack_path_var, font=self.F_SMALL, fg=SUBTLE,
                 wraplength=sp(540), justify='left').pack(anchor='w', padx=sp(40), pady=(sp(4), 0))
        self._sync_pack_path()
        combo.bind('<<ComboboxSelected>>', lambda e: self._sync_pack_path())

        tk.Label(card, text='标注轮次', font=self.F_H2, bg=CARD, fg=TEXT)\
            .pack(anchor='w', padx=sp(40), pady=(sp(16), sp(6)))
        row = tk.Frame(card, bg=CARD)
        row.pack(anchor='w', padx=sp(40))
        self.ann_var = tk.StringVar(value=self.annotator)
        for who, desc in (('A', '第一轮 A'), ('B', '第二轮 B')):
            tk.Radiobutton(row, text=desc, variable=self.ann_var, value=who, font=self.F_BODY,
                           bg=CARD, fg=TEXT, activebackground=CARD, selectcolor=CARD,
                           takefocus=0, highlightthickness=0).pack(side='left', padx=(0, sp(20)))

        tk.Label(card, text='提示：同一标注者先做 A，至少间隔 12 小时再做 B；第二轮前不要查看 A 或答案键。',
                 font=self.F_SMALL, bg=CARD, fg=SUBTLE, wraplength=sp(540), justify='left')\
            .pack(anchor='w', padx=sp(40), pady=(sp(18), 0))

        tk.Button(card, text='开始标注', font=(FAMILY, px(13), 'bold'), bg=PRIMARY, fg='white',
                  activebackground=PRIMARY_DARK, activeforeground='white', relief='flat',
                  takefocus=0, cursor='hand2', command=self.start_session)\
            .pack(fill='x', padx=sp(40), pady=(sp(20), 0), ipady=sp(10))

        self.status_var.set(f'发现 {len(packs)} 个抽样包' if packs else '未发现抽样包，请先运行 make_kappa_pack.py')
        tk.Label(card, textvariable=self.status_var, font=self.F_SMALL, bg=CARD, fg=SUBTLE)\
            .pack(pady=(sp(10), 0))

    def _sync_pack_path(self):
        name = self.pack_var.get()
        if name:
            p = self.pack_lookup.get(name, os.path.join(EXPORT_ROOT, name))
            self.pack_path_var.set(f'路径：{p}')

    # ------------------------------------------------------------------ 会话
    def start_session(self):
        name = self.pack_var.get()
        if not name:
            messagebox.showwarning('提示', '请先选择抽样包（可先运行 make_kappa_pack.py 生成）')
            return
        self.pack = self.pack_lookup.get(name, os.path.join(EXPORT_ROOT, name))
        self.annotator = self.ann_var.get()
        if not _protocol_ok(self.pack):
            messagebox.showerror('标签协议不匹配',
                                 '该抽样包不是当前252类标签协议，或公开清单/图片内容校验失败。\n'
                                 '旧254类第一轮盲标包已失效，请重新生成。')
            return
        sheet = os.path.join(self.pack, f'sheet{self.annotator}.csv')
        if not os.path.exists(sheet):
            messagebox.showerror('错误', f'找不到标注表：\n{sheet}')
            return
        with open(sheet, encoding='utf-8-sig', newline='') as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            messagebox.showerror('错误', '标注表为空')
            return
        self.sheet_path = sheet
        img_dir = os.path.join(self.pack, 'images')
        if os.path.isdir(img_dir):
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=2)
        resume_index = first_unanswered_index(self.rows)
        self.idx = 0 if resume_index is None else resume_index
        self._build_main()
        self.render()
        if resume_index is None:
            self._toast('本轮240题均已作答；如需修改可用左右键定位原题')
        else:
            remaining = sum(1 for row in self.rows if not is_answered(row))
            self._toast(f'已跳到第一道未答题：第 {resume_index + 1} 张（剩余 {remaining} 题）', kind='warn')

    def _build_main(self):
        self._clear()
        top = tk.Frame(self.container, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        top.pack(fill='x')
        tk.Label(top, text='人工标注 · 类对二选一', font=self.F_TITLE, bg=CARD, fg=TEXT)\
            .pack(side='left', padx=(sp(18), sp(10)), pady=sp(12))
        tk.Label(top, text=f'{os.path.basename(self.pack)}　|　标注轮次 {self.annotator}',
                 font=self.F_SMALL, bg=CARD, fg=SUBTLE).pack(side='left', pady=sp(18))
        tk.Label(top, text='协议v3 · 公共类子类参考：显示',
                 font=self.F_SMALL, bg=CARD, fg=OK_TEXT)\
            .pack(side='left', padx=(sp(10), 0), pady=sp(18))
        self.save_hint = tk.Label(top, text='自动保存已开启', font=self.F_SMALL, bg=CARD, fg=OK_TEXT)
        self.save_hint.pack(side='right', padx=sp(18))
        self.top_prog = tk.Label(top, text='', font=self.F_H2, bg=CARD, fg=TEXT)
        self.top_prog.pack(side='right', padx=(0, sp(16)))

        body = tk.Frame(self.container, bg=BG)
        body.pack(fill='both', expand=True, padx=sp(16), pady=sp(16))

        left = tk.Frame(body, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        left.pack(side='left', fill='both', expand=True)
        self.canvas = tk.Canvas(left, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=sp(10), pady=(sp(10), sp(4)))
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        foot = tk.Frame(left, bg=CARD)
        foot.pack(fill='x', padx=sp(12), pady=(0, sp(10)))
        tk.Label(foot, textvariable=self.pair_var, font=self.F_SMALL, bg=CARD, fg=SUBTLE)\
            .pack(side='left')

        right = tk.Frame(body, bg=CARD, width=self.right_w, highlightbackground=BORDER, highlightthickness=1)
        right.pack(side='left', fill='y', padx=(sp(16), 0))
        right.pack_propagate(False)
        rp = sp(16)

        tk.Label(right, text='进度', font=self.F_H2, bg=CARD, fg=TEXT).pack(anchor='w', padx=rp, pady=(sp(16), 0))
        num = tk.Frame(right, bg=CARD)
        num.pack(anchor='w', padx=rp, pady=(sp(2), sp(2)))
        tk.Label(num, textvariable=self.prog_var, font=self.F_BIG, bg=CARD, fg=TEXT).pack(side='left')
        tk.Label(num, textvariable=self.prog_pct, font=self.F_BODY, bg=CARD, fg=SUBTLE)\
            .pack(side='left', padx=(sp(8), 0), pady=(sp(8), 0))
        self.bar = tk.Canvas(right, height=sp(10), bg=CARD, highlightthickness=0)
        self.bar.pack(fill='x', padx=rp, pady=(sp(6), sp(14)))
        self.bar.bind('<Configure>', lambda e: self._draw_bar())

        tk.Label(right, text='图中物体属于哪一类？', font=self.F_H2, bg=CARD, fg=TEXT)\
            .pack(anchor='w', padx=rp, pady=(0, sp(8)))

        self.opt_frame = tk.Frame(right, bg=CARD)
        self.opt_frame.pack(fill='x', padx=rp)
        self.btn1 = tk.Button(self.opt_frame, text='', font=self.F_OPT, bg=OPT_STYLE['1']['idle'], fg=TEXT,
                              activebackground=OPT_STYLE['1']['idle_hover'], activeforeground=TEXT, relief='flat',
                              wraplength=self.right_w - sp(64), justify='center', takefocus=0,
                              cursor='hand2', highlightthickness=2,
                              highlightbackground=OPT_STYLE['1']['ring'],
                              highlightcolor=OPT_STYLE['1']['ring'],
                              command=lambda: self.answer(1))
        self.btn1.pack(fill='x', ipady=sp(13), pady=(0, sp(8)))
        self.btn2 = tk.Button(self.opt_frame, text='', font=self.F_OPT, bg=OPT_STYLE['2']['idle'], fg=TEXT,
                              activebackground=OPT_STYLE['2']['idle_hover'], activeforeground=TEXT, relief='flat',
                              wraplength=self.right_w - sp(64), justify='center', takefocus=0,
                              cursor='hand2', highlightthickness=2,
                              highlightbackground=OPT_STYLE['2']['ring'],
                              highlightcolor=OPT_STYLE['2']['ring'],
                              command=lambda: self.answer(2))
        self.btn2.pack(fill='x', ipady=sp(13))
        # 悬停/离开时按"当前这一题的已选状态"重刷配色：即使 Tk 残留 active 态，
        # 也保证"没选的那颗"永远用淡底悬停色，不会串成已选的实心色。
        for _b in (self.btn1, self.btn2):
            _b.bind('<Enter>', lambda e: self._hl(self._cur_choice()))
            _b.bind('<Leave>', lambda e: self._hl(self._cur_choice()))

        # 本题当前答案（显式回显，避免"我到底选了哪个"的疑虑）
        self.answer_label = tk.Label(right, textvariable=self.answer_var, font=self.F_BODY,
                                     bg=CARD, fg=SUBTLE, anchor='w')
        self.answer_label.pack(fill='x', padx=rp, pady=(sp(10), 0))

        self.btn_skip = tk.Button(right, text='无法判断 (0)', font=self.F_BODY, bg=GHOST, fg=WARN_DEEP,
                                  activebackground=GHOST_HOVER, activeforeground=WARN_DEEP,
                                  relief='flat', takefocus=0, cursor='hand2', highlightthickness=0,
                                  command=lambda: self.answer(0))
        self.btn_skip.pack(fill='x', padx=rp, pady=(sp(12), 0), ipady=sp(8))
        self.btn_skip.bind('<Enter>', lambda e: self._hl(self._cur_choice()))
        self.btn_skip.bind('<Leave>', lambda e: self._hl(self._cur_choice()))

        # 「图外/异常」标记：两个候选都不像（或图片本身有问题）时打标，写进 notes
        self.btn_flag = tk.Button(right, text='图外·异常（X）', font=self.F_BODY, bg=GHOST, fg=SUBTLE,
                                  activebackground=GHOST_HOVER, activeforeground=SUBTLE,
                                  relief='flat', takefocus=0, cursor='hand2', highlightthickness=0,
                                  command=self.toggle_outlier)
        self.btn_flag.pack(fill='x', padx=rp, pady=(sp(6), 0), ipady=sp(7))
        self.btn_flag.bind('<Enter>', lambda e: self._flag_style())
        self.btn_flag.bind('<Leave>', lambda e: self._flag_style())
        tk.Label(right, text='（两个候选类都不属于 / 图片本身有问题时点它，记为 [图外]；'
                             '不影响 choice，可再点一次取消）',
                 font=self.F_SMALL, bg=CARD, fg=SUBTLE, justify='left',
                 wraplength=self.right_w - sp(40)).pack(anchor='w', padx=rp, pady=(sp(2), 0))

        nav = tk.Frame(right, bg=CARD)
        nav.pack(fill='x', padx=rp, pady=(sp(10), 0))
        nav.grid_columnconfigure(0, weight=1)
        nav.grid_columnconfigure(1, weight=1)
        self.btn_prev = tk.Button(nav, text='← 上一张', font=self.F_SMALL, bg=GHOST, fg=TEXT,
                                  relief='flat', takefocus=0, cursor='hand2', highlightthickness=0,
                                  command=lambda: self.move(-1))
        self.btn_prev.grid(row=0, column=0, sticky='ew', padx=(0, sp(6)), ipady=sp(7))
        self.btn_next = tk.Button(nav, text='下一张 →', font=self.F_SMALL, bg=GHOST, fg=TEXT,
                                  relief='flat', takefocus=0, cursor='hand2', highlightthickness=0,
                                  command=lambda: self.move(1))
        self.btn_next.grid(row=0, column=1, sticky='ew', padx=(0, sp(6)), ipady=sp(7))
        self.btn_clear = tk.Button(nav, text='清除', font=self.F_SMALL, bg=GHOST, fg=EMPTY,
                                   relief='flat', takefocus=0, cursor='hand2', width=5, state='disabled',
                                   highlightthickness=0, command=self.clear_current)
        self.btn_clear.grid(row=0, column=2, sticky='ew', ipady=sp(7))

        tk.Label(right, text='备注（可选，Ctrl+S 保存）', font=self.F_SMALL, bg=CARD, fg=SUBTLE)\
            .pack(anchor='w', padx=rp, pady=(sp(14), sp(4)))
        self.notes_entry = tk.Entry(right, textvariable=self.notes_var, font=self.F_BODY, relief='flat',
                                    highlightbackground=BORDER, highlightcolor=PRIMARY,
                                    highlightthickness=2, bg='#FCFDFF', takefocus=1)
        self.notes_entry.pack(fill='x', padx=rp, ipady=sp(5))
        self.notes_entry.bind('<Return>', lambda e: self.save(silent=True))

        tk.Label(right, text=('选择：1 左　2 右　0 无法判断\n'
                              '翻页：← →　　清除：Delete（Ctrl+Z 撤销）\n'
                              '选错了：按 ← 回到该题，重新点另一个选项即可覆盖\n'
                              '保存：Ctrl+S　　退出：Ctrl+Q / Esc'),
                 font=self.F_SMALL, bg=CARD, fg=SUBTLE, justify='left')\
            .pack(anchor='w', padx=rp, pady=(sp(14), 0))

        self.status_label = tk.Label(right, textvariable=self.status_var, font=self.F_SMALL,
                                     bg=CARD, fg=SUBTLE, wraplength=self.right_w - sp(40),
                                     justify='left', height=2, anchor='w')
        self.status_label.pack(side='bottom', fill='x', padx=rp, pady=sp(14))

    # ------------------------------------------------------------------ 渲染
    def _done_ratio(self):
        n = len(self.rows) or 1
        done = sum(1 for x in self.rows if is_answered(x))
        return done / n

    def _draw_bar(self):
        c = self.bar
        c.delete('all')
        w = max(10, c.winfo_width())
        h = sp(8)
        y = sp(1)
        r = h // 2
        for x0, x1, col in ((0, w, BAR_TRACK), (0, int(w * self._done_ratio()), PRIMARY)):
            if x1 - x0 <= 0:
                continue
            c.create_oval(x0, y, x0 + h, y + h, fill=col, outline='')
            c.create_oval(x1 - h, y, x1, y + h, fill=col, outline='')
            c.create_rectangle(x0 + r, y, x1 - r, y + h, fill=col, outline='')

    def _opt_text(self, k, name, answered):
        hint = public_class_member_hint(name)
        suffix = f'\n公共合并类包含：{hint}' if hint else ''
        return f'{"✓" if answered else " "} {k}　{name}{suffix}'

    def render(self):
        r = self.rows[self.idx]
        done = sum(1 for x in self.rows if is_answered(x))
        remaining = len(self.rows) - done
        abst = sum(1 for x in self.rows if (x.get('choice') or '').strip() == '0')
        self.pair_var.set(f'第 {self.idx + 1} / {len(self.rows)} 张　·　类对 {r["pair_id"]}　·　'
                          f'（图片已随机命名）')
        self.top_prog.configure(text=f'已答 {done} / {len(self.rows)} · 未答 {remaining}')
        self.prog_var.set(f'{done} / {len(self.rows)}')
        self.prog_pct.set(f'{int(done * 100 / max(1, len(self.rows)))}%　（无法判断 {abst}）')
        self._draw_bar()

        cur = (r.get('choice') or '').strip()
        if cur == '1':
            self.answer_var.set(f'本题已选：① {r["option_1"]}')
        elif cur == '2':
            self.answer_var.set(f'本题已选：② {r["option_2"]}')
        elif cur == '0':
            self.answer_var.set('本题已选：无法判断')
        else:
            self.answer_var.set('本题未作答')
        self.answer_label.configure(fg=(OK_TEXT if cur else SUBTLE))
        self.btn1.configure(text=self._opt_text(1, r['option_1'], cur == '1'))
        self.btn2.configure(text=self._opt_text(2, r['option_2'], cur == '2'))
        self._hl(cur)
        self.btn_clear.configure(state=('normal' if cur else 'disabled'),
                                 fg=(WARN_DEEP if cur else EMPTY), cursor=('hand2' if cur else 'arrow'))
        self._flag_style()
        self.notes_var.set(r.get('notes') or '')
        self.render_image()
        self._prefetch()

    def _flag_on(self, row=None):
        r = row if row is not None else (self.rows[self.idx] if self.rows else {})
        return OUTLIER_TOKEN in (r.get('notes') or '')

    def _flag_style(self):
        """「图外」按钮外观随当前行的标记状态变化（未标=灰、已标=警告色实底）。"""
        on = self._flag_on()
        self.btn_flag.configure(text=('✓ 已标记「图外」· 再点取消 (X)' if on else '图外·异常（X）'),
                                bg=(WARN if on else GHOST),
                                fg=('white' if on else SUBTLE),
                                activebackground=(WARN_DEEP if on else GHOST_HOVER),
                                activeforeground=('white' if on else SUBTLE))

    def toggle_outlier(self):
        """给当前题打/取消「图外·异常」标记（写入 notes 的固定标记词；不动 choice）。"""
        if not self.rows or not self.img_ok:
            return
        r = self.rows[self.idx]
        cur = self.notes_var.get() or ''
        if OUTLIER_TOKEN in cur:
            new, on = cur.replace(OUTLIER_TOKEN, '').strip(), False
        else:
            new, on = (OUTLIER_TOKEN + ' ' + cur).strip(), True
        self.notes_var.set(new)
        r['notes'] = new
        self.save(silent=True)
        self._flag_style()
        self._toast('已标记为「图外·异常」（两个候选都不属于/图片有问题）' if on
                    else '已取消「图外·异常」标记', kind=('warn' if on else 'ok'))

    def _cur_choice(self):
        if not self.rows:
            return ''
        return (self.rows[self.idx].get('choice') or '').strip()

    def _hl(self, cur):
        """选项按钮配色（未选/悬停/已选三态）。**已选必须比悬停更醒目**，否则返回上一题时会把"没选的那颗"看成已选。

        2026-09-12 修（用户反馈"选完后返回上一题，选中颜色不对"）：原方案"未选 = 实心深色、已选 = 淡色 tint"
        视觉倒挂——指针停在按钮上时 Tk 进入 active 态绘制 activebackground（当时是更深一档的实心色），
        键盘翻页又不会复位该态，于是返回上一题常看到"深色那颗像被选中、淡色那颗像没选"；且 highlightcolor
        未设时按钮一旦获得焦点描边会变成 SystemWindowFrame（深灰）盖掉绿圈。现改为按 OPT_STYLE：
          未选 = 极淡底 + 同色相中灰描边；悬停 = 同色相深一档（永不实心）；已选 = 实心彩色 + 白字 + 绿圈 + ✓。
        并把 highlightcolor 与 highlightbackground 设成同色，焦点态不改变描边含义。
        """
        for btn, k in ((self.btn1, '1'), (self.btn2, '2')):
            st = OPT_STYLE[k]
            on = (cur == k)
            bg = st['solid'] if on else st['idle']
            hv = st['solid_hover'] if on else st['idle_hover']
            fg = 'white' if on else TEXT
            ring = OK if on else st['ring']
            btn.configure(bg=bg, fg=fg, activebackground=hv, activeforeground=fg,
                          highlightbackground=ring, highlightcolor=ring)
        skip_on = (cur == '0')
        self.btn_skip.configure(text=('✓ 无法判断 (0)' if skip_on else '无法判断 (0)'),
                                bg=(OK_TINT if skip_on else GHOST),
                                fg=(OK_TEXT_DEEP if skip_on else WARN_DEEP),
                                activebackground=(OK_TINT if skip_on else GHOST_HOVER),
                                activeforeground=(OK_TEXT_DEEP if skip_on else WARN_DEEP))

    def _on_canvas_configure(self, _e):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self.render_image)

    def _load_pil(self, j):
        """在后台线程解码（仅 PIL，不碰 Tk）；JPEG 用 draft 提前降采样以加速。"""
        from PIL import Image
        r = self.rows[j]
        path = os.path.join(self.pack, 'images', r['image'])
        im = Image.open(path)
        try:
            im.draft('RGB', (1600, 1600))
        except Exception:
            pass
        return im.convert('RGB')

    def _prefetch(self):
        if not self._pool:
            return
        for j in (self.idx + 1, self.idx + 2, self.idx - 1):
            if 0 <= j < len(self.rows) and j not in self._pil_cache:
                try:
                    self._pil_cache[j] = self._pool.submit(self._load_pil, j)
                except Exception:
                    pass
        if len(self._pil_cache) > 8:
            for k in sorted(self._pil_cache)[:-6]:
                self._pil_cache.pop(k, None)

    def render_image(self):
        if not self.rows:
            return
        from PIL import Image, ImageTk
        j = self.idx
        pil = None
        fut = self._pil_cache.get(j)
        if fut is not None:
            try:
                pil = fut.result(timeout=5)
            except Exception:
                pil = None
        if pil is None:
            try:
                pil = self._load_pil(j)
            except Exception as e:
                self.cur_pil, self.img_ok = None, False
                self.canvas.delete('all')
                self.canvas.create_text(sp(16), sp(16), anchor='nw', fill='#B91C1C', font=self.F_BODY,
                                        text=f'⚠ 图片无法打开：{self.rows[j]["image"]}\n{e}')
                for b in (self.btn1, self.btn2, self.btn_skip):
                    b.configure(state='disabled')
                self._toast('本张图片损坏：已禁用作答，请按 → 跳过（不会记录答案）', kind='warn')
                return
        self.cur_pil, self.img_ok = pil, True
        for b in (self.btn1, self.btn2, self.btn_skip):
            b.configure(state='normal')
        cw = max(80, self.canvas.winfo_width() - sp(20))
        ch = max(80, self.canvas.winfo_height() - sp(20))
        w, h = pil.size
        scale = min(cw / w, ch / h, 1.0)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        self.img_cache = ImageTk.PhotoImage(pil.resize((tw, th), Image.LANCZOS))
        self.canvas.delete('all')
        cx, cy = self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2
        self.canvas.create_image(cx, cy, image=self.img_cache)
        self.canvas.create_text(cx, self.canvas.winfo_height() - sp(12), fill=SUBTLE, font=self.F_SMALL,
                                text=f'原图 {w}×{h} · 显示 {int(scale * 100)}%')

    # ------------------------------------------------------------------ 交互
    def _toast(self, msg, kind='ok'):
        self.status_var.set(msg)
        self.status_label.configure(fg=(OK_TEXT if kind == 'ok' else WARN))
        if self._toast_job:
            try:
                self.root.after_cancel(self._toast_job)
            except Exception:
                pass
        self._toast_job = self.root.after(1600, lambda: (self.status_var.set(''),
                                                         self.status_label.configure(fg=SUBTLE)))

    def answer(self, choice):
        if not self.rows or not self.img_ok:
            return
        r = self.rows[self.idx]
        prev = (r.get('choice') or '').strip()
        prev_notes = r.get('notes') or ''
        r['choice'] = str(choice)
        if self.notes_var.get() != (r.get('notes') or ''):
            r['notes'] = self.notes_var.get()
        if not self.save(silent=True):
            r['choice'], r['notes'] = prev, prev_notes
            self.render()
            return
        name = {1: r['option_1'], 2: r['option_2'], 0: '无法判断'}.get(choice, '')
        if prev and prev != str(choice):
            self._toast(f'↻ 已改答：{"选项 " + str(choice) if choice else ""} {name}（原：选项 {prev}）')
        else:
            self._toast(f'✓ 已记录：{("选项 " + str(choice) + " · ") if choice else ""}{name}')
        # 优先跳到下一道未答题；补漏时不再落回已经答过的顺序页。
        # 全部完成或只是修改既有答案时留在当前题，明确刷新进度与选中状态。
        nxt = next_unanswered_index(self.rows, self.idx)
        if nxt is None:
            self.render()
            if all(is_answered(row) for row in self.rows):
                self._toast('✓ 本轮全部题目已完成')
        else:
            self.idx = nxt
            self.render()

    def clear_current(self):
        if not self.rows or not self.img_ok:
            return
        r = self.rows[self.idx]
        if not (r.get('choice') or '').strip():
            return
        self._undo = (self.idx, r.get('choice', ''), r.get('notes', ''), datetime.now())
        r['choice'] = ''
        self.save(silent=True)
        self.render()
        self._toast('已清除本题（4 秒内按 Ctrl+Z 可撤销）', kind='warn')

    def undo_clear(self):
        if not self._undo:
            return
        i, ch, nt, ts = self._undo
        if (datetime.now() - ts).total_seconds() > 4:
            self._undo = None
            return
        self.rows[i]['choice'], self.rows[i]['notes'] = ch, nt
        self.save(silent=True)
        self.idx = i
        self._undo = None
        self.render()
        self._toast('已撤销清除')

    def move(self, delta):
        if not self.rows:
            return
        nxt = max(0, min(len(self.rows) - 1, self.idx + delta))
        if nxt == self.idx and delta:
            self._toast('已经是第一张了' if delta < 0 else '已经是最后一张了', kind='warn')
            return
        self.idx = nxt
        self.render()

    def save(self, silent=False):
        """原子写：临时文件 + os.replace；失败不静默（提示且阻止退出）。"""
        if not self.sheet_path:
            return False
        tmp = self.sheet_path + '.tmp'
        fields = ['pair_id', 'image', 'option_1', 'option_2', 'choice', 'notes']
        try:
            with open(tmp, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in self.rows:
                    w.writerow({k: r.get(k, '') for k in fields})
            os.replace(tmp, self.sheet_path)
        except OSError as e:
            self._toast(f'⚠ 保存失败：{e}　请勿关闭窗口，按 Ctrl+S 重试', kind='warn')
            return False
        ts = datetime.now().strftime('%H:%M:%S')
        self.save_hint.configure(text=f'已保存 {os.path.basename(self.sheet_path)} · {ts}')
        if not silent:
            self._toast('已保存')
        return True

    def quit_app(self):
        ok = self.save(silent=True)
        done = sum(1 for x in self.rows if (x.get('choice') or '').strip())
        if not ok:
            if not messagebox.askyesno('保存失败', '标注表写入失败，仍要退出吗？\n离开将丢失本次未保存的修改。'):
                return
        elif done < len(self.rows):
            if not messagebox.askyesno('确认退出', f'还有 {len(self.rows) - done} 张未作答，仍要退出吗？\n'
                                                   f'（已作答内容已自动保存）'):
                return
        try:
            if self._pool:
                self._pool.shutdown(wait=False)
        except Exception:
            pass
        self.root.destroy()

    def _clear(self):
        for w in self.container.winfo_children():
            w.destroy()


def option_style_invariants(app):
    """选项按钮四态（未答/已选1/已选2/无法判断）的配色不变量，返回 [(名称, 是否通过, 详情)]。

    为什么需要：2026-09-12 用户反馈"选完后返回上一题，已选的配色看着不对"。根因是 Tk 经典 Button 在指针
    停留其内时进入 active 态、画 activebackground，而该状态在键盘翻页后不会复位；配合当时"未选=实心深色、
    已选=淡色 tint"的倒挂配色，就出现"没选的那颗（被指针压着、画更深一档的实心色）看着像已选"。此外
    highlightcolor 未设时，按钮一旦获得焦点描边会变成 SystemWindowFrame。以下不变量把修复钉住：
      ① 悬停色永远不是实心彩色；② 焦点描边 == 常态描边；③ 已选=实心彩色+白字+绿圈；④ 未选=淡底深字+灰圈。
    """
    out = []

    def chk(name, cond, detail=''):
        out.append((name, bool(cond), detail))

    for cur in ('', '1', '2', '0'):
        tag = cur or '未答'
        app._hl(cur)
        for k, b in (('1', app.btn1), ('2', app.btn2)):
            st = OPT_STYLE[k]
            on = (cur == k)
            ring, ring_focus = b.cget('highlightbackground'), b.cget('highlightcolor')
            chk(f'[{tag}] btn{k} 焦点描边 == 常态描边（不会变 SystemWindowFrame）',
                ring == ring_focus, f'{ring}/{ring_focus}')
            chk(f'[{tag}] btn{k} 悬停色不是实心彩色（悬停不会被误读成已选）',
                b.cget('activebackground') not in (PRIMARY, CAND2), str(b.cget('activebackground')))
            if on:
                chk(f'[{tag}] btn{k} 已选 = 实心彩色 + 白字 + 绿圈',
                    b.cget('bg') == st['solid'] and b.cget('fg') == 'white' and ring == OK,
                    str(b.cget('bg')))
            else:
                chk(f'[{tag}] btn{k} 未选 = 极淡底 + 同色相描边 + 深字',
                    b.cget('bg') == st['idle'] and b.cget('fg') == TEXT and ring == st['ring'],
                    str(b.cget('bg')))
        if cur in ('1', '2'):
            sel, oth = (app.btn1, app.btn2) if cur == '1' else (app.btn2, app.btn1)
            chk(f'[{tag}] 已选按钮底色 != 另一颗的悬停色（两者不会撞色）',
                sel.cget('bg') != oth.cget('activebackground'),
                f'{sel.cget("bg")} vs {oth.cget("activebackground")}')
    app._hl('0')
    chk('无法判断被选中 = ✓ + 绿字',
        '✓' in app.btn_skip.cget('text') and app.btn_skip.cget('fg') == OK_TEXT_DEEP,
        app.btn_skip.cget('text'))
    app._hl('')
    chk('无法判断未选中 = 无 ✓ + 警告色',
        '✓' not in app.btn_skip.cget('text') and app.btn_skip.cget('fg') == WARN_DEEP,
        app.btn_skip.cget('text'))
    # 恢复成当前行真实作答，并核对"渲染结果与真实作答一致"（✓ 前缀只由 render() 决定）
    app.render()          # _hl 只在内存里改颜色，不动数据
    real = app._cur_choice()
    chk('渲染后 ✓ 前缀与真实作答一致',
        ('✓' in app.btn1.cget('text')) == (real == '1')
        and ('✓' in app.btn2.cget('text')) == (real == '2'), f'真实作答={real!r}')
    chk('渲染后配色与真实作答一致',
        (app.btn1.cget('bg') == OPT_STYLE['1']['solid']) == (real == '1')
        and (app.btn2.cget('bg') == OPT_STYLE['2']['solid']) == (real == '2'),
        f'btn1={app.btn1.cget("bg")} btn2={app.btn2.cget("bg")}')
    return out


def main():
    ap = argparse.ArgumentParser(description='人工标注 GUI（类对二选一，盲标）')
    ap.add_argument('--pack', default=None, help='抽样包目录（缺省用 exports 下最新的）')
    ap.add_argument('--annotator', default='A', help='标注轮次（A / B，决定写 sheet<A|B>.csv）')
    ap.add_argument('--ui-selftest', action='store_true',
                    help='自检：构建界面 + 模拟作答（强制 UITEST 标注者，不碰真实标注表）')
    a = ap.parse_args()

    selftest = a.ui_selftest
    annotator = 'UITEST' if (selftest and a.annotator in ('A', 'B')) else a.annotator

    try:
        root = tk.Tk()
    except Exception as e:
        print(f'[gui] 无法启动 tkinter：{e}')
        return 1

    pack = a.pack
    if pack and not os.path.isabs(pack):
        cand = os.path.join(EXPORT_ROOT, pack)
        pack = cand if os.path.isdir(cand) else pack
    if pack is None:
        ps = packs_available()
        pack = ps[0] if ps else None

    app = AnnotatorApp(root, pack=pack, annotator=annotator)

    if selftest:
        sheet = os.path.join(pack, f'sheet{annotator}.csv') if pack else None
        created = False
        gap_indexes = None
        try:
            if sheet and not os.path.exists(sheet):
                src = os.path.join(pack, 'sheetA.csv')
                if os.path.exists(src):
                    import shutil
                    shutil.copy2(src, sheet)
                    created = True
            if created:
                # 构造两个分散空白题，覆盖“重新打开→首个未答→答后跳到下一未答→进度更新”。
                with open(sheet, encoding='utf-8-sig', newline='') as f:
                    fixture_rows = list(csv.DictReader(f))
                for fixture_row in fixture_rows:
                    fixture_row['choice'] = '1'
                    fixture_row['notes'] = ''
                gap_indexes = [3, 8] if len(fixture_rows) > 8 else [0, 1]
                for gap_index in gap_indexes:
                    fixture_rows[gap_index]['choice'] = ''
                with open(sheet, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(
                        f, fieldnames=['pair_id', 'image', 'option_1', 'option_2', 'choice', 'notes'])
                    writer.writeheader()
                    writer.writerows(fixture_rows)
            app.ann_var.set(annotator)
            app.start_session()
            if not app.rows:
                print('[gui-selftest] 失败：未能加载标注表')
                return 1
            # 会话从"第一道未答题"续标，故测试须针对 app.idx 指向的那一行
            # （早期版本写死 rows[0]，当抽样表已有作答时会误判为 FAIL）
            k = getattr(app, 'idx', 0)
            resume_ok = (k == gap_indexes[0]) if gap_indexes else True
            before = app.rows[k].get('choice', '')
            app.answer(1)
            written = app.rows[k].get('choice') == '1'
            jump_ok = (app.idx == gap_indexes[1]) if gap_indexes else True
            progress_ok = ((f'已答 {len(app.rows) - 1} / {len(app.rows)} · 未答 1'
                            == app.top_prog.cget('text')) if gap_indexes else True)
            # 复核磁盘内容
            with open(sheet, encoding='utf-8-sig', newline='') as f:
                disk = list(csv.DictReader(f))
            on_disk = disk[k].get('choice') == '1'
            print(f'[gui-selftest] 加载 {len(app.rows)} 行；写入行=#{k + 1}；'
                  f'内存写入={"OK" if written else "FAIL"}；'
                  f'磁盘复核={"OK" if on_disk else "FAIL"}；原值={before!r}；标注者={annotator}（真实表未改动）')
            print(f'[gui-selftest] 断点续标：首个未答定位={"OK" if resume_ok else "FAIL"}／'
                  f'下一未答跳转={"OK" if jump_ok else "FAIL"}／'
                  f'进度即时更新={"OK" if progress_ok else "FAIL"}')
            # 补完第二个空白，使后续“最后一题原地回显”测试处于全题已答状态。
            if gap_indexes:
                app.answer(2)
            # 选项按钮配色不变量（已选/未选/悬停/焦点，防止"返回上一题配色倒挂"回归）
            checks = option_style_invariants(app)
            bad = [c for c in checks if not c[1]]
            print(f'[gui-selftest] 配色不变量：通过 {len(checks) - len(bad)} / {len(checks)} 项')
            for name, ok, detail in bad:
                print(f'    FAIL {name}　{detail}')
            glass_index = next((i for i, row in enumerate(app.rows)
                                if '玻璃制品类' in (row.get('option_1'), row.get('option_2'))), None)
            if glass_index is not None:
                app.idx = glass_index
                app.render()
                hint_visible = ('公共合并类包含：玻璃器皿、玻璃瓶、玻璃壶、玻璃制品' in app.btn1.cget('text')
                                or '公共合并类包含：玻璃器皿、玻璃瓶、玻璃壶、玻璃制品' in app.btn2.cget('text'))
                hint_ok = hint_visible
            else:
                hint_ok = False
            print(f'[gui-selftest] 公共类子类参考（协议v3显示模式）={"OK" if hint_ok else "FAIL"}')
            # 最后一题答完应原地重绘并给出"已记录"（不能被"已经是最后一张了"盖掉）
            if len(app.rows) > 0:
                app.idx = len(app.rows) - 1
                app.render()
                app.answer(2)
                last_ok = (app.rows[-1].get('choice') == '2' and app.idx == len(app.rows) - 1
                           and '②' in app.answer_var.get()
                           and ('已记录' in app.status_var.get()
                                or '全部题目已完成' in app.status_var.get()))
                print(f'[gui-selftest] 最后一题原地重绘与回显={"OK" if last_ok else "FAIL"}'
                      f'（回显={app.answer_var.get()!r}；提示={app.status_var.get()!r}）')
            else:
                last_ok = True
            # 「图外/异常」标记：开→notes 写入标记词且落盘；再点→移除（净效果为零）
            flag_ok = True
            try:
                k2 = getattr(app, 'idx', 0)
                app.toggle_outlier()
                on_mem = OUTLIER_TOKEN in (app.rows[k2].get('notes') or '')
                with open(sheet, encoding='utf-8-sig', newline='') as f:
                    disk3 = list(csv.DictReader(f))
                on_disk3 = OUTLIER_TOKEN in (disk3[k2].get('notes') or '')
                app.toggle_outlier()
                off_mem = OUTLIER_TOKEN not in (app.rows[k2].get('notes') or '')
                flag_ok = on_mem and on_disk3 and off_mem
                print(f'[gui-selftest] 「图外」标记：写入={"OK" if on_mem else "FAIL"}／'
                      f'落盘={"OK" if on_disk3 else "FAIL"}／取消={"OK" if off_mem else "FAIL"}')
            except Exception as e:
                flag_ok = False
                print(f'[gui-selftest] 「图外」标记测试异常：{e}')
            return 0 if (resume_ok and jump_ok and progress_ok and written and on_disk
                         and hint_ok and last_ok and flag_ok and not bad) else 1
        finally:
            # 即使输出编码、图片解码或界面断言中途异常，也不能留下测试标注表。
            if created:
                try:
                    os.remove(sheet)
                except OSError:
                    pass
            elif sheet and os.path.exists(sheet):
                print(f'[gui-selftest] 注意：{os.path.basename(sheet)} 并非本次创建，已保留（非真实标注表）')
            try:
                root.destroy()
            except tk.TclError:
                pass

    root.mainloop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
