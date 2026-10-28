"""Shared typography, canvas and configuration colors for thesis figures."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIZE = (7.6, 5.2)
COLORS = {
    'S-NoProg': '#3C6EAB', 'S-NoSE': '#7B8794', 'S-StdSE': '#2A8C74',
    'V2-Full': '#7656A0', 'V2-NoTF': '#D78A35',
    'G-Full-CAWR': '#42A3B5', 'G-V2': '#C75664',
}


def configure():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei'],
        'axes.unicode_minus': False, 'font.size': 10,
        'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'lines.linewidth': 1.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'grid.alpha': .22,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
    })


def canvas(rows=1, cols=1, **kwargs):
    fig, axes = plt.subplots(rows, cols, figsize=SIZE, **kwargs)
    return fig, axes


def save(fig, directory, name, *, tight_layout_kwargs=None):
    directory = Path(directory)
    fig.set_dpi(300)
    layout = dict(pad=1.3, w_pad=1.0, h_pad=1.0)
    layout.update(tight_layout_kwargs or {})
    fig.tight_layout(**layout)
    fig.canvas.draw()
    fig.set_layout_engine(None)
    fig.savefig(directory / f'{name}.pdf')
    import pymupdf
    with pymupdf.open(directory / f'{name}.pdf') as document:
        document[0].get_pixmap(dpi=300, alpha=False).save(directory / f'{name}.png')
    plt.close(fig)
