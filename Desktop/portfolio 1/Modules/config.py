# Shared constants, palette, macro events, plot helpers
# Tất cả module import từ đây — không duplicate

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── rcParams ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'axes.titlecolor':  '#f0f6fc',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.5,
    'font.family':      'monospace',
    'axes.titlesize':   11,
    'axes.labelsize':   9,
    'figure.dpi':       120,
})

# ── Colour palette ────────────────────────────────────────────────────────────
BG     = '#0d1117'
BG2    = '#161b22'
BORDER = '#30363d'
TEXT   = '#f0f6fc'
ACCENT = '#58a6ff'
OK     = '#3fb950'
WARN   = '#f78166'
YELLOW = '#d29922'
PURPLE = '#bc8cff'
GRAY   = '#8b949e'

REGIME_COLOR = {
    'NORMAL'  : OK,
    'ELEVATED': YELLOW,
    'STRESS'  : WARN,
    'CRISIS'  : '#ff6e6e',
    'UNKNOWN' : GRAY,
}
ALERT_COLOR = {0: OK, 1: YELLOW, 2: WARN, 3: '#ff6e6e'}

# ── Macro events ──────────────────────────────────────────────────────────────
MACRO_EVENTS = {
    '2020-03-23': 'COVID crash',
    '2022-02-24': 'Russia-Ukraine',
    '2022-06-15': 'Fed +75bps',
    '2022-09-22': 'SBV rate hike',
    '2023-10-19': 'SBV intervention',
}

# ── Market periods (M4A, M5) ──────────────────────────────────────────────────
PERIODS = {
    'Pre-COVID\n(2018-19)'   : ('2018-01-01', '2020-02-19'),
    'COVID Shock\n(2020 H1)' : ('2020-02-20', '2020-06-30'),
    'Recovery\n(2020-21)'    : ('2020-07-01', '2021-12-31'),
    'Fed Hike\n(2022)'       : ('2022-01-01', '2022-12-31'),
    'Post-Hike\n(2023-24)'   : ('2023-01-01', '2024-12-31'),
}

# ── Shared plot helpers ───────────────────────────────────────────────────────

def style_fig(fig, axes):
    """Apply dark theme."""
    fig.patch.set_facecolor(BG)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(BG2)
        ax.tick_params(colors=GRAY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)


def annotate_events(axes, top=0):
    """Vẽ macro-event vlines trên tất cả axes, label chỉ trên panel top."""
    ax_list = list(axes) if hasattr(axes, '__iter__') else [axes]
    for dt, name in MACRO_EVENTS.items():
        try:
            xpos = pd.Timestamp(dt)
            for ax in ax_list:
                ax.axvline(xpos, color='white', alpha=0.18, lw=0.7, ls='--')
            ylim  = ax_list[top].get_ylim()
            ytext = ylim[0] + (ylim[1] - ylim[0]) * 0.88
            ax_list[top].text(
                xpos, ytext, name,
                rotation=90, fontsize=6, color='white',
                ha='right', va='top', alpha=0.55,
                bbox=dict(boxstyle='round,pad=0.1',
                          fc=BG, ec='none', alpha=0.5))
        except Exception:
            pass

def fig_footer(fig, text):
    fig.text(0.99, 0.003, text, ha='right', fontsize=6,
             color=GRAY, style='italic')
