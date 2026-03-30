"""
config.py — Shared constants, style settings, and domain metadata.
All modules import from here to stay consistent.
"""

import matplotlib.pyplot as plt

# ── Model constants ────────────────────────────────────────────────────────────
DRIVERS = ['ΔDXY', 'ΔVIX', 'ΔUS10Y', 'ΔOIL']
TARGET  = 'r_usdjpy'

# ── Regime periods ─────────────────────────────────────────────────────────────
REGIME_PERIODS = {
    'Pre-Hike (2018–2021)': ('2018-01-01', '2022-02-28'),
    'Hike Cycle (2022)':    ('2022-03-01', '2022-12-31'),
    'Post-Hike (2023–2024)':('2023-01-01', '2024-12-31'),
}

# ── Macro event markers ────────────────────────────────────────────────────────
MACRO_EVENTS = {
    '2020-03-09': 'COVID crash',
    '2022-03-16': 'Fed first hike',
    '2022-09-22': 'BoJ intervention',
    '2022-10-21': 'BoJ 2nd intervene',
    '2022-12-20': 'BoJ YCC adjust',
    '2023-07-28': 'BoJ YCC ±1%',
    '2024-07-11': 'BoJ rate hike',
}

# ── Color palette ──────────────────────────────────────────────────────────────
ACCENT = '#58a6ff'
WARN   = '#f78166'
OK     = '#3fb950'
YELLOW = '#d29922'
PURPLE = '#bc8cff'
GRAY   = '#8b949e'
TEAL   = '#39d353'

DRIVER_COLORS = {
    'ΔDXY':   ACCENT,
    'ΔVIX':   WARN,
    'ΔUS10Y': YELLOW,
    'ΔOIL':   PURPLE,
}

# ── Matplotlib dark theme ──────────────────────────────────────────────────────
def apply_style():
    """Apply global dark chart style. Call once at startup."""
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
