"""
m4_regime.py — Module 4: Regime Analysis — Coefficient Stability
================================================================
Tests whether macro driver relationships with USD/JPY shifted
across three BoJ policy regimes using per-regime OLS, Chow test,
and rolling 120-day coefficient estimation.

Regimes:
  Pre-Hike  (2018-2021): BoJ YCC stable, low divergence
  Hike Cycle (2022):     Fed rate shock, extreme divergence
  Post-Hike (2023-2024): BoJ normalization

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from config import (
    DRIVERS, TARGET, REGIME_PERIODS, MACRO_EVENTS,
    DRIVER_COLORS, ACCENT, WARN, GRAY,
)

ROLLING_WINDOW = 120


# ══════════════════════════════════════════════════════════════════════════════
# reserach OLS beta for ech periods 
# ══════════════════════════════════════════════════════════════════════════════
# OLS on a specific period, return (beta, r2, RSS, n) 
def _ols_period(period_df):
    r = period_df[TARGET].values
    x_raw = period_df[DRIVERS].values
    x_norm = (x_raw - x_raw.mean(axis = 0)) / x_raw.std(axis = 0)
    x = np.column_stack([np.ones(len(r)), x_norm])
    
    [n,k] = x.shape
    beta = np.linalg.lstsq(x,r, rcond= None)[0] #chỉ lấy ra phần từ beta
    r_prediction = np.dot(x,beta)
    rss = np.sum((r- r_prediction)**2) #sum square residuals
    tss = np.sum((r - r.mean())**2) #sum square total
    r2 = 1 - rss/tss
    r2_adj = 1 - (rss/(n-k-1))/(tss/(n-1))
    return beta[1:], r2, r2_adj, rss, len(r)


def _plot_4(roll_date, roll_beta, roll_r2):
    fig , axes = plt.subplot(3,1, figsize = (14,11), sharex = None)
    fig.suptitle(
        ' — Regime Analysis: How Driver Relationships Shift',
        fontsize=13, color='#f0f6fc',
    )
    
    #rolling R^2
    ax = axes[0]
    ax.plot(roll_date, roll_r2, color = ACCENT, lw = 0.9)
    ax.fill_between(roll_date, 0, roll_r2, color = ACCENT, alpha = 0.2 )
    ax.set_ylabel('Rolling R^2')
    ax.set_title(f'Rolling {ROLLING_WINDOW}-day R^2 — Model ')
    
    #rolling DXY & VIX coefficients
    ax = axes[1]
    for driver in ['ΔDXY', 'ΔVIX']:
        ax.plot(roll_date, roll_beta[driver], lw = 0.9, label = f"{driver}", color = DRIVER_COLORS)
    ax.axhline(0, color = GRAY, lw = 0.5, ls = "--")
    ax.set_ylabel("Coefficent")
    ax.set_title('Rolling Coefficients: DXY & VIX — Magnitude shifts across regimes')
    ax.legend(fontsize=9)

    #rolling US10Y & Oil coefficients
    ax = axes[2]
    for driver in ['ΔUS10Y', 'ΔOIL']:
        ax.plot(roll_date, roll_beta[driver], color = DRIVER_COLORS, lw = 0.9, label = f"{driver}")
    ax.axhline(0, color = GRAY, lw = 0.5, ls = "--")
    ax.set_ylabel("Coefficent")
    ax.set_title('Rolling Coefficients: US10Y & Oil — Magnitude shifts across regimes')
    ax.legend(fontsize=9)
    
    # Event markers on all panels
    for ax in axes:
        for event_date in MACRO_EVENTS:
            ax.axvline(pd.Timestamp(event_date), color = WARN, lw = 1, alpha = 0.5, ls = "--")
    
    plt.tight_layout()
    
    

def regime_analysis(df, ols_result: dict):
    """
    Per-regime OLS, Chow test, and rolling coefficient analysis.

    Returns:
        regime_results: {period_name: {beta, r2, rss, n}}
        roll_betas:     {driver: list of rolling coefficients}
        roll_r2:        list of rolling R²
        roll_dates:     DatetimeIndex
    """
    
    # regime OLS
    regime_result = {}
    for name, (start_date, end_date) in REGIME_PERIODS.items():
        period = df[start_date:end_date]
        if len(period) <50: 
            continue
        beta_period, r2_period, r2_adj_period, rss_period, n_period = _ols_period(period)
        regime_result[name] = {"beta": beta_period, "r2": r2_period, "r2 adjusted": r2_adj_period, "RSS": rss_period, "days": n_period}
    
    # Rolling OLS
    roll_beta = {d: [] for d in DRIVERS}
    roll_date = []
    roll_r2 = []
    for i in range(ROLLING_WINDOW, len(df)):
        period = df.iloc[i-ROLLING_WINDOW, i]
        beta, r2, r2_adj, rss, n = _ols_period(period)
        roll_r2.append(r2)
        roll_date.append(df.index[i])
        for index, driver in enumerate(DRIVERS):
            roll_beta[driver].append(beta[index]) 
    roll_date = pd.DatetimeIndex(roll_date)
    
    _plot_4(roll_date, roll_beta, roll_r2)
    
    return {
        'regime_results': regime_result,
        'roll_betas':     roll_beta,
        'roll_r2':        roll_r2,
        'roll_dates':     roll_date,
    }      
    


        
