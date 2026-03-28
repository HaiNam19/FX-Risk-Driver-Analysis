import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from config import (
    DRIVERS, TARGET, MACRO_EVENTS,
    DRIVER_COLORS, ACCENT, WARN, GRAY,
)

"""
m1_correlation.py — Module 1: Driver Identification & Correlation Analysis
==========================================================================
Identifies which macro drivers have meaningful, stable relationships
with USD/JPY returns via static and rolling correlation analysis.
"""

def _plot_1(df, corr_result, roll_cor):
    fig, axes = plt.subplot(3,1, figsize = (14,11))
    fig.suptitle(
        'Module 1 — Driver Identification: USD/JPY Correlation Analysis',
        fontsize=13, color='#f0f6fc',
    )

    #Panel1: USD/JPY price with events marco
    ax = axes[0]
    ax.plot(df.index, df['usdjpy'], color = ACCENT, lw = 0.9, label = "USDJPY")
    ax.fill_between(df.index, df['usdjpy'].min(),df['usdjpy'], color = ACCENT, alpha = 0.2)
    ax.set_ylabel("Price")
    ax.set_title("USD/JPY price")
    for event_date, event_name in MACRO_EVENTS.items():
        date = pd.Timestamp(event_date)
        ax.axvline(date, color = WARN, alpha = 0.7, lw = 1, ls = "--")
        ax.test(date, df['usdjpy'].max() * 0.97,event_name, rotation = 90, fontsize = 6, color = WARN, ha = "right", alpha = 0.9)
        
    #Panel 2: statistics correlation bar
    ax = axes[1]
    driver_sorted = sorted(DRIVERS, key = lambda x: abs(corr_result[x]["pearson"]), reverse= True)
    pearson_values = [corr_result[x]["pearson"] for x in driver_sorted]
    x_position = np.arange(len(driver_sorted))
    bars = ax.bar(x_position, pearson_values, color = [DRIVER_COLORS[x] for x in driver_sorted])
    for bar, val in zip(bars, pearson_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01 * np.sign(val),
            f'{val:.3f}', ha='center',
            va='bottom' if val > 0 else 'top',
            fontsize=9, color='white', fontweight='bold',
        )
    ax.axhline(0, color=GRAY, linewidth=0.5)
    ax.set_xticks(x_position)
    ax.set_xticklabels(driver_sorted, fontsize=10)
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Static Correlation with USD/JPY Return — Sign matches theory?')

    
    #Panel 3: rolling correlation 
    ax = axes[2]
    for driver in DRIVERS:
        ax.plot(driver.index, roll_cor[driver], lw = 0.9, alpha = 0.85, label = driver, color = DRIVER_COLORS)
        ax.axhline(0, color = GRAY, lw = 0.8, ls = "--")
        ax.fill_between(driver.index, -0.1,0.1, color = GRAY, ALPHA = 0.1)
        ax.set_ylabel("Correlation windows: 60 days")
        ax.set_title("Rolling Correlation  ")
        ax.legent()
        for event_date, event_name in MACRO_EVENTS.items():
            ax.axvline(pd.Timestamp(event_date), color = "white", lw = 0.8, alpha = 0.3)
    plt.tight_layout()
    


def correlation_pair(df: pd.DataFrame) -> tuple[dict,dict]:
    """
    Compute Pearson, Spearman, Fisher Z-test, and rolling 60-day correlations.

    Returns:
        corr_results: {driver: {pearson, spearman, p}}
        roll_corr:    {driver: pd.Series of rolling correlation}
    """
    print("\n" + "="*60)
    print("MODULE 1 — DRIVER IDENTIFICATION & CORRELATION")
    print("="*60)
 
    r = df[TARGET]
    corr_result = {}
    for driver in DRIVERS:
        x = df[driver]
        pearson_r, pearson_p = stats.pearsonr(r,x)
        spearman_r, spearman_p = stats.spearmanr(r,x)
    
    # Fisher Z-test: H₀ ρ = 0
    z = 0.5 * np.log((1+pearson_r)/(1-pearson_r + 1e-10 ))
    se = 1/np.sqrt(len(r) - 3)
    fisher_p = 2* (1- stats.norm.cdf(abs(z) / se))
    significant = "statistics meaning" if fisher_p< 0.05 else "X"
    print(f"  {driver:<10} {pearson_r:>10.4f} {spearman_r:>10.4f} {fisher_p:>10.4f}  {significant}")
    corr_result[driver] = {"pearson": pearson_r, "spearman": spearman_r, "p_value": fisher_p}
    
    
    #--Rolling 60 days------------
    roll_cor = {
        r.rolling(60).corr(df[driver]) for driver in DRIVERS
    }

    _plot_1(df, corr_result, roll_cor)
    
    return corr_result, roll_cor
