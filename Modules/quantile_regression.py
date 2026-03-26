"""
m3_quantile.py — Module 3: Quantile Regression — Tail Behavior
==============================================================
Fits quantile regression at q = 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95
to reveal how macro drivers amplify (or dampen) at tail conditions.

Key question: Does OLS underestimate tail sensitivity?

Method: Subgradient descent minimizing asymmetric pinball loss.
  min Σ p_q(r_t - x_t'β_q)   where   p_q(u) = u(q - 1{u<0})

Outputs:
  outputs/03_quantile_regression.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    DRIVERS, TARGET,
    DRIVER_COLORS, ACCENT, WARN, GRAY,
)

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# ══════════════════════════════════════════════════════════════════════════════
# QUANTILE FITTING (subgradient descent)
# ══════════════════════════════════════════════════════════════════════════════

def _quantile_loss(beta,x,y,q):
    u = y - np.dot(x,beta)
    return np.sum(np.where(u >=0, q*u, (q-1)*u))    

def _fit_quantile(x,y,q, n_iter: int = 2000, learning_rate: float = 0.001):
    """sub-gradient for quantile regression"""
    beta      = np.zeros(X.shape[1])
    best_beta = beta.copy()
    best_loss = np.inf

    for i in range(n_iter):
        residual   = y - x @ beta
        grad    = np.where(residual >= 0, -q, -(q - 1))
        subgrad = x.T @ grad / len(y)
        step    = learning_rate / (1 + i * 0.001)
        beta    = beta - step * subgrad
        loss    = _quantile_loss(beta, x, y, q)
        if loss < best_loss:
            best_loss = loss
            best_beta = beta.copy()

    return best_beta

    
def quantile_regression(df): 
    """
    Fit quantile regression across QUANTILES.

    Returns:
        qr_results: {q: np.array of standardized coefficients per driver}
        beta_ols:   np.array — OLS baseline for comparison
        quantiles:  list of quantiles used
    """
    
    r = df[TARGET].values
    x_raw = df[DRIVERS].values
    n,k = x_raw.shape
    
    x_std_arr = x_raw.std(axis = 0)
    x_norm = (x_raw - x_raw.mean(axis = 0)) / x_std_arr
    x = np.column_stack([np.ones(n), x_norm])
    
    qr_result = {}
    for q in QUANTILES:
        beta_q = _fit_quantile(x,r,q)
        qr_result[q] = beta_q[1:]
    
    beta_ols = np.linalg.lstsq(x,r, rcond=None)[0][1:]
    
    # ── Tail amplification summary ─────────────────────────────────────────
    print(f"\n  Tail Amplification (q=0.90 vs q=0.10 vs OLS):")
    print(f"  {'Driver':<10} {'q=0.10':>10} {'q=0.50':>10} {'q=0.90':>10} {'OLS':>10} {'Amplified?'}")
    print("  " + "-"*60)
    for i, driver in enumerate(DRIVERS):
        q_10 = qr_result[0.01][i]
        q_50 = qr_result[0.50][i]
        q_90 = qr_result[0.90][i]
        ols = beta_ols[i]
        amplified = (
            'TAIL'   if abs(q_90) > abs(ols) * 1.3 or abs(q_10) > abs(ols) * 1.3
            else '— stable'
        )
        print(f"  {driver:<10} {q_10:>10.4f} {q_50:>10.4f} {q_90:>10.4f} {ols:>10.4f}  {amplified}")

    return  {
        'qr_results': qr_result,
        'beta_ols':   beta_ols,
        'quantiles':  QUANTILES,
    } 

def plot(qr_result, beta_ols):
    fig, axes = plt.subplot(2,2, figsize = (14,9))
    
    for idx, driver in enumerate(DRIVERS):
        ax = axes[idx // 2, idx % 2]
        q_values = QUANTILES
        coef_q = [qr_result[q][idx] for q in q_values]
        ols_val = beta_ols[idx]
        color = DRIVER_COLORS[driver]
        max_coef = max(abs(c) for c in coef_q)
        
        ax.plot(q_values, coef_q, color = color, lw = 2, marker = "o", marksize = 5, label = "Quantile coff")
        ax.axhline(ols_val, color = WARN, lw = 1.5, ls = "--", label = f"OLS = {ols_val:.3f}")
        ax.fill_between(q_values, ols_val, coef_q, alpha= 0.15, color= color)
        
        ax.axvspan(0,0.20, alpha = 0.05, color = ACCENT, label = "lower tail")
        ax.axvspan(0.8,1, alpha = 0.05, color = WARN, label = "upper tail")
        
        title_suffix = (
            'Tail amplification' if max_coef > abs(ols_val) * 1.2
            else 'Stable across quantiles'
        )
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Coefficient')
        ax.set_title(f'{driver}: {title_suffix}')
        ax.legend(fontsize=7)
        ax.set_xticks(q_values)
        ax.set_xticklabels([f'{q:.2f}' for q in q_values], fontsize=7)

    plt.tight_layout()

        