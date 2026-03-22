import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from config import (
    DRIVERS, TARGET, MACRO_EVENTS,
    DRIVER_COLORS, ACCENT, WARN, OK, PURPLE, GRAY,
)

"""
m2_ols.py — Module 2: OLS Regression — Baseline Model
======================================================
Quantifies macro driver → USD/JPY relationships via OLS.
Runs full residual diagnostics: JB normality, Durbin-Watson,
Breusch-Pagan heteroskedasticity, and VIF multicollinearity.

Outputs:
  outputs/02_ols_diagnostics.png
"""

def run(df: pd.DataFrame) -> dict:
    """
    Fit OLS:  r_USDJPY = α + β₁ΔDXY + β₂ΔVIX + β₃ΔUS10Y + β₄ΔOIL + ε

    Returns dict with:
        beta, beta_orig, std_beta, p_vals,
        r2, r2_adj, residuals, r_hat,
        X_mean, X_std, jb_p, dw, bp_p, vif
    """
    print("\n" + "="*60)
    print("MODULE 2 — OLS REGRESSION: BASELINE MODEL")
    print("="*60)

    r = df[TARGET].values
    x_raw = df[DRIVERS].values
    row, col = x_raw.shape
    
    # ----- Normalize -------
    x_mean = x_raw.mean(axis = 0)
    x_std = x_raw.std(axis = 0)
    x_std[x_std == 0] = 1
    x_normalize = (x_raw - x_mean) / x_std
    X = np.column_stack([np.ones(row), x_normalize])
    
    # ----- OLS Regression -------
    beta = np.linalg.lstsq(X,r) #hệ số beta
    r_prediction = np.dot(X,beta) # return dự đoán của model
    residual = r- r_prediction #sai số 
    
    ss_res = np.sum(residual**2) # sum square sai số bình phương
    ss_total = np.sum((r - r.mean())**2) #sum square của dữ liệu ro với giá trị tbinh dữ liệu
    r2 = 1 - (ss_res/ss_total) 
    r2_adjust = 1 - (1-r2)* ((row -1)/(row - col - 1))
    
    mse = ss_res/ (row - col - 1)  # mean square error
    var_beta = mse * np.linalg.inv( X.T @ X) # ma trận hiệp phương sai của hệ số beta
    se_beta = np.sqrt(np.diag(var_beta)) #standard error 
    t_statistic = beta / se_beta
    p_value    = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=row - col - 1))

    # Back to original scale for display
    beta_orig = beta[1:] / x_std
    std_beta = list(beta[1:])
    
    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n  OLS Results (n={n}, k={k}):")
    print(f"  R² = {r2:.4f}  |  Adjusted R² = {r2_adjust:.4f}")
    print(f"\n  {'Driver':<10} {'Coeff':>12} {'Std Beta':>10} {'t-stat':>10} {'p-value':>10} {'Sig'}")
    print("  " + "-"*60)
    print(f"  {'Intercept':<10} {beta[0]:>12.6f} {'—':>10} {t_statistic[0]:>10.3f} {p_value[0]:>10.4f}")
    
    for i, driver in enumerate(DRIVERS):
        if p_value[i+1] < 0.01:
            significant = "really meaning full ***"
        elif p_value[i+1] < 0.05:
            significant = "meaningful **"
        elif p_value[i+1] < 0.1:
            significant = "nearly meaningful"
        else: 
            significant = "not meaning"
        print(f"  {driver:<10} {beta_orig[i]:>12.6f} {std_beta[i]:>10.4f} "
              f"{t_statistic[i+1]:>10.3f} {p_value[i+1]:>10.4f}  {significant}")
    
    # diagnostics
    
    jb_stat, jb_p = stats.jarque_bera(residual**2) #Jarque-Bera
    
    diff_squared_sum = sum((residual[i+1] - residual[i])**2 for i in range(len(residual)))
    dw = diff_squared_sum / np.sum(residual**2) #Durbin- Watson
    
    # Breusch-Pagan
    res2     = residual**2
    bp_beta  = np.linalg.lstsq(X, res2, rcond=None)[0]
    bp_r2    = 1 - np.sum((res2 - X @ bp_beta)**2) / np.sum((res2 - res2.mean())**2)
    bp_stat  = row * bp_r2
    bp_p     = 1 - stats.chi2.cdf(bp_stat, col)
    
    # VIF 
    vif = {}
    for i, driver in enumerate(DRIVERS):
        X_other = np.column_stack(
            [np.ones(row)] + [x_normalize[:, j] for j in range(col) if j != i] # ghép 2 list lại với nhau (+)
        )
        beta_driver       = np.linalg.lstsq(X_other, x_normalize[:, i], rcond=None)[0]
        r2_vif  = 1 - (np.sum((x_normalize[:, i] - X_other @ beta_driver)**2) /
                        np.sum((x_normalize[:, i] - x_normalize[:, i].mean())**2))
        vif[driver] = 1 / (1 - r2_vif + 1e-10)

    print(f"\n  Residual Diagnostics:")
    print(f"  Jarque-Bera:   stat={jb_stat:.2f}  p={jb_p:.4f}  "
          f"{'REJECT normality' if jb_p < 0.05 else 'OK'}")
    print(f"  Durbin-Watson: {dw:.4f}  "
          f"{'OK (≈2)' if 1.5 < dw < 2.5 else 'Autocorrelation detected'}")
    print(f"  Breusch-Pagan: stat={bp_stat:.2f}  p={bp_p:.4f}  "
          f"{'Heteroskedastic' if bp_p < 0.05 else 'OK'}")
    print(f"  VIF: {', '.join(f'{d}={v:.2f}' for d, v in vif.items())}")

    #top unexpected days
    residual_series = pd.Series(residual, index = df.index)
    top_residual = residual_series.abs().nlargest(8)
    for date, value in top_residual:
        event = MACRO_EVENTS.get(str(date.date()), "This days wasn't in Marco Events")
        event_str = f" {event}" if event else "This days wasn't in Marco Events"
        print(f" {date.date()} residual= {value*100:+.3f}% {event_str} ")
    
    
    return {
        'beta':      beta,
        'beta_orig': beta_orig,
        'std_beta':  np.array(std_beta),
        'p_vals':    p_value[1:],
        'r2':        r2,    
        'r2_adj':    r2_adjust,
        'residuals': residual,
        'r_hat':     r_prediction,
        'X_mean':    x_mean,
        'X_std':     x_std,
        'jb_p':      jb_p,
        'dw':        dw,
        'bp_p':      bp_p,
        'vif':       vif,
    }
    

def plot(df, residual, r_prediction, residual_series, beta, std_beta, 
         p_value, r2, r2_adjusted, jb_p, dw ):
    
    fig, axes = plt.subplot(3,1, figsize = (12,8))
    fig.subtitle(f'Module 2 — OLS Regression: R²={r2:.3f}, Adj-R²={r2_adjusted:.3f}',
        fontsize=13, color='#f0f6fc',
    )
    
    #residual over time
    ax = axes[0]
    ax.plot(df.index, residual*100, color = GRAY, lw = 0.5, alpha = 0.7)
    large_residual = residual_series.abs() > residual.abs().quantile(0.99)
    ax.scatter(df.index[large_residual], residual[large_residual]*100, color = WARN, s = 20, zorder = 5, label = "Top 1% residual")
    ax.axhline(0, lw = 0.5, color = GRAY)
    ax.legend()
    for event_date, event in MACRO_EVENTS.items():
        ax.axvline(pd.Timestamp(event_date), color = PURPLE, lw = 1.2, alpha = 0.5)
    
    #residual distribution
    ax = axes[1]
    x_range = np.linspace(residual.min(),residual.max(), 200)
    ax.hist(residual*100, bins = 60, density = True, color = ACCENT, alpha = 0.5, label = "Residual")
    ax.plot(x_range*100, stats.norm.pdf(x_range, residual.mean(),residual.std()) / 100, color = OK, lw = 2, label = "Normal fit")
    ax.set_xlabel("Residual %")
    ax.set_title("Residual distribution | JB p-value = {jb_p:.4f}   DW = {dw:.3f}"   )
    ax.legend()
    
    plt.tight_layout()
    
