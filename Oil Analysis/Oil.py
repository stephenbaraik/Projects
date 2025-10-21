#!/usr/bin/env python3
"""
File: analysis_brent_for_paper.py
Purpose: Complete pipeline using BRENT only (2014-01-01 -> 2024-12-31) with rich tables & visuals,
         including a 10-year Monte Carlo scenario analysis with full simulated paths for fan charts.
Usage: edit CONFIG and run: python analysis_brent_for_paper.py
Outputs: CSVs, PNGs and model summaries in outputs_brent_for_paper/
Dependencies: pandas, numpy, yfinance, matplotlib, seaborn, statsmodels, arch, scikit-learn, scipy
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from arch import arch_model
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150})

# ---------------- CONFIG ----------------
CONFIG = {
    "brent_ticker": "BZ=F",     # yfinance Brent
    "stocks": {                 # yfinance tickers -> short names
        "IOC.NS": "IOC",
        "ONGC.NS": "ONGC",
        "BPCL.NS": "BPCL",
        "HPCL.NS": "HPCL",
        "RELIANCE.NS": "RELIANCE"
    },
    "start_date": "2014-01-01",
    "end_date": "2024-12-31",
    "agg": "M",                 # month-end aggregation
    "output_dir": "outputs_brent_for_paper",
    "winsor_z": 4.0,
    "max_granger_lags": 6,
    "n_sims": 4000,             # Monte Carlo sims
    "horizon_months": 120,      # 10 years * 12
    "random_seed": 2025,
    "store_full_paths": True,   # store all sim paths (needed for monthly fan charts)
    "plot_top_n": 5,            # number of stocks to include in main plots
    "event_dates": [            # example event dates; edit as needed
        # "2020-03-08", "2014-09-01"
    ],
}
# ----------------------------------------

OUT = Path(CONFIG["output_dir"])
OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(CONFIG["random_seed"])

# ---------------- Helpers ----------------
def savefig(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")

def winsorize(s, z=4.0):
    arr = s.dropna().values
    zscores = np.abs(stats.zscore(arr))
    if len(zscores)==0:
        return s
    mask = zscores > z
    if mask.any():
        nonmask = arr[~mask]
        low, high = nonmask.min(), nonmask.max()
        arr2 = arr.copy()
        arr2[mask] = np.where(arr2[mask] > np.median(arr2), high, low)
        return pd.Series(arr2, index=s.dropna().index).reindex(s.index)
    return s

def adf_res(series, reg="c"):
    try:
        r = adfuller(series.dropna(), regression=reg, autolag="AIC")
        return {"stat": r[0], "pvalue": r[1], "nlags": r[2]}
    except Exception:
        return {"stat": np.nan, "pvalue": np.nan, "nlags": np.nan}

def kpss_res(series, reg="c"):
    try:
        r = kpss(series.dropna(), regression=reg, nlags="auto")
        return {"stat": r[0], "pvalue": r[1], "nlags": r[2]}
    except Exception:
        return {"stat": np.nan, "pvalue": np.nan, "nlags": np.nan}

# ---------------- Data download & monthly aggregation ----------------
print("Downloading data...")
tickers = [CONFIG["brent_ticker"]] + list(CONFIG["stocks"].keys())
raw = yf.download(tickers, start=CONFIG["start_date"], end=CONFIG["end_date"], progress=False)
if "Close" in raw.columns and isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"].copy()
else:
    close = raw[["Close"]].copy()
# rename stocks to short names
rename_map = {k: v for k, v in CONFIG["stocks"].items() if k in close.columns}
close = close.rename(columns=rename_map)
if CONFIG["brent_ticker"] not in close.columns:
    raise RuntimeError("Brent ticker missing from downloaded data.")
cols = [CONFIG["brent_ticker"]] + [c for c in close.columns if c != CONFIG["brent_ticker"]]
close = close[cols].dropna(how="all").sort_index()
prices_m = close.resample(CONFIG["agg"]).last().dropna(how="all")
logp_m = np.log(prices_m)
ret_m = logp_m.diff().dropna()

prices_m.to_csv(OUT / "monthly_prices.csv")
ret_m.to_csv(OUT / "monthly_returns.csv")
print("Saved monthly prices & returns.")

# Winsorize returns (optional)
for c in ret_m.columns:
    ret_m[c] = winsorize(ret_m[c], z=CONFIG["winsor_z"])

# ---------------- EDA: tables + figures ----------------
print("Generating EDA tables & visuals...")
# Price trends (log-scale)
fig, ax = plt.subplots(figsize=(10,5))
(logp_m[[CONFIG["brent_ticker"]] + [c for c in prices_m.columns if c!=CONFIG["brent_ticker"]]]
 .plot(ax=ax, lw=1))
ax.set_title("Monthly log-prices: Brent vs Indian oil stocks")
ax.set_ylabel("Log Price")
savefig(fig, "prices_log_trend.png")

# Returns histogram + summary table
ret_stats = ret_m.describe().T
ret_stats["skew"] = ret_m.skew()
ret_stats["kurt"] = ret_m.kurtosis()
ret_stats.to_csv(OUT / "returns_summary_stats.csv")

fig, ax = plt.subplots(figsize=(10,6))
ret_m.hist(bins=40, layout=(2,3), ax=ax)
plt.tight_layout()
savefig(fig, "returns_histograms.png")

# Correlation matrix & rolling correlations
corr_mat = ret_m.corr()
corr_mat.to_csv(OUT / "returns_correlation_matrix.csv")
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr_mat, annot=True, fmt=".2f", ax=ax)
ax.set_title("Static returns correlation matrix")
savefig(fig, "corr_matrix.png")

rolling_window = 12
rolling_corrs = {c: ret_m[CONFIG["brent_ticker"]].rolling(rolling_window).corr(ret_m[c]) for c in ret_m.columns if c!=CONFIG["brent_ticker"]}
rolling_corr_df = pd.DataFrame(rolling_corrs)
rolling_corr_df.to_csv(OUT / "rolling_correlations_12m.csv")
fig, ax = plt.subplots(figsize=(10,5))
rolling_corr_df.plot(ax=ax)
ax.set_title("12-month rolling correlation with Brent")
savefig(fig, "rolling_corr_with_brent.png")

# Volatility clustering: rolling vol
fig, ax = plt.subplots(figsize=(10,5))
(ret_m.rolling(12).std()*np.sqrt(12)).plot(ax=ax)  # annualized approx
ax.set_title("12-month rolling volatility (annualized approx)")
savefig(fig, "rolling_volatility_12m.png")

# Cross-correlation function (Brent vs each stock)
from statsmodels.tsa.stattools import ccf
lags = 12
for s in [c for c in ret_m.columns if c!=CONFIG["brent_ticker"]]:
    x = ret_m[CONFIG["brent_ticker"]].dropna()
    y = ret_m[s].dropna()
    minlen = min(len(x), len(y))
    x,y = x[-minlen:], y[-minlen:]
    c = [x.corr(y.shift(l)) for l in range(-lags, lags+1)]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(range(-lags, lags+1), c)
    ax.set_title(f"Cross-correlation (Brent vs {s})")
    ax.set_xlabel("lag (months)")
    savefig(fig, f"ccf_brent_{s}.png")

# ---------------- Stationarity tests ----------------
print("Stationarity tests (ADF + KPSS) on levels and returns...")
st_results = {}
for col in prices_m.columns:
    lvl = logp_m[col].dropna()
    ret = ret_m[col].dropna()
    st_results[col] = {
        "adf_level": adf_res(lvl, reg="ct"),
        "kpss_level": kpss_res(lvl, reg="ct"),
        "adf_returns": adf_res(ret, reg="c"),
        "kpss_returns": kpss_res(ret, reg="c"),
    }
pd.json_normalize(st_results).T.to_csv(OUT / "stationarity_tests.json")
print("Saved stationarity test results.")

# ---------------- Cointegration (Johansen) ----------------
print("Performing Johansen cointegration test on log-levels...")
levels = logp_m[[CONFIG["brent_ticker"]] + [c for c in prices_m.columns if c!=CONFIG["brent_ticker"]]].dropna()
if len(levels) < 20:
    print("Warning: few observations for Johansen (<20).")
jres = coint_johansen(levels, det_order=0, k_ar_diff=1+2)
# save eigenvalues and trace stats
j_summary = {
    "eigenvalues": jres.eig.tolist(),
    "trace_stat": jres.lr1.tolist(),
    "crit_90": jres.cvt[:,0].tolist(),
    "crit_95": jres.cvt[:,1].tolist(),
    "crit_99": jres.cvt[:,2].tolist()
}
pd.DataFrame(j_summary).to_csv(OUT / "johansen_summary.csv")
# determine rank at 5% critical
rank = sum([1 for stat, cv in zip(jres.lr1, jres.cvt[:,1]) if stat > cv])
print(f"Estimated cointegration rank (trace, 5%): {rank}")

# ---------------- VECM / VAR and IRFs ----------------
print("Estimating VECM or VAR & computing IRFs...")
models_info = {}
ret_dropna = ret_m.dropna()
if rank >= 1:
    vecm = VECM(levels, k_ar_diff=1+2, coint_rank=rank, deterministic="co")
    vecm_res = vecm.fit()
    txt = vecm_res.summary().as_text()
    (OUT / "vecm_summary.txt").write_text(txt)
    models_info["model"] = "VECM"
    models_info["summary_file"] = "vecm_summary.txt"
    # IRF using statsmodels VECM->irf (may not exist depending on version); fallback: convert to VAR representation
    try:
        irf = vecm_res.irf(24)  # 24 months horizon
        np.save(OUT / "vecm_irfs.npy", irf.irfs)
    except Exception:
        (OUT / "vecm_irf_unavailable.txt").write_text("IRF method unavailable for VECM with this statsmodels version.")
else:
    # VAR on returns
    sel = VAR(ret_dropna).select_order(maxlags=CONFIG["max_granger_lags"])
    chosen = sel.aic if sel.aic is not None else sel.bic if sel.bic is not None else 1
    var = VAR(ret_dropna)
    var_res = var.fit(maxlags=chosen)
    (OUT / "var_summary.txt").write_text(var_res.summary().as_text())
    models_info["model"] = "VAR"
    models_info["lag_order"] = var_res.k_ar
    # IRFs
    irf = var_res.irf(24)
    # Save IRF plots for responses of stocks to Brent shock
    for resp_col in [c for c in ret_dropna.columns if c!=CONFIG["brent_ticker"]]:
        fig = irf.plot(orth=False, impulse=CONFIG["brent_ticker"], response=resp_col)
        # statsmodels returns a matplotlib Figure list sometimes; handle both
        try:
            savefig(fig[0], f"irf_{resp_col}_to_brent.png")
        except Exception:
            # if fig is a single Figure:
            savefig(fig, f"irf_{resp_col}_to_brent.png")

print("Saved VAR/VECM outputs.")

# ---------------- Granger causality ----------------
print("Running pairwise Granger causality tests...")
gc_table = []
for a in ret_dropna.columns:
    for b in ret_dropna.columns:
        if a==b:
            continue
        try:
            res = grangercausalitytests(ret_dropna[[b,a]], maxlag=CONFIG["max_granger_lags"], verbose=False)
            # capture minimal p-value across lags (ssr_ftest)
            pvals = [res[l][0]["ssr_ftest"][1] for l in res]
            gc_table.append({"cause": a, "effect": b, "min_pvalue": min(pvals)})
        except Exception:
            gc_table.append({"cause": a, "effect": b, "min_pvalue": np.nan})
gc_df = pd.DataFrame(gc_table).pivot(index="cause", columns="effect", values="min_pvalue")
gc_df.to_csv(OUT / "granger_min_pvalues.csv")
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(gc_df.astype(float), annot=True, fmt=".3f", ax=ax, cmap="viridis", cbar_kws={"label":"min p-value"})
ax.set_title("Pairwise Granger causality (min p-value across lags)")
savefig(fig, "granger_min_pvalues_heatmap.png")

# ---------------- Volatility modeling (GARCH) ----------------
print("Fitting GARCH(1,1) to each returns series...")
garch_summary = {}
std_resids = {}
for c in ret_m.columns:
    series = ret_m[c].dropna()*100  # scale
    try:
        am = arch_model(series, vol="Garch", p=1, q=1, dist="t")
        res = am.fit(disp="off")
        garch_summary[c] = {"aic": res.aic, "bic": res.bic, **res.params.to_dict()}
        cond_vol = res.conditional_volatility
        cond_vol.to_csv(OUT / f"condvol_{c}.csv")
        std_resids[c] = pd.Series(res.std_resid, index=series.index)
    except Exception as e:
        garch_summary[c] = {"error": str(e)}
        std_resids[c] = (series - series.mean())/series.std()
pd.DataFrame(garch_summary).T.to_csv(OUT / "garch_summary.csv")

# Volatility spillover proxy: corr of squared standardized residuals
sqsr = {k: (v**2).dropna() for k,v in std_resids.items()}
vol_spill = pd.DataFrame(index=sqsr.keys(), columns=sqsr.keys(), dtype=float)
for a in sqsr:
    for b in sqsr:
        common = pd.concat([sqsr[a], sqsr[b]], axis=1).dropna()
        vol_spill.loc[a,b] = common.iloc[:,0].corr(common.iloc[:,1])
vol_spill.to_csv(OUT / "volatility_spillover_matrix.csv")
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(vol_spill.astype(float), annot=True, fmt=".2f", ax=ax)
ax.set_title("Volatility spillover proxy (corr of squared standardized residuals)")
savefig(fig, "volatility_spillover_heatmap.png")

# ---------------- PCA / Factor modelling ----------------
print("Running PCA on returns...")
ret_clean = ret_m.dropna()
pca = PCA(n_components=min(6, ret_clean.shape[1]))
pca.fit(ret_clean)
explained = pd.Series(pca.explained_variance_ratio_, index=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))])
explained.to_csv(OUT / "pca_explained_variance_ratio.csv")
# Regress each stock on PC1
pc1 = pca.transform(ret_clean)[:,0]
factor_reg = {}
for i,col in enumerate(ret_clean.columns):
    Y = ret_clean[col].values
    X = add_constant(pc1)
    mdl = OLS(Y,X).fit()
    factor_reg[col] = {"alpha": float(mdl.params[0]), "beta_pc1": float(mdl.params[1]), "r2": float(mdl.rsquared)}
pd.DataFrame(factor_reg).T.to_csv(OUT / "pca_factor_regression.csv")

# ---------------- Event study scaffold ----------------
print("Preparing event study outputs...")
events = [pd.to_datetime(d) for d in CONFIG.get("event_dates",[]) if d]
event_window = 6  # months before/after
event_results = {}
for ev in events:
    window = (ret_m.index >= ev - pd.DateOffset(months=event_window)) & (ret_m.index <= ev + pd.DateOffset(months=event_window))
    subset = ret_m.loc[window]
    # abnormal returns = stock_ret - beta*brent_ret (using previously estimated betas)
    ar = {}
    for s in [c for c in ret_m.columns if c!=CONFIG["brent_ticker"]]:
        # simple abnormal using linear regression on history
        ar[s] = (subset[s] - (relations := OLS(ret_m[s].dropna(), add_constant(ret_m[CONFIG["brent_ticker"]].dropna())).fit()).params[0] - relations.params[1]*subset[CONFIG["brent_ticker"]]).mean()
    event_results[str(ev.date())] = ar
if event_results:
    pd.DataFrame(event_results).to_csv(OUT / "event_study_summary.csv")
    print("Event study summary saved (edit CONFIG['event_dates'] to run).")
else:
    (OUT / "event_study_summary.csv").write_text("No events provided in CONFIG.")

# ---------------- Scenario analysis (Monte-Carlo with full paths) ----------------
print("Running scenario Monte Carlo with full simulated paths (monthly)...")
h = CONFIG["horizon_months"]
n = CONFIG["n_sims"]
last_price = prices_m.iloc[-1]
# Brent historical monthly stats
brent_ret = ret_m[CONFIG["brent_ticker"]].dropna()
mu_b = brent_ret.mean()
sigma_b = brent_ret.std(ddof=1)

def monthly_additive_from_cumulative(mult, months):
    return np.log(mult)/months

scenarios = {
    "baseline": {"mu": mu_b, "sigma": sigma_b, "note": "Historical mean/std"},
    "optimistic_+50pc": {"mu": mu_b + monthly_additive_from_cumulative(1.5, h), "sigma": sigma_b, "note": "+50% over 10y"},
    "pessimistic_-30pc": {"mu": mu_b + monthly_additive_from_cumulative(0.7, h), "sigma": sigma_b, "note": "-30% over 10y"},
    "high_vol": {"mu": mu_b, "sigma": sigma_b * 2.0, "note": "2x vol"},
}

# Estimate stock ~ Brent monthly linear relations (alpha, beta, sigma_eps)
relations = {}
for s in [c for c in ret_m.columns if c!=CONFIG["brent_ticker"]]:
    df = pd.concat([ret_m[s], brent_ret], axis=1).dropna()
    mdl = OLS(df[s], add_constant(df[CONFIG["brent_ticker"]])).fit()
    relations[s] = {"alpha": float(mdl.params[0]), "beta": float(mdl.params[1]), "sigma_eps": float(mdl.resid.std(ddof=1)), "r2": float(mdl.rsquared)}
pd.DataFrame(relations).T.to_csv(OUT / "stock_brent_monthly_regression.csv")

# Storage
sim_month_index = pd.date_range(start=prices_m.index[-1] + pd.DateOffset(months=1), periods=h, freq="M")
scenario_outputs = {}
for scen_name, scen in scenarios.items():
    mu_s, sigma_s = scen["mu"], scen["sigma"]
    # arrays: (n_sims, h+1)
    brent_paths = np.zeros((n, h+1))
    stock_paths = {s: np.zeros((n, h+1)) for s in relations}
    brent_paths[:,0] = last_price[CONFIG["brent_ticker"]]
    for s in relations:
        stock_paths[s][:,0] = last_price[s]
    # simulate
    for sim in range(n):
        for t in range(1, h+1):
            z = np.random.normal()
            r_b = mu_s + sigma_s * z
            brent_paths[sim, t] = brent_paths[sim, t-1] * np.exp(r_b)
            # stocks
            for s, rel in relations.items():
                eps = np.random.normal(scale=rel["sigma_eps"])
                r_s = rel["alpha"] + rel["beta"] * r_b + eps
                stock_paths[s][sim, t] = stock_paths[s][sim, t-1] * np.exp(r_s)
    # compute monthly percentiles for Brent and stocks
    percentiles = {}
    for instr, arr in [("Brent", brent_paths)] + [(s, stock_paths[s]) for s in stock_paths]:
        # arr shape (n, h+1)
        df_perc = {}
        for p in [5,25,50,75,95]:
            df_perc[f"p{p}"] = np.percentile(arr, p, axis=0)
        perc_df = pd.DataFrame(df_perc, index=[last_price[instr if instr!="Brent" else CONFIG["brent_ticker"]]] + list(sim_month_index))
        percentiles[instr] = perc_df
        # save end-of-horizon distribution table
        end_vals = arr[:, -1]
        ev_df = pd.Series({
            "median": np.median(end_vals),
            "p5": np.percentile(end_vals,5),
            "p25": np.percentile(end_vals,25),
            "p75": np.percentile(end_vals,75),
            "p95": np.percentile(end_vals,95),
            "mean": np.mean(end_vals),
            "std": np.std(end_vals)
        })
        ev_df.to_frame(name=scen_name).to_csv(OUT / f"end_distribution_{instr}_{scen_name}.csv")
    # save full percentiles (CSV)
    for instr, perc_df in percentiles.items():
        fname = f"{instr}_monthly_percentiles_{scen_name}.csv"
        perc_df.to_csv(OUT / fname)
    # save sample of simulated paths (for paper figure reproducibility)
    # write compressed numpy arrays
    np.savez_compressed(OUT / f"sim_brent_{scen_name}.npz", arr=brent_paths)
    for s in stock_paths:
        np.savez_compressed(OUT / f"sim_{s}_{scen_name}.npz", arr=stock_paths[s])
    scenario_outputs[scen_name] = {"percentiles": percentiles, "brent_paths_file": f"sim_brent_{scen_name}.npz"}
    print(f"Scenario {scen_name}: saved percentiles and sim files.")

# Plot fan charts (Brent + up to top N stocks) for each scenario
print("Creating fan charts for scenarios...")
for scen_name, out in scenario_outputs.items():
    # Brent
    perc = out["percentiles"]["Brent"]
    fig, ax = plt.subplots(figsize=(9,5))
    x = perc.index
    ax.plot(x, perc["p50"], label="median")
    ax.fill_between(x, perc["p5"], perc["p95"], alpha=0.2, label="5-95")
    ax.fill_between(x, perc["p25"], perc["p75"], alpha=0.3, label="25-75")
    ax.set_title(f"Brent simulated fan chart: {scen_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    savefig(fig, f"fan_brent_{scen_name}.png")
    # stocks: choose up to plot_top_n
    stocks_to_plot = list(relations.keys())[:CONFIG["plot_top_n"]]
    for s in stocks_to_plot:
        perc_s = out["percentiles"][s]
        fig, ax = plt.subplots(figsize=(9,5))
        x = perc_s.index
        ax.plot(x, perc_s["p50"], label="median")
        ax.fill_between(x, perc_s["p5"], perc_s["p95"], alpha=0.2)
        ax.fill_between(x, perc_s["p25"], perc_s["p75"], alpha=0.3)
        ax.set_title(f"{s} simulated fan chart: {scen_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        savefig(fig, f"fan_{s}_{scen_name}.png")

print("Scenario fan charts saved. All scenario CSVs and sim arrays saved.")

# ---------------- Final summary table compilation ----------------
print("Compiling final summary tables for paper...")
summary_rows = []
for scen_name, out in scenario_outputs.items():
    for instr, perc_df in out["percentiles"].items():
        end = perc_df.iloc[-1]
        cagr_median = ((end["p50"]/last_price[instr if instr!="Brent" else CONFIG["brent_ticker"]]) ** (1.0/(CONFIG["horizon_months"]/12.0)) - 1.0)
        summary_rows.append({
            "scenario": scen_name,
            "instrument": instr,
            "p5_end": end["p5"],
            "p25_end": end["p25"],
            "p50_end": end["p50"],
            "p75_end": end["p75"],
            "p95_end": end["p95"],
            "median_annualized_cagr": cagr_median
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "scenario_end_summary_table.csv", index=False)
print("Saved scenario_end_summary_table.csv")

print(f"\nAll outputs (CSVs, PNGs, npz sim files) are in: {OUT.resolve()}")
print("Open the CSVs and PNGs and paste into your paper. If you want a notebook version with inline plots, I can convert this script to a notebook.")

# End of script