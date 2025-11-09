#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transfer_elastic_net import estimateTransferElasticNet
from pathlib import Path
from tqdm import tqdm

# saving settings
SAVE_PDF = True
OUTPUT_DIR = Path("./")

# data & experiment settings
N = 50
P = 100
S = 10
SIGMA_NOISE = 5
N_SIM = 2000
SEED_NUM = 1

# Correlation settings for error-bound experiments
CORR_SETTINGS = [
    ("equiv_corr", 0.0),
    ("ar1", 0.7),
    ("ar1", 0.9),
]

# methods to compare
METHODS = {
    "ENet": {"alpha": 1.0, "rho": 0.5},
    "TLasso": {"alpha": 0.5, "rho": 1.0},
    "TENet": {"alpha": 0.5, "rho": 0.5},
}

# r grid for grouping effect experiments
GROUP_R_VALUES = [0.0, 0.5, 0.7, 0.8, 0.9]

# plot settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.linewidth": 1.2,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


def standardize_X_y(X, y):
    n = X.shape[0]
    y_c = y - y.mean()
    X_c = X - X.mean(axis=0, keepdims=True)
    scale = np.sqrt((np.sum(X_c**2, axis=0) / n) + 1e-12)
    X_s = X_c / scale
    return X_s, y_c


def make_beta_star(p, s, start=2.0, step=0.2):
    """
    Construct β* with s non-zero elements decaying like (2, -1.8, 1.6, -1.4, ...).
    """
    vals = np.array([start - step * i for i in range(s)])
    signs = np.array([1 if (i % 2 == 0) else -1 for i in range(s)])
    nz = vals * signs
    beta_star = np.zeros(p)
    beta_star[:s] = nz
    return beta_star


def make_covariance(structure, p, param):
    if structure == "equiv_corr":
        r = float(param)
        Sigma = np.full((p, p), r, dtype=float)
        np.fill_diagonal(Sigma, 1.0)
        return Sigma
    elif structure == "ar1":
        phi = float(param)
        idx = np.arange(p)
        Sigma = phi ** np.abs(idx[:, None] - idx[None, :])
        return Sigma


def simulate_data(n, p, Sigma, beta_star, sigma, rng):
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(Sigma.shape[0]))
    Z = rng.normal(size=(n, Sigma.shape[0]))
    X = Z @ L.T
    eps = rng.normal(loc=0.0, scale=sigma, size=n)
    y = X @ beta_star + eps
    Xs, yc = standardize_X_y(X, y)
    return Xs, yc


def set_lambda(n, p, sigma, scale=1.0):
    return float(scale * sigma * np.sqrt(2.0 * np.log(2 * p) / n))


def l2_error(bhat, bstar):
    return float(np.linalg.norm(bhat - bstar, 2))


# Error-bound experiments

rng = np.random.default_rng(SEED_NUM)
beta_star = make_beta_star(P, S)
beta_tilde = beta_star.copy()
lam = set_lambda(N, P, SIGMA_NOISE)

print("Error Bound Experiments")
print(f"lambda={lam:.4f}")

rows = []

for structure, param in CORR_SETTINGS:
    Sigma = make_covariance(structure, P, param)

    for rep in tqdm(range(N_SIM)):
        X, y = simulate_data(N, P, Sigma, beta_star, SIGMA_NOISE, rng)

        for mname, pars in METHODS.items():
            alpha, rho = pars["alpha"], pars["rho"]
            beta_hat = estimateTransferElasticNet(
                X, y, lam, alpha, rho, beta_tilde, skip_intercept=False, max_iter=1000, tol=1e-4
            )
            err = l2_error(beta_hat, beta_star)
            rows.append(
                {
                    "structure": f"{structure} ({param})",
                    "method": mname,
                    "rep": rep,
                    "l2_error": err,
                }
            )

    df_tmp = pd.DataFrame(
        [r for r in rows if r["structure"] == f"{structure} ({param})"]
    )
    fig = plt.figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    methods_order = list(METHODS.keys())
    data_to_plot = [
        df_tmp.loc[df_tmp["method"] == m, "l2_error"].values for m in methods_order
    ]
    ax.boxplot(data_to_plot, labels=methods_order, showmeans=True)
    ax.set_ylabel(r"$\|\hat{\beta} - \beta^*\|_2$")
    ax.set_xlabel("Method")
    plt.tight_layout()
    if SAVE_PDF:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / f"l2_error_{param}.pdf")
    plt.show()

df_err = pd.DataFrame(rows)
summary = df_err.groupby(["structure", "method"], as_index=False).agg(
    median_l2_error=("l2_error", "median"),
    q3_l2_error=("l2_error", lambda s: s.quantile(0.75)),
    max_l2_error=("l2_error", "max"),
)
print(summary)


# Grouping effect experiments

print("Grouping Effect Experiments")

def make_cov_with_one_correlated_pair(p, j, k, r):
    Sigma = np.eye(p)
    Sigma[j, k] = Sigma[k, j] = r
    return Sigma


rng = np.random.default_rng(SEED_NUM + 10000)
beta_star = make_beta_star(P, S)
j_idx, k_idx = 0, 1

lam = set_lambda(N, P, SIGMA_NOISE)

recs = []

for r in GROUP_R_VALUES:
    diffs_TENet = []
    diffs_ENet = []
    diffs_TLasso = []

    for rep in tqdm(range(N_SIM)):
        Sigma = make_cov_with_one_correlated_pair(P, j_idx, k_idx, r)
        X, y = simulate_data(N, P, Sigma, beta_star, SIGMA_NOISE, rng)

        # TENet
        alpha_T, rho_T = METHODS["TENet"]["alpha"], METHODS["TENet"]["rho"]
        bh_T = estimateTransferElasticNet(X, y, lam, alpha_T, rho_T, beta_tilde, skip_intercept=False)
        diffs_TENet.append(abs(bh_T[j_idx] - bh_T[k_idx]))

        # ENet
        alpha_E, rho_E = METHODS["ENet"]["alpha"], METHODS["ENet"]["rho"]
        bh_E = estimateTransferElasticNet(X, y, lam, alpha_E, rho_E, beta_tilde, skip_intercept=False)
        diffs_ENet.append(abs(bh_E[j_idx] - bh_E[k_idx]))

        # TLasso
        alpha_L, rho_L = METHODS["TLasso"]["alpha"], METHODS["TLasso"]["rho"]
        bh_L = estimateTransferElasticNet(X, y, lam, alpha_L, rho_L, beta_tilde, skip_intercept=False)
        diffs_TLasso.append(abs(bh_L[j_idx] - bh_L[k_idx]))

    mean_T = float(np.mean(diffs_TENet))
    mean_E = float(np.mean(diffs_ENet))
    mean_L = float(np.mean(diffs_TLasso))
    std_diff_T = float(np.std(diffs_TENet))
    std_diff_E = float(np.std(diffs_ENet))
    std_diff_L = float(np.std(diffs_TLasso))

    recs.append(
        {"r": r, "method": "TENet", "mean_abs_diff": mean_T, "std_diff": std_diff_T}
    )
    recs.append(
        {"r": r, "method": "ENet", "mean_abs_diff": mean_E, "std_diff": std_diff_E}
    )
    recs.append(
        {"r": r, "method": "TLasso", "mean_abs_diff": mean_L, "std_diff": std_diff_L}
    )

df_group = pd.DataFrame(recs)

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
for m in ["ENet", "TLasso", "TENet"]:
    sub = df_group[df_group["method"] == m]
    ax.plot(sub["r"].values, sub["mean_abs_diff"].values, marker="o", label=m, lw=2)
    ax.fill_between(
        sub["r"].values,
        sub["mean_abs_diff"].values - sub["std_diff"].values,
        sub["mean_abs_diff"].values + sub["std_diff"].values,
        alpha=0.15,
    )
ax.axhline(
    (1 - alpha) * np.abs(beta_star[j_idx] - beta_star[k_idx]),
    lw=1.5,
    linestyle="--",
    color="gray",
    label=r"$(1-\alpha) |{\beta}^{*}_j - {\beta}^{*}_k|$",
)
ax.set_xlabel("Correlation between the covariate pair")
ax.set_ylabel(r"$|\hat{\beta}_j - \hat{\beta}_k|$")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
if SAVE_PDF:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "grouping_effect_mean_abs_diff_vs_r.pdf")
plt.show()

pivot_tbl = df_group.pivot(
    index="r", columns="method", values="mean_abs_diff"
).reset_index()
print(pivot_tbl)
