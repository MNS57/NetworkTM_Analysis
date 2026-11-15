# -*- coding: utf-8 -*-
"""Preliminary analysis of public traffic-matrix time series (GEANT / TOTEM).

Analyses
  1. Spectral structure of the traffic matrix (raw vs. log scale)
  2. Estimation of unobserved OD flows under partial observation
     (soft-impute matrix completion vs. per-flow-mean baseline)
  3. Directed statistical dependence between node-level series
     (binned transfer entropy with circular-shift surrogates)
  4. Anomaly detection and source localization at the OD-flow level

Data: GEANT intradomain traffic matrices (23 nodes, 529 OD pairs,
15-min bins, Jan-Apr 2005), provided by the TOTEM project.
  S. Uhlig, B. Quoitin, J. Lepropre, S. Balon,
  "Providing public intradomain traffic matrices to the research
  community", ACM SIGCOMM CCR, 36(1), 2006.
The data file is NOT redistributed here; see README for how to obtain it.

Usage:  python analysis.py <path/to/geant-flat-tms.csv>
"""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)

def load_tms(csv_path):
    df = pd.read_csv(csv_path, header=None)
    ts = pd.to_datetime(df[0], format='%Y-%m-%d-%H-%M')
    X = df.iloc[:, 1:].to_numpy(float)
    return ts, X

def spectral_structure(X):
    out = {}
    for name, M in (('raw', X), ('log', np.log1p(X))):
        s = np.linalg.svd(M - M.mean(0), compute_uv=False)
        e = np.cumsum(s**2) / np.sum(s**2)
        out[name] = {'sv': s, 'energy': e,
                     'k90': int(np.searchsorted(e, 0.90)) + 1,
                     'k95': int(np.searchsorted(e, 0.95)) + 1}
    return out

def soft_impute(X_obs, mask, lam_frac=0.02, iters=80):
    """Iterative singular-value soft-thresholding (Mazumder et al., 2010)."""
    Z = np.where(mask, X_obs, 0.0)
    lam = np.linalg.svd(Z, compute_uv=False)[0] * lam_frac
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        Zl = (U * np.maximum(s - lam, 0)) @ Vt
        Z = np.where(mask, X_obs, Zl)
    return Zl

def completion_experiment(X, rates=(0.3, 0.5, 0.7, 0.9), stride=4, tmax=2000):
    L = np.log1p(X)[::stride][:tmax]
    rows = []
    for p in rates:
        mask = RNG.random(L.shape) > p          # True = observed
        rec = soft_impute(L, mask)
        true = np.expm1(L[~mask]); est = np.expm1(np.maximum(rec[~mask], 0))
        nmae = np.sum(np.abs(true - est)) / np.sum(np.abs(true))
        cm = np.nanmean(np.where(mask, L, np.nan), 0)
        base = np.expm1(np.broadcast_to(cm, L.shape)[~mask])
        nmae_b = np.sum(np.abs(true - base)) / np.sum(np.abs(true))
        rows.append({'missing': p, 'soft_impute': nmae, 'flow_mean': nmae_b})
    return rows

def transfer_entropy(x, y, bins=4):
    """TE x->y with history length 1, equiquantal binning."""
    def digit(v):
        q = np.quantile(v, np.linspace(0, 1, bins + 1))
        return np.clip(np.searchsorted(q, v, 'right') - 1, 0, bins - 1)
    dx, dy = digit(x), digit(y)
    y1, y0, x0 = dy[1:], dy[:-1], dx[:-1]
    def H(*a):
        idx = np.ravel_multi_index(a, [bins] * len(a))
        c = np.bincount(idx, minlength=bins ** len(a)).astype(float)
        p = c[c > 0] / c.sum()
        return -np.sum(p * np.log(p))
    return H(y1, y0) + H(y0, x0) - H(y1, y0, x0) - H(y0)

def dependence_network(X, n_nodes, stride=4, n_surrogate=200):
    T = X.shape[0]
    node_out = X.reshape(T, n_nodes, n_nodes).sum(2)
    S = np.log1p(node_out[::stride])
    n = n_nodes
    TE = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                TE[i, j] = transfer_entropy(S[:, i], S[:, j])
    null = []
    for _ in range(n_surrogate):
        i, j = RNG.integers(n, size=2)
        if i == j: continue
        shift = RNG.integers(100, S.shape[0] - 100)
        null.append(transfer_entropy(np.roll(S[:, i], shift), S[:, j]))
    thr = float(np.quantile(null, 0.99))
    return TE, thr

def anomaly_localization(X, ts, period=96, z_thr=6.0):
    L = np.log1p(X)
    med = np.zeros_like(L)
    for k in range(period):
        med[k::period] = np.median(L[k::period], 0)
    R = L - med
    z = (R - R.mean(0)) / (R.std(0) + 1e-9)
    hits = np.argwhere(z > z_thr)
    order = np.argsort(-z[hits[:, 0], hits[:, 1]])
    return hits[order], z

def main(csv_path):
    ts, X = load_tms(csv_path)
    T, F = X.shape
    n = int(round(F ** 0.5))
    print(f"T={T}  flows={F}  nodes={n}  {ts.iloc[0]} .. {ts.iloc[-1]}")

    spec = spectral_structure(X)
    comp = completion_experiment(X)
    TE, thr = dependence_network(X, n)
    hits, z = anomaly_localization(X, ts)

    # ---- figure 1: spectrum & completion ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].semilogy(spec['raw']['sv'][:60] / spec['raw']['sv'][0], 'o-', ms=3, label='raw')
    ax[0].semilogy(spec['log']['sv'][:60] / spec['log']['sv'][0], 's-', ms=3, label='log')
    ax[0].set_xlabel('component'); ax[0].set_ylabel('normalized singular value')
    ax[0].set_title('Spectral structure of GEANT TM'); ax[0].legend()
    r = [c['missing'] for c in comp]
    ax[1].plot(r, [c['soft_impute'] for c in comp], 'o-', label='soft-impute (low-rank)')
    ax[1].plot(r, [c['flow_mean'] for c in comp], 's--', label='per-flow mean')
    ax[1].set_xlabel('fraction of unobserved entries'); ax[1].set_ylabel('NMAE')
    ax[1].set_title('OD-flow recovery under partial observation'); ax[1].legend()
    fig.tight_layout(); fig.savefig('figures/fig1_spectrum_completion.png', dpi=200)

    # ---- figure 2: directed dependence ----
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(TE, cmap='viridis')
    ax.set_title(f'Transfer entropy between node series\n(99% surrogate threshold = {thr:.3f})')
    ax.set_xlabel('target node'); ax.set_ylabel('source node')
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig('figures/fig2_te_network.png', dpi=200)

    results = {
        'data': {'T': T, 'flows': F, 'nodes': n,
                 'start': str(ts.iloc[0]), 'end': str(ts.iloc[-1])},
        'spectral': {k: {'k90': v['k90'], 'k95': v['k95']} for k, v in spec.items()},
        'completion_nmae': comp,
        'te_edges_above_99pct_surrogate': int((TE > thr).sum()),
        'te_threshold': thr,
        'anomalies_z_gt_6': int(len(hits)),
        'top_anomalies': [
            {'time': str(ts.iloc[t]), 'od': f'{f // n}->{f % n}', 'z': float(z[t, f])}
            for t, f in hits[:10]],
    }
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2)[:1200])

if __name__ == '__main__':
    main(sys.argv[1])
