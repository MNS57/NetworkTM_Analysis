# NetworkTM_Analysis

Preliminary analyses of public intradomain traffic matrices (GEANT / TOTEM),
treating traffic-matrix problems as inverse problems on partially observed
multivariate time series.

通信ネットワークの公開トラフィック行列（GEANT, TOTEMプロジェクト）を対象に、
「一部しか観測できない多変量時系列からの逆推定」という観点で行った予備解析です。
神経活動の多点計測で用いてきた推定の手続き（有向依存の推定、低ランク性などの
先験的制約、代理データによる有意性評価）を、そのまま通信データに適用しています。

## Analyses / 解析内容

1. **Spectral structure** — singular-value spectrum of the traffic matrix,
   raw vs. log scale. The effective dimensionality depends strongly on the
   scale of observation (a single heavy flow dominates the raw spectrum).
2. **Recovery under partial observation** — randomly hide 30–90% of OD
   entries and reconstruct them by soft-impute matrix completion
   (singular-value soft-thresholding), against a per-flow-mean baseline.
3. **Directed dependence between nodes** — binned transfer entropy between
   node-level traffic series, with circular-shift surrogates for the
   significance threshold.
4. **Anomaly source localization** — residuals after removing the daily
   profile, localized to individual OD pairs.

## Data / データ

GEANT traffic matrices (23 nodes, 529 OD pairs, 15-min bins, Jan–Apr 2005),
published by the TOTEM project. The data file is **not** redistributed here.

> S. Uhlig, B. Quoitin, J. Lepropre, S. Balon,
> "Providing public intradomain traffic matrices to the research community",
> ACM SIGCOMM Computer Communication Review, 36(1), 2006.

A flat CSV copy (`geant-flat-tms.csv`, one row per 15-min bin, 529 columns)
is available in public mirrors of the TOTEM dataset. Place it anywhere and run:

```
pip install numpy pandas matplotlib
python analysis.py path/to/geant-flat-tms.csv
```

Outputs: `figures/fig1_spectrum_completion.png`,
`figures/fig2_te_network.png`, `results.json`.

## Notes / 留意点

- Low-rank completion clearly beats the baseline up to ~70% missing entries,
  and breaks down around 90% — where additional structure (temporal
  regularity, gravity-type priors, routing information) becomes necessary.
  この破れ方自体が、先験的制約の設計という次の研究課題を定めます。
- Transfer entropy here measures directed statistical dependence between
  observed series; mapping it onto physical routes requires the routing
  matrix, which is the subject of ongoing work.
