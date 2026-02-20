from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import genpareto


FEATURE_NAMES = {
    0: "inflow",
    1: "outflow",
}


def _parse_quantile(value: str) -> float:
    q = float(value)
    if q > 1.0:
        q = q / 100.0
    if not (0.0 < q < 1.0):
        raise argparse.ArgumentTypeError("quantile must be in (0,1) or (0,100)")
    return q


def _mask(arr: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    if mode == "gt":
        return arr > threshold
    if mode == "ge":
        return arr >= threshold
    if mode == "lt":
        return arr < threshold
    if mode == "le":
        return arr <= threshold
    raise ValueError(f"unknown filter mode: {mode}")


@dataclass(frozen=True)
class FitResult:
    threshold_u: float | None
    n_filtered: int
    n_exceed: int
    xi: float | None
    sigma: float | None
    status: str


def _fit_gpd_exceedances(
    series: np.ndarray,
    *,
    min_value: float,
    filter_mode: str,
    tail_quantile: float,
    min_exceedances: int,
) -> tuple[np.ndarray | None, FitResult]:
    series = np.asarray(series).astype(float)
    series = series[np.isfinite(series)]

    filtered = series[_mask(series, min_value, filter_mode)]
    if filtered.size == 0:
        return None, FitResult(
            threshold_u=None,
            n_filtered=0,
            n_exceed=0,
            xi=None,
            sigma=None,
            status="skip:no_filtered_samples",
        )

    u = float(np.quantile(filtered, tail_quantile))
    exceedances = filtered[filtered > u] - u
    if exceedances.size < min_exceedances:
        return None, FitResult(
            threshold_u=u,
            n_filtered=int(filtered.size),
            n_exceed=int(exceedances.size),
            xi=None,
            sigma=None,
            status=f"skip:too_few_exceedances(<{min_exceedances})",
        )

    try:
        xi, loc, sigma = genpareto.fit(exceedances, floc=0)
    except Exception as exc:  # noqa: BLE001 - diagnostic tool
        return None, FitResult(
            threshold_u=u,
            n_filtered=int(filtered.size),
            n_exceed=int(exceedances.size),
            xi=None,
            sigma=None,
            status=f"error:fit_failed:{type(exc).__name__}",
        )

    if not (np.isfinite(xi) and np.isfinite(sigma) and sigma > 0):
        return None, FitResult(
            threshold_u=u,
            n_filtered=int(filtered.size),
            n_exceed=int(exceedances.size),
            xi=None,
            sigma=None,
            status="error:non_finite_params",
        )

    return exceedances, FitResult(
        threshold_u=u,
        n_filtered=int(filtered.size),
        n_exceed=int(exceedances.size),
        xi=float(xi),
        sigma=float(sigma),
        status="ok",
    )


def _qq_plot_gpd(ax, exceedances: np.ndarray, xi: float, sigma: float, *, ci_level: float | None) -> None:
    exceedances = np.asarray(exceedances, dtype=float)
    exceedances = exceedances[np.isfinite(exceedances)]
    exceedances = np.sort(exceedances)
    n = exceedances.size
    if n == 0:
        ax.text(0.5, 0.5, "skip:no_exceedances", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    probs = (np.arange(1, n + 1) - 0.5) / n  # avoid 0 and 1
    theoretical = genpareto.ppf(probs, c=xi, loc=0, scale=sigma)

    ci_lo = None
    ci_hi = None
    ok_points = np.isfinite(theoretical)
    if ci_level is not None:
        # Pointwise CI band for order statistics under the fitted distribution:
        # If X_i ~ F, then F(X_(i)) ~ Beta(i, n-i+1).
        alpha = 1.0 - float(ci_level)
        order = np.arange(1, n + 1)
        p_lo = beta_dist.ppf(alpha / 2, order, n - order + 1)
        p_hi = beta_dist.ppf(1 - alpha / 2, order, n - order + 1)
        ci_lo = genpareto.ppf(p_lo, c=xi, loc=0, scale=sigma)
        ci_hi = genpareto.ppf(p_hi, c=xi, loc=0, scale=sigma)

    exceedances_points = exceedances[ok_points]
    theoretical_points = theoretical[ok_points]

    if theoretical_points.size == 0:
        ax.text(0.5, 0.5, "skip:non_finite_quantiles", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    if ci_lo is not None and ci_hi is not None:
        ok_ci = ok_points & np.isfinite(ci_lo) & np.isfinite(ci_hi)
        if np.any(ok_ci):
            ax.fill_between(theoretical[ok_ci], ci_lo[ok_ci], ci_hi[ok_ci], color="0.7", alpha=0.25, linewidth=0)

    ax.scatter(theoretical_points, exceedances_points, s=10, alpha=0.6)
    max_candidates = [theoretical.max(initial=0.0), exceedances.max(initial=0.0)]
    max_candidates = [theoretical_points.max(initial=0.0), exceedances_points.max(initial=0.0)]
    max_val = float(max(max_candidates))
    ax.plot([0, max_val], [0, max_val], "r--", lw=1)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Theoretical quantiles (GPD)")
    ax.set_ylabel("Empirical exceedances")
    ax.grid(True, alpha=0.25)

def _mrl_series(
    series: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    thresholds = np.asarray(thresholds, dtype=float)
    thresholds = thresholds[np.isfinite(thresholds)]

    mean_excess = np.full_like(thresholds, fill_value=np.nan, dtype=float)
    support = np.zeros_like(thresholds, dtype=int)
    for i, u in enumerate(thresholds):
        exceedances = series[series > u] - u
        support[i] = int(exceedances.size)
        if exceedances.size > 0:
            mean_excess[i] = float(np.mean(exceedances))
    return mean_excess, support


def _runs_declustering_cluster_maxima(series: np.ndarray, u: float, runs: int) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if series.size == 0:
        return np.array([], dtype=float)

    exceed_idx = np.flatnonzero(series > u)
    if exceed_idx.size == 0:
        return np.array([], dtype=float)

    runs = max(0, int(runs))
    maxima: list[float] = []
    current_max = float(series[exceed_idx[0]])
    prev = int(exceed_idx[0])
    for idx in exceed_idx[1:]:
        idx = int(idx)
        if idx - prev > runs:
            maxima.append(current_max)
            current_max = float(series[idx])
        else:
            v = float(series[idx])
            if v > current_max:
                current_max = v
        prev = idx
    maxima.append(current_max)
    return np.asarray(maxima, dtype=float)


def _quantile_from_sorted(sorted_x: np.ndarray, q: float) -> float:
    n = int(sorted_x.size)
    if n == 0:
        return float("nan")
    if q <= 0.0:
        return float(sorted_x[0])
    if q >= 1.0:
        return float(sorted_x[-1])
    pos = (n - 1) * q
    idx = int(np.floor(pos))
    frac = float(pos - idx)
    if idx >= n - 1:
        return float(sorted_x[-1])
    return float((1.0 - frac) * sorted_x[idx] + frac * sorted_x[idx + 1])


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0 or n_boot <= 0:
        return float("nan"), float("nan")

    alpha = 1.0 - float(ci_level)
    means = np.empty((n_boot,), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(values[idx]))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def _bootstrap_mrl_ci(
    filtered: np.ndarray,
    qs: np.ndarray,
    *,
    n_boot: int,
    ci_level: float,
    rng: np.random.Generator,
    sample_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    filtered = np.asarray(filtered, dtype=float)
    filtered = filtered[np.isfinite(filtered)]
    qs = np.asarray(qs, dtype=float)
    qs = qs[np.isfinite(qs)]

    n = int(filtered.size)
    if n < 2 or n_boot <= 0 or qs.size == 0:
        nan_arr = np.full((qs.size,), np.nan, dtype=float)
        return nan_arr, nan_arr

    draw_n = n if sample_size is None else int(sample_size)
    draw_n = max(2, min(draw_n, n))

    boot_me = np.full((n_boot, qs.size), np.nan, dtype=float)

    total = 0.0
    alpha = 1.0 - float(ci_level)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=draw_n)
        sample = np.sort(filtered[idx])
        csum = np.cumsum(sample)
        total = float(csum[-1])

        for j, q in enumerate(qs):
            u = _quantile_from_sorted(sample, float(q))
            k = int(np.searchsorted(sample, u, side="right"))
            tail_n = draw_n - k
            if tail_n <= 0:
                continue
            tail_sum = total - float(csum[k - 1]) if k > 0 else total
            boot_me[b, j] = tail_sum / tail_n - u

    lo = np.nanquantile(boot_me, alpha / 2, axis=0)
    hi = np.nanquantile(boot_me, 1 - alpha / 2, axis=0)
    return lo, hi


def _fit_gpd_series_over_thresholds(
    ts_raw: np.ndarray,
    *,
    min_value: float,
    filter_mode: str,
    thresholds: np.ndarray,
    min_exceedances: int,
    decluster_runs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    ts_raw = np.asarray(ts_raw, dtype=float)
    ts_raw = ts_raw[np.isfinite(ts_raw)]
    thresholds = np.asarray(thresholds, dtype=float)

    xi = np.full((thresholds.size,), np.nan, dtype=float)
    sigma = np.full((thresholds.size,), np.nan, dtype=float)
    support = np.zeros((thresholds.size,), dtype=int)
    status: list[str] = [""] * int(thresholds.size)

    filtered = ts_raw[_mask(ts_raw, min_value, filter_mode)]

    for j, u in enumerate(thresholds):
        u = float(u)
        if not np.isfinite(u):
            status[j] = "skip:non_finite_u"
            continue

        if decluster_runs > 0:
            maxima = _runs_declustering_cluster_maxima(ts_raw, u, runs=int(decluster_runs))
            z = maxima - u
            z = z[np.isfinite(z)]
            z = z[z > 0]
        else:
            z = filtered[filtered > u] - u

        support[j] = int(z.size)
        if z.size < int(min_exceedances):
            status[j] = f"skip:too_few_exceedances(<{min_exceedances})"
            continue

        try:
            xi_hat, loc, sigma_hat = genpareto.fit(z, floc=0)
        except Exception as exc:  # noqa: BLE001 - diagnostic tool
            status[j] = f"error:fit_failed:{type(exc).__name__}"
            continue

        if not (np.isfinite(xi_hat) and np.isfinite(sigma_hat) and sigma_hat > 0):
            status[j] = "error:non_finite_params"
            continue

        xi[j] = float(xi_hat)
        sigma[j] = float(sigma_hat)
        status[j] = "ok"

    return xi, sigma, support, status


def _return_level_pot(
    *,
    u: float,
    xi: float,
    sigma: float,
    p_u: float,
    return_period: float,
    eps: float = 1.0e-12,
) -> float:
    """
    POT return level for return period `return_period` (in units of the chosen sampling interval).

    Uses approximation:
      P(X > x) ≈ p_u * (1 + xi*(x-u)/sigma)^(-1/xi)
    where p_u = P(X > u).

    Solving P(X > x_m) = 1/return_period gives:
      x_m = u + (sigma/xi) * ((return_period * p_u)^xi - 1)   if xi != 0
      x_m = u + sigma * log(return_period * p_u)              if xi == 0
    Only meaningful in the tail when return_period * p_u >= 1.
    """
    if not (np.isfinite(u) and np.isfinite(xi) and np.isfinite(sigma) and np.isfinite(p_u) and np.isfinite(return_period)):
        return float("nan")
    if sigma <= 0 or p_u <= 0 or return_period <= 0:
        return float("nan")

    t = float(return_period) * float(p_u)
    if t < 1.0:
        return float("nan")

    if abs(float(xi)) < eps:
        return float(u + sigma * np.log(t))
    return float(u + (sigma / xi) * (t**xi - 1.0))


def _plot_stability_with_support(
    ax,
    *,
    thresholds: np.ndarray,
    y: np.ndarray,
    y_label: str,
    support: np.ndarray,
    support_label: str,
    support_scale: str,
    min_support_ref: int | None,
    vline_u: float | None,
) -> None:
    thresholds = np.asarray(thresholds, dtype=float)
    y = np.asarray(y, dtype=float)
    support = np.asarray(support, dtype=int)

    ok = np.isfinite(thresholds) & np.isfinite(y)
    if np.any(ok):
        ax.plot(thresholds[ok], y[ok], "o-", color="C0", markersize=3, lw=1.2)
    else:
        ax.text(
            0.5,
            0.5,
            "no fits\n(insufficient support)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Threshold u")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)

    if vline_u is not None and np.isfinite(vline_u):
        ax.axvline(float(vline_u), color="k", linestyle="--", lw=1, alpha=0.6)

    if np.any(np.isfinite(thresholds)):
        ax.set_xlim(float(np.nanmin(thresholds)), float(np.nanmax(thresholds)))

    ax2 = ax.twinx()
    support_plot = support.astype(float)
    ok_sup = np.isfinite(thresholds) & np.isfinite(support_plot)
    if support_scale == "log":
        support_plot[support_plot < 1] = np.nan
    ax2.plot(thresholds[ok_sup], support_plot[ok_sup], "x-", color="C1", markersize=3, lw=1.0, alpha=0.85)
    ax2.set_ylabel(support_label)
    if support_scale == "log":
        ax2.set_yscale("log")
        ax2.set_ylim(bottom=1)
    else:
        ax2.set_ylim(bottom=0)
    if min_support_ref is not None and min_support_ref > 0:
        ax2.axhline(float(min_support_ref), color="C1", linestyle=":", lw=1, alpha=0.8)


def _plot_mrl_with_support(
    ax,
    *,
    thresholds: np.ndarray,
    mean_excess: np.ndarray,
    mean_excess_ci: tuple[np.ndarray, np.ndarray] | None,
    support: np.ndarray,
    support_scale: str,
    support_ref: int | None = None,
    fit_threshold_u: float | None = None,
    support_label: str = "Support (# exceedances)",
) -> None:
    thresholds = np.asarray(thresholds, dtype=float)
    mean_excess = np.asarray(mean_excess, dtype=float)
    support = np.asarray(support, dtype=int)

    ok_me = np.isfinite(thresholds) & np.isfinite(mean_excess)
    ax.plot(thresholds[ok_me], mean_excess[ok_me], "o-", color="C0", markersize=3, lw=1.2)
    if mean_excess_ci is not None:
        ci_lo, ci_hi = mean_excess_ci
        ci_lo = np.asarray(ci_lo, dtype=float)
        ci_hi = np.asarray(ci_hi, dtype=float)
        ok_ci = np.isfinite(thresholds) & np.isfinite(ci_lo) & np.isfinite(ci_hi)
        if np.any(ok_ci):
            ax.fill_between(thresholds[ok_ci], ci_lo[ok_ci], ci_hi[ok_ci], color="C0", alpha=0.18, linewidth=0)
    ax.set_xlabel("Threshold u")
    ax.set_ylabel("Mean excess E[X-u | X>u]")
    ax.grid(True, alpha=0.25)

    if fit_threshold_u is not None and np.isfinite(fit_threshold_u):
        ax.axvline(float(fit_threshold_u), color="k", linestyle="--", lw=1, alpha=0.6)

    ax2 = ax.twinx()
    support_plot = support.astype(float)
    ok_sup = np.isfinite(thresholds) & np.isfinite(support_plot)
    if support_scale == "log":
        support_plot[support_plot < 1] = np.nan
    ax2.plot(thresholds[ok_sup], support_plot[ok_sup], "x-", color="C1", markersize=3, lw=1.0, alpha=0.85)
    ax2.set_ylabel(support_label)
    if support_scale == "log":
        ax2.set_yscale("log")
        ax2.set_ylim(bottom=1)
    else:
        ax2.set_ylim(bottom=0)

    if support_ref is not None and support_ref > 0:
        ax2.axhline(float(support_ref), color="C1", linestyle=":", lw=1, alpha=0.8)


def _load_y(data_dir: Path, dataset: str, split: str) -> np.ndarray:
    path = data_dir / dataset / f"{split}.npz"
    with np.load(path) as data:
        y = data["y"]
    if y.ndim != 4:
        raise ValueError(f"Expected y to have 4 dims [S,1,N,F], got {y.shape} in {path}")
    return y[:, 0, :, :]  # (S, N, F)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit per-node GPDs (via MLE) on tail exceedances and generate QQ plots for the top-K most active nodes."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_with_EVs"))
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset folder names under --data-dir (default: all subdirectories)",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-value", type=float, default=5.0)
    parser.add_argument(
        "--filter-mode",
        choices=["gt", "ge", "lt", "le"],
        default="gt",
        help="Keep samples relative to --min-value (default: gt keeps > min-value)",
    )
    parser.add_argument(
        "--qq-ci-level",
        type=_parse_quantile,
        default=0.95,
        help="Pointwise QQ confidence band level under fitted GPD (e.g. 0.95 or 95).",
    )
    parser.add_argument("--tail-quantile", type=_parse_quantile, default=0.95)
    parser.add_argument("--min-exceedances", type=int, default=30)
    parser.add_argument("--mrl-quantile-min", type=_parse_quantile, default=0.90)
    parser.add_argument("--mrl-quantile-max", type=_parse_quantile, default=0.99)
    parser.add_argument("--mrl-num-points", type=int, default=25)
    parser.add_argument("--mrl-support-scale", choices=["linear", "log"], default="linear")
    parser.add_argument(
        "--decluster-runs",
        type=int,
        default=0,
        help=(
            "If >0, use runs declustering for MRL support/mean-excess at each threshold; "
            "clusters are split when exceedances are separated by > runs timesteps (default: 0/off)."
        ),
    )
    parser.add_argument(
        "--mrl-bootstrap-n",
        type=int,
        default=0,
        help="If >0, run nonparametric bootstrap and draw CI band for MRL mean-excess curve (default: 0/off).",
    )
    parser.add_argument(
        "--mrl-ci-level",
        type=_parse_quantile,
        default=0.95,
        help="MRL bootstrap CI level (e.g. 0.95 or 95).",
    )
    parser.add_argument(
        "--mrl-bootstrap-seed",
        type=int,
        default=0,
        help="Seed for MRL bootstrap RNG (default: 0).",
    )
    parser.add_argument(
        "--mrl-bootstrap-sample-size",
        type=int,
        default=None,
        help="Optional bootstrap resample size (default: full filtered length).",
    )
    parser.add_argument("--stability-quantile-min", type=_parse_quantile, default=None)
    parser.add_argument("--stability-quantile-max", type=_parse_quantile, default=None)
    parser.add_argument("--stability-num-points", type=int, default=None)
    parser.add_argument(
        "--stability-min-exceedances",
        type=int,
        default=None,
        help="Min exceedances/clusters required for each stability fit (default: --min-exceedances).",
    )
    parser.add_argument("--stability-support-scale", choices=["linear", "log"], default="log")
    parser.add_argument(
        "--return-level-periods",
        nargs="*",
        type=float,
        default=[500.0, 1000.0, 5000.0],
        help="Return periods to plot for return-level stability (in units of the chosen denominator).",
    )
    parser.add_argument(
        "--return-level-denom",
        choices=["filtered", "total"],
        default="filtered",
        help="Denominator for exceedance rate p_u when computing return levels (default: filtered).",
    )
    parser.add_argument(
        "--histogram-scope",
        choices=["topk", "all"],
        default="all",
        help="Which nodes to include in xi/sigma histograms (default: all).",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=30,
        help="Number of bins for xi/sigma histograms (default: 30).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/gpd_node_qq"))
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if args.datasets is None:
        datasets = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    else:
        datasets = list(args.datasets)

    if not datasets:
        raise SystemExit(f"No datasets found under {data_dir}")

    if not (0.5 < float(args.qq_ci_level) < 1.0):
        raise SystemExit("--qq-ci-level must be in (0.5,1.0) or (50,100)")

    if args.mrl_quantile_min >= args.mrl_quantile_max:
        raise SystemExit("--mrl-quantile-min must be < --mrl-quantile-max")

    if args.mrl_bootstrap_n < 0:
        raise SystemExit("--mrl-bootstrap-n must be >= 0")
    if args.mrl_bootstrap_n > 0 and not (0.5 < float(args.mrl_ci_level) < 1.0):
        raise SystemExit("--mrl-ci-level must be in (0.5,1.0) or (50,100)")
    if args.decluster_runs < 0:
        raise SystemExit("--decluster-runs must be >= 0")

    stability_q_min = float(args.stability_quantile_min) if args.stability_quantile_min is not None else float(args.mrl_quantile_min)
    stability_q_max = float(args.stability_quantile_max) if args.stability_quantile_max is not None else float(args.mrl_quantile_max)
    stability_n = int(args.stability_num_points) if args.stability_num_points is not None else int(args.mrl_num_points)
    stability_min_exc = int(args.stability_min_exceedances) if args.stability_min_exceedances is not None else int(args.min_exceedances)
    if not (0.0 < stability_q_min < stability_q_max < 1.0):
        raise SystemExit("stability quantile range must be within (0,1) and min < max")
    if stability_n < 3:
        raise SystemExit("--stability-num-points must be >= 3")
    if stability_min_exc < 1:
        raise SystemExit("--stability-min-exceedances must be >= 1")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    all_records: list[dict] = []
    global_fit_records: list[dict] = []
    mrl_records: list[dict] = []
    stability_records: list[dict] = []
    return_level_records: list[dict] = []

    for dataset in datasets:
        y = _load_y(data_dir, dataset, args.split)  # (S, N, F)
        s, n, f = y.shape
        if f < 1:
            raise ValueError(f"Expected y to have at least 1 feature, got {y.shape}")

        activity_mask = _mask(y, args.min_value, args.filter_mode)
        activity = (y * activity_mask).sum(axis=(0, 2))  # (N,)
        top_nodes = np.argsort(activity)[::-1][: args.top_k]

        nrows = len(top_nodes)
        ncols = min(f, 2) if f >= 2 else 1
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6.0 * ncols, 3.25 * nrows),
            squeeze=False,
        )

        dataset_out = out_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)

        for row_idx, node in enumerate(top_nodes):
            for feat in range(ncols):
                ax = axes[row_idx][feat]
                ts = y[:, node, feat]
                exceedances, fit = _fit_gpd_exceedances(
                    ts,
                    min_value=args.min_value,
                    filter_mode=args.filter_mode,
                    tail_quantile=args.tail_quantile,
                    min_exceedances=args.min_exceedances,
                )

                feature_name = FEATURE_NAMES.get(feat, f"feature{feat}")
                title_prefix = f"node={int(node)} {feature_name}"
                if fit.status != "ok":
                    ax.text(
                        0.5,
                        0.5,
                        fit.status,
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(title_prefix)
                    ax.axis("off")
                else:
                    assert exceedances is not None
                    assert fit.xi is not None and fit.sigma is not None
                    _qq_plot_gpd(ax, exceedances, fit.xi, fit.sigma, ci_level=float(args.qq_ci_level))
                    ax.set_title(
                        f"{title_prefix} | q={args.tail_quantile:.2f} u={fit.threshold_u:.2f} "
                        f"n_exc={fit.n_exceed} ξ={fit.xi:.3f} σ={fit.sigma:.2f} ci={args.qq_ci_level*100:.0f}%"
                    )

                all_records.append(
                    {
                        "dataset": dataset,
                        "split": args.split,
                        "node": int(node),
                        "feature": int(feat),
                        "feature_name": feature_name,
                        "min_value": float(args.min_value),
                        "filter_mode": args.filter_mode,
                        "tail_quantile": float(args.tail_quantile),
                        "n_total": int(s),
                        "n_filtered": int(fit.n_filtered),
                        "threshold_u": fit.threshold_u,
                        "n_exceed": int(fit.n_exceed),
                        "xi": fit.xi,
                        "sigma": fit.sigma,
                        "status": fit.status,
                        "activity_score": float(activity[int(node)]),
                    }
                )

        fig.suptitle(
            f"{dataset} ({args.split}) top{args.top_k} nodes by activity | "
            f"filter={args.filter_mode}{args.min_value:g} | tail q={args.tail_quantile:.2f} | "
            f"QQ CI={args.qq_ci_level*100:.0f}%",
            y=0.995,
        )
        fig.tight_layout()

        png_path = dataset_out / (
            f"top{args.top_k}_qq_tailq{int(round(args.tail_quantile * 100)):02d}_"
            f"ci{int(round(args.qq_ci_level * 100)):02d}_"
            f"filter{args.filter_mode}{args.min_value:g}_{args.split}.png"
        )
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

        # Stability plots: xi_hat(u), sigma_hat(u) vs u (with support)
        stab_qs = np.linspace(stability_q_min, stability_q_max, stability_n)
        # Ensure the reference tail quantile is included so the stability curve can be read at the dashed line.
        if stability_q_min <= float(args.tail_quantile) <= stability_q_max:
            stab_qs = np.unique(np.sort(np.concatenate([stab_qs, np.array([float(args.tail_quantile)])])))
        xi_fig, xi_axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6.0 * ncols, 3.25 * nrows),
            squeeze=False,
        )
        sigma_fig, sigma_axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6.0 * ncols, 3.25 * nrows),
            squeeze=False,
        )
        rl_fig, rl_axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6.0 * ncols, 3.25 * nrows),
            squeeze=False,
        )

        for row_idx, node in enumerate(top_nodes):
            for feat in range(ncols):
                feature_name = FEATURE_NAMES.get(feat, f"feature{feat}")
                ts_raw = np.asarray(y[:, node, feat], dtype=float)
                ts_raw = ts_raw[np.isfinite(ts_raw)]
                filtered = ts_raw[_mask(ts_raw, args.min_value, args.filter_mode)]

                xi_ax = xi_axes[row_idx][feat]
                sigma_ax = sigma_axes[row_idx][feat]
                rl_ax = rl_axes[row_idx][feat]
                title_prefix = f"node={int(node)} {feature_name}"

                if filtered.size < 2:
                    for ax in (xi_ax, sigma_ax):
                        ax.text(0.5, 0.5, "skip:no_filtered_samples", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(title_prefix)
                        ax.axis("off")
                    rl_ax.text(0.5, 0.5, "skip:no_filtered_samples", ha="center", va="center", transform=rl_ax.transAxes)
                    rl_ax.set_title(title_prefix)
                    rl_ax.axis("off")
                    continue

                thresholds = np.quantile(filtered, stab_qs)
                _, fit_ref = _fit_gpd_exceedances(
                    ts_raw,
                    min_value=args.min_value,
                    filter_mode=args.filter_mode,
                    tail_quantile=args.tail_quantile,
                    min_exceedances=args.min_exceedances,
                )
                xi_hat, sigma_hat, support_hat, status_hat = _fit_gpd_series_over_thresholds(
                    ts_raw,
                    min_value=args.min_value,
                    filter_mode=args.filter_mode,
                    thresholds=thresholds,
                    min_exceedances=stability_min_exc,
                    decluster_runs=int(args.decluster_runs),
                )

                support_label = "Support (# clusters)" if args.decluster_runs > 0 else "Support (# exceedances)"
                _plot_stability_with_support(
                    xi_ax,
                    thresholds=thresholds,
                    y=xi_hat,
                    y_label="xi (shape)",
                    support=support_hat,
                    support_label=support_label,
                    support_scale=args.stability_support_scale,
                    min_support_ref=stability_min_exc,
                    vline_u=fit_ref.threshold_u,
                )
                _plot_stability_with_support(
                    sigma_ax,
                    thresholds=thresholds,
                    y=sigma_hat,
                    y_label="sigma (scale)",
                    support=support_hat,
                    support_label=support_label,
                    support_scale=args.stability_support_scale,
                    min_support_ref=stability_min_exc,
                    vline_u=fit_ref.threshold_u,
                )

                xi_ax.set_title(f"{title_prefix} | ref u@q={args.tail_quantile:.2f}: {fit_ref.threshold_u:.2f}")
                sigma_ax.set_title(f"{title_prefix} | ref u@q={args.tail_quantile:.2f}: {fit_ref.threshold_u:.2f}")

                # Return-level stability: z_T(u) curves for chosen return periods
                denom_n = float(filtered.size if args.return_level_denom == "filtered" else ts_raw.size)
                if denom_n <= 0:
                    denom_n = float("nan")
                support_plot = support_hat.astype(float)
                p_u = np.where(denom_n > 0, support_plot / denom_n, np.nan)

                rl_ax.set_xlabel("Threshold u")
                rl_ax.set_ylabel("Return level z_T")
                rl_ax.grid(True, alpha=0.25)
                if fit_ref.threshold_u is not None and np.isfinite(fit_ref.threshold_u):
                    rl_ax.axvline(float(fit_ref.threshold_u), color="k", linestyle="--", lw=1, alpha=0.6)

                if args.return_level_periods:
                    for T in args.return_level_periods:
                        zT = np.array(
                            [
                                _return_level_pot(u=float(u), xi=float(xi), sigma=float(sg), p_u=float(p), return_period=float(T))
                                for u, xi, sg, p in zip(thresholds, xi_hat, sigma_hat, p_u)
                            ],
                            dtype=float,
                        )
                        ok_rl = np.isfinite(thresholds) & np.isfinite(zT)
                        if np.any(ok_rl):
                            rl_ax.plot(thresholds[ok_rl], zT[ok_rl], "-", lw=1.2, label=f"T={T:g}")

                        for q, u, xih, sigh, sup, st, pu_val, z_val in zip(
                            stab_qs, thresholds, xi_hat, sigma_hat, support_hat, status_hat, p_u, zT
                        ):
                            return_level_records.append(
                                {
                                    "dataset": dataset,
                                    "split": args.split,
                                    "node": int(node),
                                    "feature": int(feat),
                                    "feature_name": feature_name,
                                    "min_value": float(args.min_value),
                                    "filter_mode": args.filter_mode,
                                    "tail_quantile_ref": float(args.tail_quantile),
                                    "threshold_u_ref": fit_ref.threshold_u,
                                    "stability_quantile": float(q),
                                    "threshold_u": float(u),
                                    "decluster_runs": int(args.decluster_runs),
                                    "min_exceedances": int(stability_min_exc),
                                    "support": int(sup),
                                    "denom": args.return_level_denom,
                                    "denom_n": float(denom_n),
                                    "p_u": float(pu_val) if np.isfinite(pu_val) else np.nan,
                                    "xi": float(xih) if np.isfinite(xih) else np.nan,
                                    "sigma": float(sigh) if np.isfinite(sigh) else np.nan,
                                    "return_period": float(T),
                                    "return_level": float(z_val) if np.isfinite(z_val) else np.nan,
                                    "status": st,
                                }
                            )

                    rl_ax.legend(fontsize=8, loc="best")

                # Support overlay on return-level panel
                rl_ax2 = rl_ax.twinx()
                sup_plot = support_hat.astype(float)
                ok_sup = np.isfinite(thresholds) & np.isfinite(sup_plot)
                if args.stability_support_scale == "log":
                    sup_plot[sup_plot < 1] = np.nan
                rl_ax2.plot(thresholds[ok_sup], sup_plot[ok_sup], "x-", color="C1", markersize=3, lw=1.0, alpha=0.6)
                rl_ax2.set_ylabel(support_label)
                if args.stability_support_scale == "log":
                    rl_ax2.set_yscale("log")
                    rl_ax2.set_ylim(bottom=1)
                else:
                    rl_ax2.set_ylim(bottom=0)
                rl_ax2.axhline(float(stability_min_exc), color="C1", linestyle=":", lw=1, alpha=0.8)

                rl_ax.set_title(f"{title_prefix} | denom={args.return_level_denom}")

                for q, u, xih, sigh, sup, st in zip(stab_qs, thresholds, xi_hat, sigma_hat, support_hat, status_hat):
                    stability_records.append(
                        {
                            "dataset": dataset,
                            "split": args.split,
                            "node": int(node),
                            "feature": int(feat),
                            "feature_name": feature_name,
                            "min_value": float(args.min_value),
                            "filter_mode": args.filter_mode,
                            "tail_quantile_ref": float(args.tail_quantile),
                            "threshold_u_ref": fit_ref.threshold_u,
                            "stability_quantile": float(q),
                            "threshold_u": float(u),
                            "decluster_runs": int(args.decluster_runs),
                            "min_exceedances": int(stability_min_exc),
                            "support": int(sup),
                            "xi": float(xih) if np.isfinite(xih) else np.nan,
                            "sigma": float(sigh) if np.isfinite(sigh) else np.nan,
                            "status": st,
                        }
                    )

        xi_fig.suptitle(
            f"{dataset} ({args.split}) top{args.top_k} stability: xi(u) | "
            f"filter={args.filter_mode}{args.min_value:g} | ref tail q={args.tail_quantile:.2f} | "
            f"u grid q={stability_q_min:.2f}..{stability_q_max:.2f} | decluster_runs={args.decluster_runs}",
            y=0.995,
        )
        xi_fig.tight_layout()
        xi_path = dataset_out / (
            f"top{args.top_k}_stability_xi_tailq{int(round(args.tail_quantile * 100)):02d}_"
            f"stabq{int(round(stability_q_min * 100)):02d}to{int(round(stability_q_max * 100)):02d}_"
            f"decluster{int(args.decluster_runs)}_filter{args.filter_mode}{args.min_value:g}_{args.split}.png"
        )
        xi_fig.savefig(xi_path, dpi=150)
        plt.close(xi_fig)

        sigma_fig.suptitle(
            f"{dataset} ({args.split}) top{args.top_k} stability: sigma(u) | "
            f"filter={args.filter_mode}{args.min_value:g} | ref tail q={args.tail_quantile:.2f} | "
            f"u grid q={stability_q_min:.2f}..{stability_q_max:.2f} | decluster_runs={args.decluster_runs}",
            y=0.995,
        )
        sigma_fig.tight_layout()
        sigma_path = dataset_out / (
            f"top{args.top_k}_stability_sigma_tailq{int(round(args.tail_quantile * 100)):02d}_"
            f"stabq{int(round(stability_q_min * 100)):02d}to{int(round(stability_q_max * 100)):02d}_"
            f"decluster{int(args.decluster_runs)}_filter{args.filter_mode}{args.min_value:g}_{args.split}.png"
        )
        sigma_fig.savefig(sigma_path, dpi=150)
        plt.close(sigma_fig)

        rl_fig.suptitle(
            f"{dataset} ({args.split}) top{args.top_k} return-level stability | "
            f"filter={args.filter_mode}{args.min_value:g} | ref tail q={args.tail_quantile:.2f} | "
            f"denom={args.return_level_denom} | decluster_runs={args.decluster_runs}",
            y=0.995,
        )
        rl_fig.tight_layout()
        rl_path = dataset_out / (
            f"top{args.top_k}_stability_return_level_tailq{int(round(args.tail_quantile * 100)):02d}_"
            f"stabq{int(round(stability_q_min * 100)):02d}to{int(round(stability_q_max * 100)):02d}_"
            f"decluster{int(args.decluster_runs)}_denom{args.return_level_denom}_filter{args.filter_mode}{args.min_value:g}_{args.split}.png"
        )
        rl_fig.savefig(rl_path, dpi=150)
        plt.close(rl_fig)

        # Mean residual life (mean excess) plots for top-K nodes
        mrl_fig, mrl_axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6.0 * ncols, 3.25 * nrows),
            squeeze=False,
        )

        mrl_rng = np.random.default_rng(int(args.mrl_bootstrap_seed))

        for row_idx, node in enumerate(top_nodes):
            for feat in range(ncols):
                ax = mrl_axes[row_idx][feat]
                ts_raw = np.asarray(y[:, node, feat], dtype=float)
                ts_raw = ts_raw[np.isfinite(ts_raw)]
                filtered = ts_raw[_mask(ts_raw, args.min_value, args.filter_mode)]

                feature_name = FEATURE_NAMES.get(feat, f"feature{feat}")
                title_prefix = f"node={int(node)} {feature_name}"

                if filtered.size < 2:
                    ax.text(0.5, 0.5, "skip:no_filtered_samples", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(title_prefix)
                    ax.axis("off")
                    continue

                mrl_qs = np.linspace(args.mrl_quantile_min, args.mrl_quantile_max, args.mrl_num_points)
                mrl_thresholds = np.quantile(filtered, mrl_qs)
                if args.decluster_runs > 0:
                    mean_excess = np.full((mrl_thresholds.size,), np.nan, dtype=float)
                    support = np.zeros((mrl_thresholds.size,), dtype=int)
                    declustered_exceedances: list[np.ndarray] = []
                    for j, u in enumerate(mrl_thresholds):
                        maxima = _runs_declustering_cluster_maxima(ts_raw, float(u), runs=int(args.decluster_runs))
                        z = maxima - float(u)
                        z = z[np.isfinite(z)]
                        z = z[z > 0]
                        declustered_exceedances.append(z)
                        support[j] = int(z.size)
                        if z.size > 0:
                            mean_excess[j] = float(np.mean(z))
                else:
                    mean_excess, support = _mrl_series(filtered, mrl_thresholds)
                    declustered_exceedances = []
                ci_band = None
                if args.mrl_bootstrap_n > 0:
                    if args.decluster_runs > 0:
                        ci_lo = np.full((mrl_qs.size,), np.nan, dtype=float)
                        ci_hi = np.full((mrl_qs.size,), np.nan, dtype=float)
                        for j, z in enumerate(declustered_exceedances):
                            lo, hi = _bootstrap_mean_ci(
                                z,
                                n_boot=int(args.mrl_bootstrap_n),
                                ci_level=float(args.mrl_ci_level),
                                rng=mrl_rng,
                            )
                            ci_lo[j] = lo
                            ci_hi[j] = hi
                        ci_band = (ci_lo, ci_hi)
                    else:
                        ci_lo, ci_hi = _bootstrap_mrl_ci(
                            filtered,
                            mrl_qs,
                            n_boot=int(args.mrl_bootstrap_n),
                            ci_level=float(args.mrl_ci_level),
                            rng=mrl_rng,
                            sample_size=args.mrl_bootstrap_sample_size,
                        )
                        ci_band = (ci_lo, ci_hi)

                _, fit = _fit_gpd_exceedances(
                    ts_raw,
                    min_value=args.min_value,
                    filter_mode=args.filter_mode,
                    tail_quantile=args.tail_quantile,
                    min_exceedances=args.min_exceedances,
                )

                if mean_excess.size < 2:
                    ax.text(0.5, 0.5, "skip:too_few_thresholds", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(title_prefix)
                    ax.axis("off")
                    continue

                _plot_mrl_with_support(
                    ax,
                    thresholds=mrl_thresholds,
                    mean_excess=mean_excess,
                    mean_excess_ci=ci_band,
                    support=support,
                    support_scale=args.mrl_support_scale,
                    support_ref=args.min_exceedances,
                    fit_threshold_u=fit.threshold_u,
                    support_label=("Support (# clusters)" if args.decluster_runs > 0 else "Support (# exceedances)"),
                )

                if fit.status == "ok":
                    assert fit.threshold_u is not None
                    assert fit.xi is not None
                    assert fit.sigma is not None
                    ax.set_title(
                        f"{title_prefix} | fit@q={args.tail_quantile:.2f} u={fit.threshold_u:.2f} "
                        f"n_exc={fit.n_exceed} ξ={fit.xi:.3f} σ={fit.sigma:.2f}"
                    )
                else:
                    ax.set_title(f"{title_prefix} | fit:{fit.status}")

                for q, u, me, sup in zip(mrl_qs, mrl_thresholds, mean_excess, support):
                    ci_lo_val = np.nan
                    ci_hi_val = np.nan
                    if ci_band is not None:
                        j = int(np.searchsorted(mrl_qs, q))
                        j = min(max(j, 0), len(mrl_qs) - 1)
                        ci_lo_val = float(ci_band[0][j])
                        ci_hi_val = float(ci_band[1][j])
                    mrl_records.append(
                        {
                            "dataset": dataset,
                            "split": args.split,
                            "node": int(node),
                            "feature": int(feat),
                            "feature_name": feature_name,
                            "min_value": float(args.min_value),
                            "filter_mode": args.filter_mode,
                            "tail_quantile": float(args.tail_quantile),
                            "mrl_quantile_min": float(args.mrl_quantile_min),
                            "mrl_quantile_max": float(args.mrl_quantile_max),
                            "mrl_num_points": int(args.mrl_num_points),
                            "decluster_runs": int(args.decluster_runs),
                            "mrl_quantile": float(q),
                            "mrl_percentile": float(q * 100.0),
                            "threshold_u": float(u),
                            "mean_excess": float(me) if np.isfinite(me) else np.nan,
                            "mean_excess_ci_lo": ci_lo_val,
                            "mean_excess_ci_hi": ci_hi_val,
                            "mrl_ci_level": float(args.mrl_ci_level) if args.mrl_bootstrap_n > 0 else np.nan,
                            "mrl_bootstrap_n": int(args.mrl_bootstrap_n),
                            "mrl_bootstrap_sample_size": (
                                int(args.mrl_bootstrap_sample_size) if args.mrl_bootstrap_sample_size is not None else np.nan
                            ),
                            "support": int(sup),
                            "fit_status": fit.status,
                            "fit_threshold_u": fit.threshold_u,
                            "fit_xi": fit.xi,
                            "fit_sigma": fit.sigma,
                        }
                    )

        mrl_fig.suptitle(
            f"{dataset} ({args.split}) top{args.top_k} MRL | "
            f"filter={args.filter_mode}{args.min_value:g} | tail q={args.tail_quantile:.2f} | "
            f"MRL qs={args.mrl_quantile_min:.2f}..{args.mrl_quantile_max:.2f} | "
            f"decluster_runs={args.decluster_runs} | "
            f"boot={args.mrl_bootstrap_n} ci={args.mrl_ci_level*100:.0f}%",
            y=0.995,
        )
        mrl_fig.tight_layout()
        mrl_path = dataset_out / (
            f"top{args.top_k}_mrl_tailq{int(round(args.tail_quantile * 100)):02d}_"
            f"mrlq{int(round(args.mrl_quantile_min * 100)):02d}to{int(round(args.mrl_quantile_max * 100)):02d}_"
            f"decluster{int(args.decluster_runs)}_"
            f"filter{args.filter_mode}{args.min_value:g}_{args.split}.png"
        )
        mrl_fig.savefig(mrl_path, dpi=150)
        plt.close(mrl_fig)

        # Dataset-wide fits for histogram(s)
        if args.histogram_scope == "topk":
            nodes_for_hist = top_nodes
        else:
            nodes_for_hist = np.arange(n, dtype=int)

        for node in nodes_for_hist:
            for feat in range(ncols):
                ts = y[:, node, feat]
                _, fit = _fit_gpd_exceedances(
                    ts,
                    min_value=args.min_value,
                    filter_mode=args.filter_mode,
                    tail_quantile=args.tail_quantile,
                    min_exceedances=args.min_exceedances,
                )
                feature_name = FEATURE_NAMES.get(feat, f"feature{feat}")
                global_fit_records.append(
                    {
                        "dataset": dataset,
                        "split": args.split,
                        "node": int(node),
                        "feature": int(feat),
                        "feature_name": feature_name,
                        "min_value": float(args.min_value),
                        "filter_mode": args.filter_mode,
                        "tail_quantile": float(args.tail_quantile),
                        "n_total": int(s),
                        "n_filtered": int(fit.n_filtered),
                        "threshold_u": fit.threshold_u,
                        "n_exceed": int(fit.n_exceed),
                        "xi": fit.xi,
                        "sigma": fit.sigma,
                        "status": fit.status,
                        "activity_score": float(activity[int(node)]),
                        "histogram_scope": args.histogram_scope,
                    }
                )

        df_hist = pd.DataFrame([r for r in global_fit_records if r["dataset"] == dataset])
        df_hist_ok = df_hist[df_hist["status"] == "ok"].copy()

        hist_fig, hist_axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        hist_fig.suptitle(
            f"{dataset} ({args.split}) GPD fit params | scope={args.histogram_scope} | "
            f"filter={args.filter_mode}{args.min_value:g} | tail q={args.tail_quantile:.2f}",
            y=0.98,
        )

        xi_ax, sigma_ax = hist_axes
        if df_hist_ok.empty:
            xi_ax.text(0.5, 0.5, "no valid fits", ha="center", va="center", transform=xi_ax.transAxes)
            sigma_ax.text(0.5, 0.5, "no valid fits", ha="center", va="center", transform=sigma_ax.transAxes)
        else:
            for feat in range(ncols):
                feature_name = FEATURE_NAMES.get(feat, f"feature{feat}")
                sub = df_hist_ok[df_hist_ok["feature"] == feat]
                if sub.empty:
                    continue
                xi_ax.hist(sub["xi"].astype(float), bins=args.hist_bins, alpha=0.6, label=feature_name)
                sigma_ax.hist(sub["sigma"].astype(float), bins=args.hist_bins, alpha=0.6, label=feature_name)

            xi_ax.set_title(f"xi histogram (n={len(df_hist_ok)})")
            xi_ax.set_xlabel("xi")
            xi_ax.set_ylabel("count")
            xi_ax.grid(True, alpha=0.25)
            xi_ax.legend()

            sigma_ax.set_title(f"sigma histogram (n={len(df_hist_ok)})")
            sigma_ax.set_xlabel("sigma")
            sigma_ax.set_ylabel("count")
            sigma_ax.grid(True, alpha=0.25)
            sigma_ax.legend()

        hist_fig.tight_layout()
        hist_path = dataset_out / (
            f"xi_sigma_hist_tailq{int(round(args.tail_quantile * 100)):02d}_"
            f"filter{args.filter_mode}{args.min_value:g}_{args.split}_{args.histogram_scope}.png"
        )
        hist_fig.savefig(hist_path, dpi=150)
        plt.close(hist_fig)

    df = pd.DataFrame(all_records)
    df.sort_values(["dataset", "node", "feature"], inplace=True)
    df.to_csv(out_dir / "gpd_node_qq_summary.csv", index=False)
    print(f"Wrote: {out_dir / 'gpd_node_qq_summary.csv'}")

    df_global = pd.DataFrame(global_fit_records)
    if not df_global.empty:
        df_global.sort_values(["dataset", "node", "feature"], inplace=True)
        df_global.to_csv(out_dir / "gpd_node_fit_params.csv", index=False)
        print(f"Wrote: {out_dir / 'gpd_node_fit_params.csv'}")

    df_mrl = pd.DataFrame(mrl_records)
    if not df_mrl.empty:
        df_mrl.sort_values(["dataset", "node", "feature", "mrl_quantile"], inplace=True)
        df_mrl.to_csv(out_dir / "gpd_node_mrl_points.csv", index=False)
        print(f"Wrote: {out_dir / 'gpd_node_mrl_points.csv'}")

    df_stab = pd.DataFrame(stability_records)
    if not df_stab.empty:
        df_stab.sort_values(["dataset", "node", "feature", "stability_quantile"], inplace=True)
        df_stab.to_csv(out_dir / "gpd_node_stability_points.csv", index=False)
        print(f"Wrote: {out_dir / 'gpd_node_stability_points.csv'}")

    df_rl = pd.DataFrame(return_level_records)
    if not df_rl.empty:
        df_rl.sort_values(["dataset", "node", "feature", "return_period", "stability_quantile"], inplace=True)
        df_rl.to_csv(out_dir / "gpd_node_return_level_points.csv", index=False)
        print(f"Wrote: {out_dir / 'gpd_node_return_level_points.csv'}")

    print(f"Wrote plots under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
