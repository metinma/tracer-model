import csv
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

INJECTOR = "a1"
PRODUCERS = ["b", "c", "d", "e", "f"]

# Project Table 1: distance and pressure difference from A1 to each well
WELL_PARAMS = {
    "b": {"dist_m": 750.0, "delta_p_mpa": 7.0},
    "f": {"dist_m": 1000.0, "delta_p_mpa": 8.0},
    "e": {"dist_m": 1400.0, "delta_p_mpa": 9.5},
    "d": {"dist_m": 1500.0, "delta_p_mpa": 10.0},
    "c": {"dist_m": 1600.0, "delta_p_mpa": 11.0},
}

# Assumed fluid and rock properties (not in the data)
MU = 1.5e-4  # Pa s, water viscosity at ~180 C
D_M = 2e-9  # m2/s, molecular diffusion coefficient of the tracer in hot water
PHI_M = 0.005  # matrix porosity (project Table 2)
AT_RATIO = 0.2  # alpha_T / alpha_L (project Table 2)
DAY = 86400.0  # s

PLOT_T_END_DAYS = 1000.0

MODEL_STYLES = {
    "sauty_uniform": ("blue", "-"),
    "sauty_radial": ("red", "-"),
    "bullivant_single_fracture": ("green", "--"),
    "bullivant_fracture_matrix": ("orange", "-."),
    "houseworth_fracture_matrix": ("purple", ":"),
}


def main():
    tracer_data = import_tracer_data("./tracer_data")
    coordinates = import_coordinates("./coordinates")

    plot_3d_spatial_data(coordinates)
    plot_tracer_data(tracer_data)

    rows = []
    for well in PRODUCERS:
        t_obs, C_raw = tracer_data[well]
        C_bg = estimate_background(C_raw)
        C_obs = np.clip(C_raw - C_bg, 0.0, None)
        fits = fit_analytical_models(t_obs, C_obs)
        plot_fitted_curves(t_obs, C_obs, fits, well, C_bg)
        for fit in fits:
            rows.append({"well": well, "model": fit["model"], **summary_row(fit, well)})

    print_table(rows)
    write_csv(rows, "./fits/summary.csv")
    print_permeability_tensors(rows, well_directions(coordinates))


# --- Analytical models (t in days) ------------------------------------------


def _positive(t):
    t = np.asarray(t, dtype=float)
    return np.where(t > 0, t, np.nan)


def sauty_uniform(t, t_c, Pe, C0):
    """Sauty (1980) Eq. 42-44: slug injection, 2-D uniform flow, on the flow axis."""
    t_R = _positive(t) / t_c
    t_Rmax = np.sqrt(1 + 4 / Pe**2) - 2 / Pe  # Eq. 44
    K = t_Rmax * np.exp(Pe * (1 - t_Rmax) ** 2 / (4 * t_Rmax))  # Eq. 43
    return np.nan_to_num(C0 * K / t_R * np.exp(-Pe * (1 - t_R) ** 2 / (4 * t_R)))


def sauty_radial(t, t_c, Pe, C0):
    """Sauty (1980) Eq. 52: diverging radial flow from the injector.

    Eq. 52 is for continuous injection; the slug response is its time
    derivative, C ~ t_R^-1.75 (0.75 + 0.25 t_R) exp(-g^2).
    """
    t_R = _positive(t) / t_c
    g = np.sqrt(3 * Pe) * (1 - t_R) / (4 * t_R**0.75)
    return np.nan_to_num(C0 * t_R**-1.75 * (0.75 + 0.25 * t_R) * np.exp(-(g**2)))


def bullivant_single_fracture(t, t_m, w, L):
    """Bullivant & O'Sullivan (1989) Eq. 8: single fracture, Taylor dispersion."""
    t = _positive(t)
    return np.nan_to_num(L * 2 * t_m / t**1.5 * np.exp(-w * (t - t_m) ** 2 / (4 * t_m * t)))


def bullivant_fracture_matrix(t, t_b, w, L):
    """Bullivant & O'Sullivan (1989) Eq. 17: fracture with matrix diffusion."""
    dt = _positive(np.asarray(t, dtype=float) - t_b)
    return np.nan_to_num(L / dt**1.5 * np.exp(-t_b / (w * dt)))


def houseworth_fracture_matrix(t, t_a, beta, gamma, M):
    """Houseworth et al. (2013) Eq. 57, release in the fracture (eta_0 = 0).

    With no matrix flow along the fracture and theta = (t - t_a)/t_a, Eq. 57
    becomes M theta^-1.5 exp(-beta/(4 theta) - gamma theta/4), where
    beta = xi/Pe (matrix diffusion) and gamma = Pe V^2 xi (cross-flow through
    the matrix). With gamma = 0 it is Bullivant's Eq. 17.
    """
    theta = _positive(np.asarray(t, dtype=float) - t_a) / t_a
    return np.nan_to_num(M * theta**-1.5 * np.exp(-beta / (4 * theta) - gamma * theta / 4))


# --- Data preparation --------------------------------------------------------


def estimate_background(C_obs):
    """Background level of the measured data, outside the tracer plume.

    Taken as the lower of the median of the first and last 10% of samples, so
    that a record ending near its peak, or starting with residual tracer, does
    not inflate it.
    """
    n = max(5, len(C_obs) // 10)
    return min(np.median(C_obs[:n]), np.median(C_obs[-n:]))


def robust_peak(t_obs, C_obs):
    smooth = median_filter(C_obs, size=5, mode="nearest")
    i = np.argmax(smooth)
    return t_obs[i], smooth[i]


def first_arrival(t_obs, C_obs, frac=0.05):
    """Last time before the peak at which the smoothed data are below frac * peak.

    A wide median window keeps isolated low or high samples from moving it.
    """
    t_peak, C_peak = robust_peak(t_obs, C_obs)
    smooth = median_filter(C_obs, size=21, mode="nearest")
    before = (t_obs < t_peak) & (smooth < frac * C_peak)
    return t_obs[before].max() if before.any() else t_obs[0]


def sampling_weights(t_obs):
    dt = np.gradient(t_obs) if len(t_obs) > 1 else np.ones_like(t_obs)
    dt = np.maximum(dt, 1e-3 * np.median(dt))  # guard against repeated times
    return 1 / np.sqrt(dt / np.mean(dt))


# --- Fitting -----------------------------------------------------------------


def fit_analytical_models(t_obs, C_obs):
    """Fit every model. Times of the fracture-matrix models are fixed.

    In both fracture-matrix models no tracer arrives before the advective time
    t_b (t_a), so it is fixed at the observed first arrival. Left free it slides
    to ~0 and both models turn into the single-fracture curve.
    """
    t_peak, _ = robust_peak(t_obs, C_obs)
    t_arr = first_arrival(t_obs, C_obs)
    T = 10 * np.max(t_obs)

    sauty_starts = [[f * t_peak, pe] for f in (1.0, 1.5) for pe in (3.0, 10.0, 30.0)]
    models = {
        # name: (function, fixed parameters, starts, lower, upper) - amplitude excluded
        "sauty_uniform": (sauty_uniform, {}, sauty_starts, [1, 1], [T, 1e4]),
        "sauty_radial": (sauty_radial, {}, sauty_starts, [1, 1], [T, 1e4]),
        "bullivant_single_fracture": (
            bullivant_single_fracture,
            {},
            [[f * t_peak, 6 * f / (f**2 - 1)] for f in (1.05, 1.2, 1.5, 2.0)],  # Eq. 9
            [1, 0.1],
            [T, 1e4],
        ),
        "bullivant_fracture_matrix": (
            bullivant_fracture_matrix,
            {"t_b": t_arr},
            [[2 / 3 * t_arr / (t_peak - t_arr)]],  # Eq. 18
            [1e-4],
            [1e4],
        ),
        "houseworth_fracture_matrix": (
            houseworth_fracture_matrix,
            {"t_a": t_arr},
            [[6 * (t_peak - t_arr) / t_arr, g] for g in (1e-3, 0.1, 1.0, 10.0, 100.0)],
            [1e-4, 1e-6],
            [1e4, 1e3],
        ),
    }
    return [
        fit_model(name, *config, t_obs, C_obs, t_peak)
        for name, config in models.items()
    ]


def fit_model(name, func, fixed, starts, lower, upper, t_obs, C_obs, t_peak):
    """Weighted least squares in log-parameters, best of several starts."""
    free = [n for n in list(inspect.signature(func).parameters)[1:] if n not in fixed]
    sigma = sampling_weights(t_obs)

    def model(t, *q):
        return func(t, **fixed, **dict(zip(free, np.exp(q))))

    best = None
    for start in starts:
        shape = func(np.array([t_peak]), **fixed, **dict(zip(free, start + [1.0])))[0]
        amp0 = np.max(C_obs) / shape if shape > 0 else 1.0
        lo = np.log(lower + [amp0 * 1e-6])
        hi = np.log(upper + [amp0 * 1e6])
        q0 = np.clip(np.log(start + [amp0]), lo + 1e-6, hi - 1e-6)
        try:
            q, cov = curve_fit(
                model, t_obs, C_obs, p0=q0, sigma=sigma, bounds=(lo, hi), maxfev=20000
            )
        except RuntimeError:
            continue  # this start did not converge
        cost = np.sum((C_obs - model(t_obs, *q)) ** 2 / sigma**2)
        if best is None or cost < best[0]:
            best = (cost, q, cov, lo, hi)

    cost, q, cov, lo, hi = best
    with np.errstate(over="ignore"):
        factor = np.exp(np.sqrt(np.diag(cov)))
    params = {**fixed, **dict(zip(free, np.exp(q)))}
    return {
        "model": name,
        "predict": lambda t: func(t, **params),
        "params": params,
        # multiplicative 1-sigma uncertainty of each fitted parameter
        "factor": dict(zip(free, factor)),
        "rmse": np.sqrt(cost / np.sum(1 / sigma**2)),
        "at_bound": [n for n, a, b, c in zip(free, q, lo, hi) if a - b < 1e-3 or c - a < 1e-3],
    }


# --- Physical parameters -----------------------------------------------------


def summary_row(fit, well):
    """Velocity, dispersivity, aperture and permeability from the fitted parameters."""
    R = WELL_PARAMS[well]["dist_m"]
    dP = WELL_PARAMS[well]["delta_p_mpa"] * 1e6
    p, name = fit["params"], fit["model"]
    row = {"rmse_ppb": fit["rmse"], "at_bound": " ".join(fit["at_bound"])}

    if name.startswith("sauty"):
        t_adv, t_key = p["t_c"], "t_c"
        row["alpha_L_m"] = R / p["Pe"]
        row["alpha_T_m"] = AT_RATIO * row["alpha_L_m"]
    elif name == "bullivant_single_fracture":
        t_adv, t_key = p["t_m"], "t_m"
        u = R / (t_adv * DAY)
        D_Tr = R**2 / (p["w"] * t_adv * DAY)  # Eq. 11
        row["alpha_L_m"] = R / p["w"]
        row["aperture_m"] = 2 / u * np.sqrt(105 / 2 * D_Tr * D_M)  # Eq. 4
    elif name == "bullivant_fracture_matrix":
        t_adv, t_key = p["t_b"], "t_b"
        # w = (q/RA) a^2/(D_m n_m^2) and q/RA = 1/t_b (Eq. 17)
        row["aperture_m"] = PHI_M * np.sqrt(p["w"] * t_adv * DAY * D_M)
    else:
        t_adv, t_key = p["t_a"], "t_a"
        # beta = xi/Pe = t_a D_m (2 phi_m / b)^2, gamma = t_a v_mx^2 / D_m
        row["aperture_m"] = 2 * PHI_M * np.sqrt(t_adv * DAY * D_M / p["beta"])
        row["v_matrix_m_day"] = np.sqrt(p["gamma"] * D_M / (t_adv * DAY)) * DAY

    v = R / (t_adv * DAY)  # m/s
    k = MU * v * R / dP  # Darcy along the well pair
    row["t_adv_day"] = t_adv
    row["t_adv_factor"] = fit["factor"].get(t_key, 1.0)  # 1.0: fixed at first arrival
    row["v_m_day"] = v * DAY
    row["k_m2"] = k
    row["h_cubic_m"] = np.sqrt(12 * k)  # cubic law, k = h^2/12
    return row


def well_directions(coordinates):
    """Horizontal unit vector from the injector bottom to each producer bottom."""
    inj = coordinates[INJECTOR][-1]
    dirs = {}
    for well in PRODUCERS:
        d = coordinates[well][-1][:2] - inj[:2]
        dirs[well] = d / np.linalg.norm(d)
    return dirs


def permeability_tensor(k_by_well, dirs):
    """Least-squares 2-D tensor from directional values k_n = n.K.n."""
    A = [[n[0] ** 2, 2 * n[0] * n[1], n[1] ** 2] for n in (dirs[w] for w in k_by_well)]
    kxx, kxy, kyy = np.linalg.lstsq(np.array(A), np.array(list(k_by_well.values())), rcond=None)[0]
    eigval, eigvec = np.linalg.eigh([[kxx, kxy], [kxy, kyy]])
    azimuth = np.degrees(np.arctan2(eigvec[0, 1], eigvec[1, 1])) % 180  # from north (+y)
    return eigval[1], eigval[0], azimuth


# --- Output ------------------------------------------------------------------


def print_table(rows):
    print(
        f"{'well':<5}{'model':<28}{'RMSE':>7}{'t_adv(d)':>13}{'v(m/d)':>8}"
        f"{'aL(m)':>8}{'aperture':>10}{'k(m2)':>10}{'h_cubic':>9}  at bound"
    )
    for r in rows:
        print(
            f"{r['well'].upper():<5}{r['model']:<28}{r['rmse_ppb']:>7.3f}"
            f"{r['t_adv_day']:>7.0f} x{r['t_adv_factor']:<5.3g}{r['v_m_day']:>8.2f}"
            f"{r.get('alpha_L_m', np.nan):>8.1f}{r.get('aperture_m', np.nan):>10.2e}"
            f"{r['k_m2']:>10.2e}{r['h_cubic_m']:>9.2e}  {r['at_bound']}"
        )


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_permeability_tensors(rows, dirs):
    print("\nHorizontal permeability tensor (from k along each well pair)")
    for model in MODEL_STYLES:
        k = {r["well"]: r["k_m2"] for r in rows if r["model"] == model}
        k_max, k_min, azimuth = permeability_tensor(k, dirs)
        print(
            f"  {model:<28} k_max={k_max:.2e}  k_min={k_min:.2e}  "
            f"ratio={k_max / k_min:.1f}  k_max azimuth={azimuth:.0f} deg"
        )


def import_tracer_data(data_dir):
    tracer_data = {}
    for file in sorted(Path(data_dir).glob("*.txt")):
        C, t = np.loadtxt(file, skiprows=2, unpack=True)
        keep = np.isfinite(C) & np.isfinite(t) & (C != 0)
        C, t = C[keep], t[keep]
        order = np.argsort(t, kind="stable")
        tracer_data[file.stem] = (t[order], C[order])
    return tracer_data


def import_coordinates(data_dir):
    return {
        file.stem: np.loadtxt(file, skiprows=1, usecols=(0, 1, 2), ndmin=2)
        for file in sorted(Path(data_dir).glob("*.txt"))
    }


def plot_3d_spatial_data(coordinates):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for name, xyz in coordinates.items():
        x, y, z = xyz.T
        if name == "fault":
            ax.scatter(x, y, z, color="red", s=2, alpha=0.4, label=name)
        else:
            ax.plot(x, y, z, linewidth=2, label=name)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Well Trajectories and Fault Geometry")

    plt.legend()
    plt.savefig("field_overview.jpg", dpi=300)
    plt.close()


def plot_tracer_data(tracer_data):
    for well, (t, C) in tracer_data.items():
        plt.scatter(t, C, label=f"C_{well}", s=2)

    plt.xlabel("Time (days)")
    plt.ylabel("Tracer Concentration (ppb)")
    plt.title("Tracer Concentration vs Time")

    plt.legend()
    plt.savefig("tracer_data.jpg", dpi=300)
    plt.close()


def plot_fitted_curves(t_obs, C_obs, fits, well, C_bg):
    _, C_max = robust_peak(t_obs, C_obs)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        t_obs,
        C_obs / C_max,
        color="black",
        s=15,
        alpha=0.7,
        label=f"Measured Field Data (background {C_bg:.3g} ppb removed)",
    )

    t_end = max(PLOT_T_END_DAYS, np.max(t_obs))
    t_smooth = np.linspace(0, t_end, 1000)

    for fit in fits:
        color, linestyle = MODEL_STYLES[fit["model"]]
        plt.plot(
            t_smooth,
            fit["predict"](t_smooth) / C_max,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"{fit['model']} (RMSE: {fit['rmse'] / C_max:.3f})",
        )

    plt.xlabel("Time (days)")
    plt.ylabel(f"$C/C_{{max}}$  ($C_{{max}}$ = {C_max:.3g} ppb)")
    plt.title(f"Analytical Model Fits - Well {well.upper()}")
    plt.xlim(0, t_end)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.savefig(f"./fits/fit_{well}.jpg", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
