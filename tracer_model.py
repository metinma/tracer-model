import csv
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

INJECTOR = "a1"
PRODUCERS = ["b", "c", "d", "e", "f"]

# Table 1 of the project description: injector to producer Δz and ΔP.
# Kept for the permeability calculation (not used yet).
WELL_PARAMS = {
    "b": {"dist_m": 750.0, "delta_p_mpa": 7.0},
    "f": {"dist_m": 1000.0, "delta_p_mpa": 8.0},
    "e": {"dist_m": 1400.0, "delta_p_mpa": 9.5},
    "d": {"dist_m": 1500.0, "delta_p_mpa": 10.0},
    "c": {"dist_m": 1600.0, "delta_p_mpa": 11.0},
}

# Model curves on the fitted-curve plots extend to this time
PLOT_T_END_DAYS = 1000.0

# Plot style per model: (color, linestyle)
MODEL_STYLES = {
    "sauty_uniform": ("blue", "-"),
    "bullivant_single_fracture": ("green", "--"),
    "bullivant_fracture_matrix": ("orange", "-."),
    "houseworth_fracture_matrix": ("purple", ":"),
}


def main():
    tracer_data = import_tracer_data("./tracer_data")
    coordinates = import_coordinates("./coordinates")

    plot_3d_spatial_data(coordinates)
    plot_tracer_data(tracer_data)

    print("Well distances from injector (m)")
    for well, dist in compute_well_distances(coordinates).items():
        print(f"  {well}: {dist:.1f}")

    all_fits = []
    for well in PRODUCERS:
        t_obs, C_obs = tracer_data[well]
        fits = fit_analytical_models(t_obs, C_obs)
        plot_fitted_curves(t_obs, C_obs, fits, well)
        all_fits += [dict(fit, well=well) for fit in fits]

    print_fit_summary(all_fits)
    save_fit_summary(all_fits, "fit_summary.csv")


# ---------------------------------------------------------------- models
# Each model takes time t (days) as an array and returns concentration (ppb).
# Times outside a model's valid range return zero concentration.
#
# Equation numbers refer to the original papers. They differ from the
# numbers in the project description (e.g. its "Eq. 30" for Houseworth is
# the paper's Eq. 57).


def sauty_uniform(t, t_c, Pe, C_max):
    """Sauty (1980) Eq. 42-44: slug injection, 2D uniform flow, on the flow axis.

    t_c = X/u: convective travel time (days)
    Pe = X/alpha_L: longitudinal Peclet number
    C_max: peak concentration (ppb)
    """
    t = np.asarray(t, dtype=float)
    t_R = np.where(t > 0, t, np.nan) / t_c  # dimensionless time

    t_Rmax = np.sqrt(1 + 4 * Pe ** (-2)) - 2 / Pe  # Eq. 44
    K = t_Rmax * np.exp(Pe * (1 - t_Rmax) ** 2 / (4 * t_Rmax))  # Eq. 43

    C_R = K / t_R * np.exp(-Pe * (1 - t_R) ** 2 / (4 * t_R))  # Eq. 42
    return np.nan_to_num(C_R * C_max)


def bullivant_single_fracture(t, t_m, w, L):
    """Bullivant & O'Sullivan (1989) Eq. 8: single fracture.

    t_m = RA/q: mean arrival time (days)
    w = qR/(D_Tr A): Peclet number
    L = (m/4Q)(w/(pi t_m))^0.5: amplitude (ppb day^0.5)
    """
    t = np.asarray(t, dtype=float)
    t_v = np.where(t > 0, t, np.nan)

    exponent = -w * (t_v - t_m) ** 2 / (4 * t_m * t_v)
    return np.nan_to_num(L * 2 * t_m / t_v**1.5 * np.exp(exponent))


def bullivant_fracture_matrix(t, t_b, w, L):
    """Bullivant & O'Sullivan (1989) Eq. 17: fracture with matrix diffusion.

    t_b = RA/q: advective breakthrough time (days)
    w = (q/RA)(a^2/(D_m n_m^2)): fracture-to-matrix transport ratio
    L = (m/Q)(t_b/(w pi))^0.5: amplitude (ppb day^1.5)
    Peak time from Eq. 18: t_p = t_b (1 + 2/(3w)).
    """
    t = np.asarray(t, dtype=float)
    dt = np.where(t > t_b, t - t_b, np.nan)  # time after breakthrough

    return np.nan_to_num(L / dt**1.5 * np.exp(-t_b / (w * dt)))


def houseworth_fracture_matrix(t, t_a, beta, gamma, M):
    """Houseworth et al. (2013) Eq. 57: fracture concentration, source in fracture.

    Eq. 57 with eta_0 = 0 and v_mz = 0, written in physical time using
    tau - xi = xi * theta, where theta = (t - t_a)/t_a:

        C = M sqrt(beta/pi)/4 * theta^-1.5 * exp(-beta/(4 theta) - gamma theta/4)

    t_a = z/v_f*: advective arrival time (days)
    beta = xi/Pe = t_a D_m*/l^2: matrix-diffusion strength (dimensionless)
    gamma = Pe xi V^2: loss by matrix cross-flow (dimensionless)
    M = 2 M0/(A_f phi_f R_f z): amplitude (ppb)
    With gamma = 0 this is the same curve as bullivant_fracture_matrix
    with w = 4/beta, so gamma measures what the matrix flow adds.
    """
    t = np.asarray(t, dtype=float)
    theta = np.where(t > t_a, (t - t_a) / t_a, np.nan)

    exponent = -beta / (4 * theta) - gamma * theta / 4
    return np.nan_to_num(M * np.sqrt(beta / np.pi) / 4 / theta**1.5 * np.exp(exponent))


# --------------------------------------------------------------- fitting


def robust_peak(t_obs, C_obs, window=5):
    """Time and value of the peak of a running median, so single outliers
    do not set the initial guesses or the C_max used to normalize plots."""
    if len(C_obs) < window:
        i = np.argmax(C_obs)
        return t_obs[i], C_obs[i]
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(C_obs, window), axis=1)
    i = np.argmax(smooth) + window // 2
    return t_obs[i], smooth[i - window // 2]


def sampling_weights(t_obs):
    """Curve_fit sigma that makes every day of record count equally.

    Readings are much denser on some periods (e.g. 291 of well B's 398
    readings fall between days 70 and 190). Weighting each reading by its
    sampling interval stops dense periods from dominating the fit.
    """
    dt = np.gradient(t_obs) if len(t_obs) > 1 else np.ones_like(t_obs)
    dt = np.maximum(dt, 1e-3 * np.median(dt))  # guard against repeated times
    return 1 / np.sqrt(dt / np.mean(dt))


def _fm_to_model(t_b, dt_p, L):
    """Bullivant fracture-matrix: fitted (t_b, t_p - t_b, L) -> (t_b, w, L).

    Eq. 18 gives w = (2/3) t_b / (t_p - t_b). Fitting the peak delay
    dt_p = t_p - t_b instead of w keeps t_p > t_b and bounds the peak time.
    """
    return t_b, (2 / 3) * t_b / dt_p, L


def fit_analytical_models(t_obs, C_obs):
    """Fit every model plus a constant background C_bg to one well.

    Returns a list of result dicts. Model parameters are fitted as log
    values (all are positive and span several orders of magnitude); C_bg is
    fitted linearly, between 0 and the median of the earliest 10% of
    readings, since the background cannot exceed what was measured before
    the tracer arrived. Readings are weighted by their sampling interval.

    Each model is fitted from several starting points and the best fit is
    kept, so the result does not depend on one initial guess. Amplitude
    guesses are scaled so each initial curve passes through the observed peak.
    """
    t_peak, C_peak = robust_peak(t_obs, C_obs)
    t_max = np.max(t_obs)
    sigma = sampling_weights(t_obs)
    weights = 1 / sigma**2

    # Background: capped at the early (pre-arrival) level of the readings
    n_early = max(5, len(C_obs) // 10)
    C_bg_max = min(np.median(C_obs[:n_early]), C_peak)
    C_bg0 = 0.5 * C_bg_max

    # Starting points, each consistent with a peak near t_peak
    arrival_fracs = [0.05, 0.2, 0.4, 0.6, 0.8, 0.9]
    sauty_starts = [[f * t_peak, pe] for f in (1.0, 1.5) for pe in (3.0, 10.0, 30.0)]
    sf_starts = [
        [f * t_peak, 6 * f / (f**2 - 1)]  # B&O Eq. 9 with t_m = f t_peak
        for f in (1.05, 1.2, 1.5, 2.0)
    ]
    fm_starts = [[f * t_peak, (1 - f) * t_peak] for f in arrival_fracs]
    hw_starts = [
        [f * t_peak, 6 * (1 - f) / f, gamma]  # peak at theta = beta/6
        for f in arrival_fracs for gamma in (1e-3, 0.1)
    ]

    # model name: (function, fitted parameter names, starting points,
    #              lower, upper, map from fitted to model parameters)
    # The last fitted parameter is the amplitude; its guess and bounds are
    # set below from the observed peak.
    models_config = {
        "sauty_uniform": (
            sauty_uniform, ["t_c", "Pe", "C_max"], sauty_starts,
            [0.01, 0.01], [10 * t_max, 1e4], None,
        ),
        "bullivant_single_fracture": (
            bullivant_single_fracture, ["t_m", "w", "L"], sf_starts,
            [0.01, 0.01], [10 * t_max, 1e4], None,
        ),
        "bullivant_fracture_matrix": (
            bullivant_fracture_matrix, ["t_b", "dt_p", "L"], fm_starts,
            [0.01, 0.01], [t_max, 2 * t_max], _fm_to_model,
        ),
        "houseworth_fracture_matrix": (
            houseworth_fracture_matrix, ["t_a", "beta", "gamma", "M"], hw_starts,
            [0.01, 1e-4, 1e-8], [t_max, 1e4, 1e4], None,
        ),
    }

    results = []
    for name, (func, names, starts, lower, upper, to_model) in models_config.items():
        to_model = to_model or (lambda *p: p)

        def fit_func(t, *q, func=func, to_model=to_model):
            return func(t, *to_model(*np.exp(q[:-1]))) + q[-1]

        best = None
        for shape0 in starts:
            # Amplitude guess: make the initial curve pass through the peak
            shape_at_peak = func(np.array([t_peak]), *to_model(*shape0, 1.0))[0]
            amp0 = (C_peak - C_bg0) / shape_at_peak if shape_at_peak > 0 else C_peak

            q_lower = np.append(np.log(lower + [amp0 * 1e-8]), 0.0)
            q_upper = np.append(np.log(upper + [amp0 * 1e8]), C_bg_max)
            q0 = np.append(np.log(shape0 + [amp0]), C_bg0)
            q0 = np.clip(q0, q_lower + 1e-6, q_upper - 1e-6)
            try:
                q_opt, _ = curve_fit(
                    fit_func, t_obs, C_obs, p0=q0, sigma=sigma,
                    bounds=(q_lower, q_upper), method="trf", maxfev=20000,
                )
            except RuntimeError:
                continue  # this start did not converge
            cost = np.sum(weights * (C_obs - fit_func(t_obs, *q_opt)) ** 2)
            if best is None or cost < best[0]:
                best = (cost, q_opt, q_lower, q_upper)

        cost, q_opt, q_lower, q_upper = best
        fitted, C_bg = np.exp(q_opt[:-1]), q_opt[-1]
        model_params = to_model(*fitted)

        # Fitted parameters within 0.1% of a bound are not really estimated.
        # C_bg = 0 is a valid result (no background); C_bg at its cap is flagged.
        at_bound = [
            p for p, q, lo, hi in zip(names, q_opt, q_lower, q_upper)
            if q - lo < 1e-3 or hi - q < 1e-3
        ]
        if C_bg_max - C_bg < 1e-3 * C_bg_max:
            at_bound.append("C_bg")

        def predict(t, func=func, model_params=model_params, C_bg=C_bg):
            return func(t, *model_params) + C_bg

        # Report the fitted parameters plus any model parameters derived
        # from them (e.g. w for the fracture-matrix model)
        params = dict(zip(names, fitted))
        model_names = list(inspect.signature(func).parameters)[1:]  # skip t
        params.update(
            (k, v) for k, v in zip(model_names, model_params) if k not in params
        )
        params["C_bg"] = C_bg
        results.append(
            {
                "model": name,
                "predict": predict,
                "params": params,
                "rmse": np.sqrt(cost / np.sum(weights)),  # time-weighted
                "at_bound": at_bound,
            }
        )
    return results


def print_fit_summary(all_fits):
    print("\nFit summary  (RMSE is time-weighted, in ppb; "
          "* = parameter stopped at a bound, not estimated)")
    for fit in all_fits:
        params = ", ".join(
            f"{k}={v:.4g}{'*' if k in fit['at_bound'] else ''}"
            for k, v in fit["params"].items()
        )
        print(f"  {fit['well'].upper()}  {fit['model']:<28} "
              f"RMSE={fit['rmse']:.3f}  {params}")


def save_fit_summary(all_fits, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["well", "model", "rmse", "parameters", "at_bound"])
        for fit in all_fits:
            params = "; ".join(f"{k}={v:.6g}" for k, v in fit["params"].items())
            writer.writerow([fit["well"], fit["model"], f"{fit['rmse']:.6g}",
                             params, "; ".join(fit["at_bound"])])


# -------------------------------------------------------------- data I/O


def import_tracer_data(data_dir):
    """Return {well: (time_days, concentration_ppb)}, sorted by time.

    Files have a '%' title line, a 'ppb days' header, then two columns.
    Zero readings are an artifact of the measurement tool and are dropped,
    along with any missing (NaN) values.
    """
    tracer_data = {}
    for file in sorted(Path(data_dir).glob("*.txt")):
        C, t = np.loadtxt(file, skiprows=2, unpack=True)
        keep = np.isfinite(C) & np.isfinite(t) & (C != 0)
        C, t = C[keep], t[keep]
        order = np.argsort(t, kind="stable")
        tracer_data[file.stem] = (t[order], C[order])
    return tracer_data


def import_coordinates(data_dir):
    """Return {name: array of shape (n, 3)} with x, y, z columns."""
    return {
        file.stem: np.loadtxt(file, skiprows=1, usecols=(0, 1, 2), ndmin=2)
        for file in sorted(Path(data_dir).glob("*.txt"))
    }


def compute_well_distances(coordinates):
    """Straight-line distance from the injector bottom to each producer bottom."""
    inj_bottom = coordinates[INJECTOR][-1]
    return {
        well: np.linalg.norm(coordinates[well][-1] - inj_bottom) for well in PRODUCERS
    }


# -------------------------------------------------------------- plotting


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


def plot_fitted_curves(t_obs, C_obs, fits, well):
    """Plot measured and fitted curves as C/C_max against time.

    Measurements and all models are divided by the same C_max (the
    running-median peak of the measurements), so the fitted curves still
    overlay the data and RMSE is shown as a fraction of C_max.
    """
    _, C_max = robust_peak(t_obs, C_obs)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        t_obs, C_obs / C_max, color="black", s=15, alpha=0.7, label="Measured Field Data"
    )

    # Smooth time grid for the analytical curves, extended past the data
    # to show each model's predicted peak and tail
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
    plt.ylim(0, 1.2)  # measurements far above C_max are clipped
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.savefig(f"./fitted_curves/fitted_curves_{well}.jpg", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
