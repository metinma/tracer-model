import csv
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

# Plot style per model: (color, linestyle)
MODEL_STYLES = {
    "sauty_uniform_radial": ("blue", "-"),
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


def sauty_uniform_radial(t, t_c, Pe, C_max):
    """Sauty (1980) 1D advection-dispersion, scaled so the peak equals C_max.

    t_c: convective (mean travel) time, Pe: Peclet number.
    """
    t = np.asarray(t, dtype=float)
    t_R = np.where(t > 0, t, np.nan) / t_c  # dimensionless time

    # Dimensionless time of peak concentration and the scaling factor
    t_Rmax = np.sqrt(1 + Pe ** (-2)) - 1 / Pe
    K = np.sqrt(t_Rmax) * np.exp(Pe / (4 * t_Rmax) * (1 - t_Rmax) ** 2)

    C_R = K / np.sqrt(t_R) * np.exp(-Pe / (4 * t_R) * (1 - t_R) ** 2)
    return np.nan_to_num(C_R * C_max)


def bullivant_single_fracture(t, t_b, t_p, L):
    """Bullivant & O'Sullivan (1989) single fracture model.

    t_b: breakthrough time, t_p: peak time, L: amplitude.
    """
    if t_b >= t_p:
        raise ValueError("Breakthrough time must be less than peak time.")

    t = np.asarray(t, dtype=float)
    t_v = np.where(t > 0, t, np.nan)

    t_m = 2 * t_p - t_b  # mean arrival time
    w = 6 * t_m * t_p / (t_m**2 - t_p**2)  # Peclet number

    exponent = -w / 4 * (t_v - t_m) ** 2 / t_m / t_v
    return np.nan_to_num(L * 2 * t_m / t_v**1.5 * np.exp(exponent))


def bullivant_fracture_matrix(t, t_p, t_b, L):
    """Bullivant & O'Sullivan (1989) fracture-matrix model.

    t_p: peak time, t_b: advective breakthrough time, L: amplitude.
    """
    t = np.asarray(t, dtype=float)
    if t_b >= t_p:
        return np.full_like(t, 1e9)

    dt = np.where(t > t_b, t - t_b, np.nan)  # time after breakthrough
    w = (2 / 3) * t_b / (t_p - t_b)  # Peclet number

    return np.nan_to_num(L / dt**1.5 * np.exp(-t_b / (w * dt)))


def houseworth_fracture_matrix(t, t_a, Pe, V_m, A, eta_0=0.0):
    """Houseworth et al. (2013) fracture-matrix model.

    t_a: advective arrival time, Pe: Peclet number,
    V_m: cross-flow term, A: amplitude, eta_0: spatial offset.
    """
    t = np.asarray(t, dtype=float)
    dt = np.where(t > t_a, t - t_a, np.nan)  # time after arrival
    offset = 1.0 + eta_0

    diffusive_term = -Pe * offset**2 / (4.0 * dt)
    crossflow_term = -Pe * V_m**2 * dt / 4.0

    return np.nan_to_num(A * offset / dt**1.5 * np.exp(diffusive_term + crossflow_term))


# --------------------------------------------------------------- fitting


def fit_analytical_models(t_obs, C_obs):
    """Fit every model to one well. Returns a list of result dicts."""
    # Initial guesses from the observed peak
    peak_idx = np.argmax(C_obs)
    t_peak = t_obs[peak_idx] if t_obs[peak_idx] > 0 else 1.0
    C_peak = C_obs[peak_idx] if C_obs[peak_idx] > 0 else 1.0
    t_b_guess = t_peak * 0.5
    t_max = np.max(t_obs)

    # model name: (function, parameter names, p0, lower bounds, upper bounds)
    models_config = {
        "sauty_uniform_radial": (
            sauty_uniform_radial,
            ["t_c", "Pe", "C_max"],
            [t_peak, 10.0, C_peak],
            [0.1, 0.1, 1e-5],
            [t_max * 2.0, 500.0, C_peak * 5.0],
        ),
        "bullivant_single_fracture": (
            bullivant_single_fracture,
            ["t_b", "t_p", "L"],
            [t_b_guess, t_peak, C_peak],
            [0.01, 0.1, 1e-5],
            [t_peak * 0.99, t_max, C_peak * 10.0],
        ),
        "bullivant_fracture_matrix": (
            bullivant_fracture_matrix,
            ["t_p", "t_b", "L"],
            [t_peak, t_b_guess, C_peak],
            [0.1, 0.01, 1e-5],
            [t_max, t_peak * 0.99, C_peak * 10.0],
        ),
        "houseworth_fracture_matrix": (
            houseworth_fracture_matrix,
            ["t_a", "Pe", "V_m", "A"],
            [t_b_guess, 10.0, 0.1, C_peak],
            [0.01, 0.1, 1e-4, 1e-5],
            [t_peak * 0.99, 500.0, 10.0, C_peak * 10.0],
        ),
    }

    results = []
    for name, (func, param_names, p0, lower, upper) in models_config.items():
        popt, _ = curve_fit(
            func, t_obs, C_obs, p0=p0, bounds=(lower, upper), method="trf"
        )
        rmse = np.sqrt(np.mean((C_obs - func(t_obs, *popt)) ** 2))
        results.append(
            {
                "model": name,
                "func": func,
                "params": dict(zip(param_names, popt)),
                "rmse": rmse,
            }
        )
    return results


def print_fit_summary(all_fits):
    print("\nFit summary")
    for fit in all_fits:
        params = ", ".join(f"{k}={v:.4g}" for k, v in fit["params"].items())
        print(f"  {fit['well'].upper()}  {fit['model']:<28} "
              f"RMSE={fit['rmse']:.3f}  {params}")


def save_fit_summary(all_fits, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["well", "model", "rmse", "parameters"])
        for fit in all_fits:
            params = "; ".join(f"{k}={v:.6g}" for k, v in fit["params"].items())
            writer.writerow([fit["well"], fit["model"], f"{fit['rmse']:.6g}", params])


# -------------------------------------------------------------- data I/O


def import_tracer_data(data_dir):
    """Return {well: (time_days, concentration_ppb)}, sorted by time.

    Files have a '%' title line, a 'ppb days' header, then two columns.
    """
    tracer_data = {}
    for file in sorted(Path(data_dir).glob("*.txt")):
        C, t = np.loadtxt(file, skiprows=2, unpack=True)
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
        # Skip missing (NaN) and zero concentrations
        mask = np.isfinite(C) & (C != 0)
        plt.scatter(t[mask], C[mask], label=f"C_{well}", s=2)

    plt.xlabel("Time (days)")
    plt.ylabel("Tracer Concentration (ppb)")
    plt.title("Tracer Concentration vs Time")

    plt.legend()
    plt.savefig("tracer_data.jpg", dpi=300)
    plt.close()


def plot_fitted_curves(t_obs, C_obs, fits, well):
    plt.figure(figsize=(10, 6))
    plt.scatter(t_obs, C_obs, color="black", s=15, alpha=0.7, label="Measured Field Data")

    # Smooth time grid for the analytical curves
    t_smooth = np.linspace(np.min(t_obs), np.max(t_obs), 500)

    for fit in fits:
        color, linestyle = MODEL_STYLES[fit["model"]]
        plt.plot(
            t_smooth,
            fit["func"](t_smooth, *fit["params"].values()),
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"{fit['model']} (RMSE: {fit['rmse']:.2f})",
        )

    plt.xlabel("Time (days)")
    plt.ylabel("Tracer Concentration (ppb)")
    plt.title(f"Analytical Model Fits - Well {well.upper()}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.savefig(f"./fitted_curves/fitted_curves_{well}.jpg", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
