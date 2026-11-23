from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Well parameters for permeability and aperture calculations
WELL_PARAMS = {
    "b": {"dist_m": 750.0, "delta_p_mpa": 7.0},
    "f": {"dist_m": 1000.0, "delta_p_mpa": 8.0},
    "e": {"dist_m": 1400.0, "delta_p_mpa": 9.5},
    "d": {"dist_m": 1500.0, "delta_p_mpa": 10.0},
    "c": {"dist_m": 1600.0, "delta_p_mpa": 11.0},
}


def main():
    # Load data
    tracer_data = import_tracer_data("./tracer_data")
    coordinate_data = import_coordinates("./coordinates")
    well_distances = compute_well_distances(coordinate_data)

    plot_3d_spatial_data(coordinate_data)
    plot_tracer_data(tracer_data)

    print("Well Distances")
    print(well_distances)

    # List of production wells to evaluate
    production_wells = ["b", "c", "d", "e", "f"]
    fitting_summary_list = []

    # Run fitting step function and export plot figures for each production well
    for well in production_wells:
        well_fits = fit_analytical_models(tracer_data, well)
        fitting_summary_list.append(well_fits)

        # Export overlay plot figure for current well
        plot_fitted_curves(tracer_data, well)


def sauty_uniform_radial(t, t_c, Pe, C_max):
    # Sauty (1980) 1D Advection-Dispersion Model.
    t = np.array(t, dtype=float)
    C_calc = np.zeros_like(t)

    # Dimensionless time
    valid = t > 0
    t_R = t[valid] / t_c

    # Time of maximum concentration
    t_Rmax = np.sqrt(1 + Pe ** (-2)) - (1 / Pe)

    # Scaling factor
    K = np.sqrt(t_Rmax) * np.exp((Pe / (4 * t_Rmax)) * ((1 - t_Rmax) ** 2))

    # Dimensionless concentration
    C_R = (K / np.sqrt(t_R)) * np.exp(-(Pe / (4 * t_R)) * ((1 - t_R) ** 2))

    # Observed concentration
    C_calc[valid] = C_R * C_max

    return C_calc


def bullivant_single_fracture(t, t_b, t_p, L):
    # Bullivant & O'Sullivan (1989) Single Fracture Model.
    t = np.array(t, dtype=float)
    C_calc = np.zeros_like(t)

    # Validated time
    valid = t > 0
    t_v = t[valid]

    if t_b >= t_p:
        raise ValueError("Breakthrough time must be less than peak time.")

    # Mean arrival time
    t_m = 2 * t_p - t_b

    # Peclet number. Tracer transport by advection to diffusion
    w = 6 * (t_m * t_p) / (t_m**2 - t_p**2)

    # Observed concentration
    exponent = -w / 4 * (t_v - t_m) ** 2 / t_m / t_v
    C_calc[valid] = L * (2 * t_m) / t_v**1.5 * np.exp(exponent)

    return C_calc


def bullivant_fracture_matrix(t, t_p, t_b, L):
    # Bullivant & O'Sullivan (1989) Fracture-Matrix Model.
    t = np.array(t, dtype=float)
    C_calc = np.zeros_like(t)

    if t_b >= t_p:
        return np.full_like(t, 1e9)

    # Time values after advective breakthrough
    valid = t > t_b
    dt = t[valid] - t_b

    # Peclet number. Tracer transport by advection to diffusion
    w = (2 / 3) * (t_b / (t_p - t_b))

    # Observed concentration
    exponent = -t_b / (w * dt)
    C_calc[valid] = (L / (dt**1.5)) * np.exp(exponent)

    return C_calc


def houseworth_fracture_matrix(t, t_a, Pe, V_m, A, eta_0=0.0):
    # Houseworth et al. (2013) Fracture-Matrix Model.
    t = np.array(t, dtype=float)
    C_calc = np.zeros_like(t)

    # Validated time after advective breakthrough (t > t_a)
    valid = t > t_a
    dt = t[valid] - t_a

    # Spatial offset term
    offset = 1.0 + eta_0

    # Exponential terms representing diffusive exchange and cross-flow loss
    diffusive_term = -(Pe * (offset**2)) / (4.0 * dt)
    crossflow_term = -(Pe * (V_m**2) * dt) / 4.0

    # Dimensionless to physical concentration profile
    prefactor = (A * offset) / (dt**1.5)
    C_calc[valid] = prefactor * np.exp(diffusive_term + crossflow_term)

    return C_calc


def fit_analytical_models(tracer_data, well_name):
    # Clean observation dataset for the given well
    well_df = tracer_data[["Time", f"C_{well_name}"]].dropna()
    t_obs = well_df["Time"].values
    C_obs = well_df[f"C_{well_name}"].values

    # Extract breakthrough peak heuristics for better initial guesses
    peak_idx = np.argmax(C_obs)
    t_peak = float(t_obs[peak_idx]) if t_obs[peak_idx] > 0 else 1.0
    C_peak = float(C_obs[peak_idx]) if C_obs[peak_idx] > 0 else 1.0
    t_b_guess = t_peak * 0.5

    # Define model configurations dictionary
    models_config = {
        "sauty_uniform_radial": {
            "func": sauty_uniform_radial,
            "p0": [t_peak, 10.0, C_peak],
            "bounds": ([0.1, 0.1, 1e-5], [np.max(t_obs) * 2.0, 500.0, C_peak * 5.0]),
            "param_names": ["t_c", "Pe", "C_max"],
        },
        "bullivant_single_fracture": {
            "func": bullivant_single_fracture,
            "p0": [t_b_guess, t_peak, C_peak],
            "bounds": (
                [0.01, 0.1, 1e-5],
                [t_peak * 0.99, np.max(t_obs), C_peak * 10.0],
            ),
            "param_names": ["t_b", "t_p", "L"],
        },
        "bullivant_fracture_matrix": {
            "func": bullivant_fracture_matrix,
            "p0": [t_peak, t_b_guess, C_peak],
            "bounds": (
                [0.1, 0.01, 1e-5],
                [np.max(t_obs), t_peak * 0.99, C_peak * 10.0],
            ),
            "param_names": ["t_p", "t_b", "L"],
        },
        "houseworth_fracture_matrix": {
            "func": houseworth_fracture_matrix,
            "p0": [t_b_guess, 10.0, 0.1, C_peak],
            "bounds": (
                [0.01, 0.1, 1e-4, 1e-5],
                [t_peak * 0.99, 500.0, 10.0, C_peak * 10.0],
            ),
            "param_names": ["t_a", "Pe", "V_m", "A"],
        },
    }

    results_list = []

    # Iterate fitting optimization loop across each analytical function
    for model_name, cfg in models_config.items():
        popt, _ = curve_fit(
            cfg["func"],
            t_obs,
            C_obs,
            p0=cfg["p0"],
            bounds=cfg["bounds"],
            method="trf",
        )

        # Predict concentrations and calculate RMSE
        C_fit = cfg["func"](t_obs, *popt)
        rmse = float(np.sqrt(np.mean((C_obs - C_fit) ** 2)))

        # Formatted parameter dictionary
        fitted_params_dict = {
            name: round(val, 4) for name, val in zip(cfg["param_names"], popt)
        }
        results_list.append(
            {
                "Well": well_name.upper(),
                "Model": model_name,
                "RMSE": round(rmse, 4),
                "Fitted Parameters": fitted_params_dict,
                "C_fit": C_fit,
            }
        )

    return pd.DataFrame(results_list)


def compute_well_distances(coordinates, inj_identifier="a1"):
    well_bottoms = coordinates.groupby("identifier").last().reset_index()

    inj_coords = well_bottoms[well_bottoms["identifier"] == inj_identifier][
        ["x", "y", "z"]
    ].values[0]

    distances = {}

    for _, row in well_bottoms.iterrows():
        well = row["identifier"]

        # Ignore the fault and injector coordinates
        if well == inj_identifier or "fault" in well:
            continue

        well_coords = np.array([row["x"], row["y"], row["z"]])

        # Compute the distance between wells to injector.
        distance = np.linalg.norm(well_coords - inj_coords)
        distances[well] = distance

    return pd.DataFrame(list(distances.items()), columns=["identifier", "Distance, m"])


def import_tracer_data(data_dir):
    tracer_data_path = Path(data_dir)
    tracer_data = None

    for file in tracer_data_path.iterdir():
        well_name = file.stem
        df = pd.read_csv(file, sep=r"\s+", comment="%")

        concentration_identifier = df.columns[0]
        time_identifier = df.columns[1]

        df = df.rename(
            columns={
                time_identifier: "Time",
                concentration_identifier: f"C_{well_name}",
            }
        )

        df = df[["Time", f"C_{well_name}"]]

        if tracer_data is None:
            tracer_data = df
        else:
            tracer_data = pd.merge(tracer_data, df, on="Time", how="outer")

    return tracer_data.sort_values(by="Time").reset_index(drop=True)


def import_coordinates(data_dir):
    coordinates_data_path = Path(data_dir)

    coordinates_data_list = []

    for file in coordinates_data_path.iterdir():
        coordinate_name = file.stem
        df = pd.read_csv(file, sep=r"\s+")

        df.columns = ["x", "y", "z"] + list(df.columns[3:])

        df["identifier"] = coordinate_name

        coordinates_data_list.append(df[["identifier", "x", "y", "z"]])

    return pd.concat(coordinates_data_list, axis=0, ignore_index=True)


def plot_3d_spatial_data(coordinate_data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for identifier, group in coordinate_data.groupby("identifier"):
        if "fault" in identifier:
            ax.scatter(
                group["x"],
                group["y"],
                group["z"],
                color="red",
                s=2,
                alpha=0.4,
                label=identifier,
            )
        else:
            ax.plot(
                group["x"],
                group["y"],
                group["z"],
                linewidth=2,
                label=identifier,
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Well Trajectories and Fault Geometry")

    plt.legend()
    plt.savefig("field_overview.jpg", dpi=300)
    plt.close()


def plot_fitted_curves(tracer_data, well_name):
    # Clean data and extract observed arrays
    well_df = tracer_data[["Time", f"C_{well_name}"]].dropna()
    t_obs = well_df["Time"].values
    C_obs = well_df[f"C_{well_name}"].values

    # Run optimization to get model parameters
    fits_df = fit_analytical_models(tracer_data, well_name)

    # Create a smooth continuous time grid for smooth analytical curve plotting
    t_smooth = np.linspace(np.min(t_obs), np.max(t_obs), 500)

    # Dictionary mapping model names to their analytical functions and plot styling
    model_func_map = {
        "sauty_uniform_radial": (sauty_uniform_radial, "blue", "-"),
        "bullivant_single_fracture": (bullivant_single_fracture, "green", "--"),
        "bullivant_fracture_matrix": (bullivant_fracture_matrix, "orange", "-."),
        "houseworth_fracture_matrix": (houseworth_fracture_matrix, "purple", ":"),
    }

    # Initialize Plot
    plt.figure(figsize=(10, 6))

    # Plot measured tracer scatter data
    plt.scatter(
        t_obs, C_obs, color="black", s=15, alpha=0.7, label="Measured Field Data"
    )

    # Evaluate and plot smooth fitted curves for each model
    for _, row in fits_df.iterrows():
        model_key = row["Model"]
        params_dict = row["Fitted Parameters"]

        if model_key in model_func_map:
            func, color, linestyle = model_func_map[model_key]

            # Extract fitted parameter values in positional order
            param_vals = list(params_dict.values())

            # Predict concentrations on the smooth time vector
            C_smooth = func(t_smooth, *param_vals)

            plt.plot(
                t_smooth,
                C_smooth,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=f"{model_key} (RMSE: {row['RMSE']:.2f})",
            )

    plt.xlabel("Time (days)")
    plt.ylabel("Tracer Concentration (ppb)")
    plt.title(f"Analytical Model Fits - Well {well_name.upper()}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    output_filename = f"fitted_curves_{well_name.lower()}.jpg"
    plt.savefig(f"./fitted_curves/{output_filename}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_tracer_data(tracer_data):
    for column in tracer_data.columns:
        if column != "Time":
            plt.scatter(tracer_data["Time"], tracer_data[column], label=column, s=2)

    plt.xlabel("Time (days)")
    plt.ylabel("Tracer Concentration (ppb)")
    plt.title("Tracer Concentration vs Time")

    plt.legend()
    plt.savefig("tracer_data.jpg", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
