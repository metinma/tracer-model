import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import least_squares
from scipy.signal import medfilt

WELLS = ["b", "c", "d", "e", "f"]
INJECTOR = "a1"
FIG_DIR = "figures"

# TABLE 1: injector A1 -> producer: (dz [m], dP [MPa])
TABLE_1 = {
    "b": (750.0, 7.0),
    "f": (1000.0, 8.0),
    "e": (1400.0, 9.5),
    "d": (1500.0, 10.0),
    "c": (1600.0, 11.0),
}

# TABLE 2 parameters: example estimates given in the project description.
TABLE_2 = {
    "sauty": {"alpha_L": 150.0, "alpha_T": 0.2 * 150.0},
    "bullivant_single_fracture": {"h": 3e-3},
    "bullivant_fracture_matrix": {"h": 3e-3, "phi_matrix": 0.005},
    "houseworth": {"h": 2e-4},
}

# Shape parameters held fixed, as in Figure 1 of the project description.
# The brief does not give the constants it used; these reproduce Figure 1.
D_M = 1e-10                    # matrix diffusion coefficient [m2/s]
DAY = 86400.0
# B&O Eq. 8: Peclet number w, read off Figure 1 (no relation to h = 3 mm found).
W_SINGLE_FRACTURE = 15.0
# B&O Eq. 17: w * t_b = h^2 / (D_m * phi_m) [days], from the Table 2 h and phi_m.
G_FRACTURE_MATRIX = (TABLE_2["bullivant_fracture_matrix"]["h"] ** 2
                     / (D_M * TABLE_2["bullivant_fracture_matrix"]["phi_matrix"]) / DAY)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    wells = {w: load_well(w) for w in [INJECTOR] + WELLS}
    fault = np.loadtxt("coordinates/fault.txt", skiprows=1)
    geometry = geometric_parameters(wells, fault)
    plot_field(wells, fault, geometry)

    print(f"\n{'well':>4} {'model':>44} {'parameters':>40} {'RMSE':>6}")
    for w in WELLS:
        btc = process_btc(*load_btc(w))
        fits = fit_models(btc)
        for name, fit in fits.items():
            p = ", ".join(f"{k}={v:.3g}" for k, v in fit["params"].items())
            print(f"{w:>4} {name:>44} {p:>40} {fit['rmse']:6.3f}")
        plot_fits(w, btc, fits)

    plt.show()


# func: load btc data.
def load_btc(well):
    """Return time [days] and concentration [ppb], sorted by time."""
    data = np.loadtxt(f"tracer_data/{well}.txt", skiprows=2)
    C, t = data[:, 0], data[:, 1]
    order = np.argsort(t, kind="stable")
    return t[order], C[order]


# func: load well data.
def load_well(well):
    """Return the well trajectory as an (n, 3) array of x, y, z [m]."""
    return np.loadtxt(f"coordinates/{well}.txt", skiprows=1)


# func: process btc data.
def process_btc(t, C, n_background=5, kernel=5, arrival_fraction=0.05):
    # Substract a fixed background concentration.
    background = np.median(C[:n_background])
    C = np.clip(C - background, 0.0, None)

    # Handle noise: the median filter is only used to find the peak and the
    # arrival; the models are fitted to the raw (background-free) data.
    C_smooth = medfilt(C, kernel)

    # Weight the uneven samples by the time interval each sample represents.
    edges = np.concatenate(([t[0]], (t[1:] + t[:-1]) / 2, [t[-1]]))
    weights = np.diff(edges)
    weights = weights / weights.mean()

    # Find robust peak.
    i_peak = np.argmax(C_smooth)
    C_peak = C_smooth[i_peak]

    # Find arrival time: first time the smoothed curve exceeds a fraction of the peak.
    t_arrival = t[np.argmax(C_smooth > arrival_fraction * C_peak)]

    # Find peak time.
    t_peak = t[i_peak]

    return {"t": t, "C": C, "weights": weights, "background": background,
            "C_peak": C_peak, "t_arrival": t_arrival, "t_peak": t_peak}


# func: geometric parameters.
def geometric_parameters(wells, fault):
    # Find wellbore - fracture intersections.
    points = {w: fault_intersection(xyz, fault) for w, xyz in wells.items()}

    geometry = {}
    for w in WELLS:
        v = points[w] - points[INJECTOR]
        # Find straight distance between the intersection points.
        L = np.linalg.norm(v)
        # Find the direction vector for anisotropy calculations.
        geometry[w] = {"point": points[w], "L": L, "direction": v / L}
    geometry[INJECTOR] = {"point": points[INJECTOR]}
    return geometry


def fault_intersection(xyz, fault, n=5000):
    """First point where the well trajectory crosses the fault surface.

    If the well never reaches the fault, the trajectory point closest to the
    fault (in z) is returned instead.
    """
    # Resample the trajectory evenly along its length.
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))))
    s_new = np.linspace(0.0, s[-1], n)
    path = np.column_stack([np.interp(s_new, s, xyz[:, k]) for k in range(3)])

    # Height of the well above the fault surface at every sample.
    dz = path[:, 2] - griddata(fault[:, :2], fault[:, 2], path[:, :2])
    ok = ~np.isnan(dz)
    path, dz = path[ok], dz[ok]

    crossing = np.nonzero(np.diff(np.sign(dz)))[0]
    if len(crossing) == 0:
        return path[np.argmin(np.abs(dz))]
    return path[crossing[0]]


# Draw a 3D plot of the field.
def plot_field(wells, fault, geometry):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(fault[:, 0], fault[:, 1], fault[:, 2], color="grey", alpha=0.3)
    for w, xyz in wells.items():
        line, = ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=w.upper())
        ax.scatter(*geometry[w]["point"], color=line.get_color(), s=30)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    ax.set_title("Wells and fault (dots: well-fault intersections)")
    fig.savefig(f"{FIG_DIR}/field.png", dpi=150)


# All models return C/C_max: each curve is scaled so that its own peak is 1,
# as in Figure 1 of the project description.

# func: Sauty's uniform flow.
def sauty(t, t_c, P, y_R=0.0):
    """Sauty (1980) Eqs. 39-42: slug injection in 2-D uniform flow.

    t_c: convective transit time [days], P = X / alpha_L (longitudinal Peclet),
    y_R = y / sqrt(alpha_L * alpha_T) (0 = observation well on the flow axis,
    which gives Eq. 42).
    """
    a2 = P**2 + y_R**2                           # mixed Peclet number squared, Eq. 33
    tR = np.maximum(P * t / t_c, 1e-12)          # t_R' = u t / alpha_L, Eq. 32
    tR_max = np.sqrt(a2 + 4.0) - 2.0             # Eq. 41

    def log_shape(x):                            # Eq. 39 without the constant K
        return -np.log(x) - (a2 + x**2) / (4.0 * x)

    return np.exp(log_shape(tR) - log_shape(tR_max))


def sauty_1d(t, t_c, P):
    """Sauty (1980) Eq. 22 with K' from Eq. 23 ("Eq. 23" in Figure 1).

    1-D uniform flow, derivative of the step solution (flux concentration).
    """
    tR = np.maximum(t / t_c, 1e-12)
    tR_max = np.sqrt(1.0 + 9.0 / P**2) - 3.0 / P     # Eq. 23

    def log_shape(x):
        return -1.5 * np.log(x) - P * (1.0 - x) ** 2 / (4.0 * x)

    return np.exp(log_shape(tR) - log_shape(tR_max))


# func: Bullivant's single fracture flow.
def bullivant_single_fracture(t, t_m, w):
    """Bullivant & O'Sullivan (1989) Eq. 8.

    t_m: mean arrival time [days], w: Peclet number q R / (D_Tr A).
    """
    t = np.maximum(t, 1e-12)
    t_p = t_m * (np.sqrt(9.0 + w**2) - 3.0) / w   # peak time, from Eq. 9

    def log_shape(x):
        return -1.5 * np.log(x) - w * (x - t_m) ** 2 / (4.0 * t_m * x)

    return np.exp(log_shape(t) - log_shape(t_p))


# func: Bullivant's fracture matrix flow.
def bullivant_fracture_matrix(t, t_b, w):
    """Bullivant & O'Sullivan (1989) Eq. 17.

    t_b: arrival time without matrix diffusion [days],
    w = (q / R A) a^2 / (D_m n_m^2).
    """
    return houseworth(t, t_b, w, 0.0)


# func: Houseworth's fracture matrix flow.
def houseworth(t, t_b, w, k):
    """Houseworth et al. (2013) Eq. 57, fracture concentration.

    Source in the fracture (eta_0 = 0) and negligible longitudinal matrix
    velocity (v_mz = 0), so xi = L / l is constant and tau - xi = xi (t - t_b) / t_b.
    Eq. 57 then becomes
        c ~ (t - t_b)^(-3/2) exp(-t_b / (w (t - t_b)) - k (t - t_b) / t_b)
    with w = 4 Pe / xi (the same w as Bullivant Eq. 17) and k = Pe V^2 xi / 4,
    the matrix cross-flow term. k = 0 gives Bullivant & O'Sullivan Eq. 17.
    """
    s = t - t_b
    after = s > 0
    s = np.where(after, s, 1.0)
    # Peak delay from d(log c)/ds = 0, written so that k = 0 is allowed.
    s_p = 2.0 * (t_b / w) / (1.5 + np.sqrt(2.25 + 4.0 * k / w))

    def log_shape(x):
        return -1.5 * np.log(x) - t_b / (w * x) - k * x / t_b

    return np.where(after, np.exp(log_shape(s) - log_shape(s_p)), 0.0)


# func: fit models.
def fit_models(btc):
    """Fit the models for every well using SciPy's NLLS algorithm.

    The models are fitted to C/C_max. Sauty and Houseworth fit all their
    parameters; for the two Bullivant models the shape is held fixed (see the
    constants at the top) and only the time scale is fitted.
    """
    t_p, t_end = btc["t_peak"], btc["t"][-1]
    G = G_FRACTURE_MATRIX

    # label: (function, parameter names, starting points, lower bounds, upper bounds)
    setups = {
        "Eq. 23": (
            sauty_1d, ["t_c", "P"],
            [[f * t_p, P] for f in (1.0, 1.5) for P in (3.0, 10.0, 30.0)],
            [1.0, 0.1], [10.0 * t_end, 1e3]),
        "Uniform porous - Sauty (1980)": (
            sauty, ["t_c", "P"],
            [[f * t_p, P] for f in (1.0, 1.5) for P in (3.0, 10.0, 30.0)],
            [1.0, 0.1], [10.0 * t_end, 1e3]),
        "Single fracture - Bullivant & Sul.(1989)": (
            lambda t, t_m: bullivant_single_fracture(t, t_m, W_SINGLE_FRACTURE), ["t_m"],
            [[f * t_p] for f in (1.0, 1.5, 2.0)],
            [1.0], [10.0 * t_end]),
        "Fracture-matrix - Bullivant & Sul. (1989)": (
            lambda t, t_b: bullivant_fracture_matrix(t, t_b, G / t_b), ["t_b"],
            [[f * t_p] for f in (0.2, 0.5, 0.8)],
            [1.0], [t_p]),
        "Fracture-matrix - Houseworth et al. (2013)": (
            houseworth, ["t_b", "w", "k"],
            [[f * t_p, 2 * f / (3 * (1 - f)), k]                 # Eq. 18 of B&O
             for f in (0.2, 0.4, 0.6, 0.8) for k in (0.0, 0.1, 1.0)],
            [0.0, 1e-6, 0.0], [t_p, 1e3, 1e2]),
    }

    return {name: fit_model(func, names, starts, lower, upper, btc)
            for name, (func, names, starts, lower, upper) in setups.items()}


def fit_model(func, names, starts, lower, upper, btc):
    """Weighted least squares on C/C_max from several starting points; keep the best."""
    t, y, sw = btc["t"], btc["C"] / btc["C_peak"], np.sqrt(btc["weights"])

    def residuals(p):
        return sw * (func(t, *p) - y)

    best = None
    for p0 in starts:
        p0 = np.clip(p0, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
        res = least_squares(residuals, p0, bounds=(lower, upper))
        if best is None or res.cost < best.cost:
            best = res

    rmse = np.sqrt(np.mean((func(t, *best.x) - y) ** 2))
    return {"func": func, "params": dict(zip(names, best.x)), "rmse": rmse}


# Draw graphs for time (days) vs C/Cmax for every well, in the style of Figure 1.
STYLES = {
    "Eq. 23": dict(ls="-"),
    "Uniform porous - Sauty (1980)": dict(ls="--"),
    "Single fracture - Bullivant & Sul.(1989)": dict(ls="-."),
    "Fracture-matrix - Bullivant & Sul. (1989)": dict(ls=":"),
    "Fracture-matrix - Houseworth et al. (2013)": dict(ls="-", marker="x", markevery=40, ms=4),
}


def plot_fits(well, btc, fits):
    t, y = btc["t"], btc["C"] / btc["C_peak"]
    t_fine = np.linspace(0.0, 1000.0, 2001)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(fits)
    # Same legend order as Figure 1: Eq. 23, Measurement, then the four models.
    ax.plot(t_fine, fits[names[0]]["func"](t_fine, *fits[names[0]]["params"].values()),
            color="k", lw=1, label=names[0], **STYLES[names[0]])
    ax.plot(t, y, "o", ms=4, mfc="none", color="k", label="Measurement")
    for name in names[1:]:
        fit = fits[name]
        ax.plot(t_fine, fit["func"](t_fine, *fit["params"].values()),
                color="k", lw=1, label=name, **STYLES[name])
    ax.set_xlabel("time (days)")
    ax.set_ylabel("$c/c_{max}$")
    ax.set_xlim(0.0, 1000.0)
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", ".2", ".4", ".6", ".8", "1"])
    ax.grid(color="0.85")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title(f"A1 -> {well.upper()}")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fit_{well}.png", dpi=150)


if __name__ == "__main__":
    main()
