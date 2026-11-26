import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import least_squares
from scipy.signal import medfilt

WELLS = ["b", "c", "d", "e", "f"]
INJECTOR = "a1"
EXCLUDED_FROM_TENSOR = ["e"]    # no usable breakthrough curve
MAX_CONDITION = 10.0          # larger: the well directions cannot resolve a tensor
FIG_DIR = "figures"
RESULTS_FILE = "results.txt"

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

# Constants the brief does not give: solute diffusion in water is of the
# order of 1e-9 m2/s (Houseworth et al. 2013, paragraph 16).
D_0 = 1e-9      # molecular diffusion coefficient in the fracture [m2/s]
D_M = 1e-9      # matrix diffusion coefficient [m2/s]
MU = 2e-4       # water viscosity at ~150 C [Pa s]; reservoir temperature is not given
DAY = 86400.0
MILLIDARCY = 9.869233e-16   # [m2]


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    wells = {w: load_well(w) for w in [INJECTOR] + WELLS}
    fault = np.loadtxt("coordinates/fault.txt", skiprows=1)
    geometry = geometric_parameters(wells, fault)
    plot_field(wells, fault, geometry)

    rows, k_by_model = [], {}
    for w in WELLS:
        btc = process_btc(*load_btc(w))
        fits = fit_models(btc, TABLE_1[w][0])
        plot_fits(w, btc, fits)
        dz, dP = TABLE_1[w]
        for name, fit in fits.items():
            p = fit["params"]
            L = p.get("L", dz)              # fitted path length, otherwise dz (Table 1)
            t_adv = list(p.values())[0]     # t_c, t_m or t_b
            v, k, h_cubic = permeability(L, t_adv, dP)
            k_by_model.setdefault(name, {})[w] = k
            values = dict(FIXED_VALUES[name])
            values.update({key: (val, "*") for key, val in p.items() if key in ("L", "h")})
            values.setdefault("L", (L, ""))
            rows.append([w.upper(), SHORT_NAMES[name],
                         cell(values, "L", "{:.0f}"), f"{v * DAY:.2f}",
                         cell(values, "alpha_L", "{:.0f}"), cell(values, "alpha_T", "{:.0f}"),
                         cell(values, "phi_m", "{:.3f}"), cell(values, "h", "{:.3f}", 1e3),
                         f"{k / MILLIDARCY:.0f}", f"{h_cubic * 1e6:.1f}", "-", f"{fit['rmse']:.3f}"])

    # Anisotropy in the fault plane; well E has no usable breakthrough curve.
    tensor_wells = [w for w in WELLS if w not in EXCLUDED_FROM_TENSOR]
    e1, e2, _ = fault_plane(geometry)
    for name, k_by_well in k_by_model.items():
        k = [k_by_well[w] for w in tensor_wells]
        t = permeability_tensor(k, [geometry[w]["direction"] for w in tensor_wells], e1, e2)
        if t["cond"] > MAX_CONDITION or t["k_min"] <= 0:
            # Directions too close together: only the down-dip value is constrained.
            k_text = f"{np.exp(np.mean(np.log(k))) / MILLIDARCY:.0f}"
            anisotropy = f"n/r (cond {t['cond']:.0f})"
        else:
            k_text = f"{t['k_max'] / MILLIDARCY:.0f}/{t['k_min'] / MILLIDARCY:.0f}"
            anisotropy = f"{t['k_max'] / t['k_min']:.1f} at {t['azimuth']:.0f}/{t['plunge']:.0f}"
        rows.append(["".join(tensor_wells).upper(), SHORT_NAMES[name], "-", "-", "-", "-", "-",
                     "-", k_text, "-", anisotropy, "-"])

    header = ["Well", "Model", "L (m)", "v (m/d)", "aL (m)", "aT (m)", "phi_m", "h (mm)",
              "k (mD)", "h_cubic (um)", "k_max/k_min", "RMSE"]
    write_table(header, rows, RESULTS_FILE)

    plt.show()


# Values held fixed from Table 2 for each model, as (value, marker).
FIXED_VALUES = {
    "Uniform porous - Sauty (1980)": {"alpha_L": (TABLE_2["sauty"]["alpha_L"], ""),
                                      "alpha_T": (TABLE_2["sauty"]["alpha_T"], "")},
    "Single fracture - Bullivant & Sul.(1989)": {
        "alpha_L": (TABLE_2["sauty"]["alpha_L"], ""),
        "h": (TABLE_2["bullivant_single_fracture"]["h"], "")},
    "Fracture-matrix - Bullivant & Sul. (1989)": {
        "phi_m": (TABLE_2["bullivant_fracture_matrix"]["phi_matrix"], "")},
    "Fracture-matrix - Houseworth et al. (2013)": {
        "phi_m": (TABLE_2["bullivant_fracture_matrix"]["phi_matrix"], ""),
        "h": (TABLE_2["houseworth"]["h"], "")},
}

SHORT_NAMES = {
    "Uniform porous - Sauty (1980)": "Sauty 2-D (Eq. 39-42)",
    "Single fracture - Bullivant & Sul.(1989)": "B&O single fracture",
    "Fracture-matrix - Bullivant & Sul. (1989)": "B&O fracture-matrix",
    "Fracture-matrix - Houseworth et al. (2013)": "Houseworth",
}


def cell(values, key, fmt, scale=1.0):
    """Format a reported value, '-' if the model does not have it."""
    if key not in values:
        return "-"
    value, marker = values[key]
    return fmt.format(value * scale) + marker


def write_table(header, rows, path):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = "  ".join(h.rjust(n) for h, n in zip(header, widths))
    lines = [line, "-" * len(line)]
    lines += ["  ".join(str(c).rjust(n) for c, n in zip(r, widths)) for r in rows]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


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
def fit_models(btc, dz):
    """Fit the models for every well using SciPy's NLLS algorithm.

    The models are fitted to C/C_max with the Table 2 values held fixed.
    Besides the time scale of every model, only these are fitted:
      - the flow path length L for Sauty and the single fracture (the true
        path through the fault is unknown; dz from Table 1 is the first guess),
      - the fracture aperture h for the B&O fracture-matrix model. Its Table 2
        value (3 mm) would need a matrix diffusion coefficient ~500 times that
        of water to fit, so h is estimated with phi_m = 0.005 kept,
      - the matrix cross-flow k for Houseworth.
    """
    t_end, t_p = btc["t"][-1], btc["t_peak"]
    times = np.linspace(btc["t_arrival"], t_end, 8)
    lengths = (0.5 * dz, dz, 2.0 * dz)

    # Sauty: the producer is on the flow axis of the forced flow A1 -> producer,
    # so y = 0 and alpha_T drops out of Eqs. 39-42; P = L / alpha_L (Eq. 26).
    alpha_L = TABLE_2["sauty"]["alpha_L"]

    # B&O single fracture: Taylor dispersion (Eq. 4) plus mechanical dispersion
    # alpha_L u, since real fractures disperse far more than parallel plates
    # (Houseworth section 6). w = q R / (D A) = L / (alpha_L + D_Tr / u), with
    # D_Tr / u = u h^2 / (210 D_0) and u = L / t_m.
    h_sf = TABLE_2["bullivant_single_fracture"]["h"]

    def w_single_fracture(t_m, L):
        u = L / (t_m * DAY)
        return L / (alpha_L + u * h_sf**2 / (210.0 * D_0))

    # Fracture-matrix models: w = (q / R A) a^2 / (D_m n_m^2) = h^2 / (t_b D_m phi_m^2)
    # (B&O Eq. 17; Houseworth Eq. 57 with l = b / (2 phi_m) gives the same group).
    # Houseworth's Table 2 row gives no phi_m; the B&O value is used (same rock).
    phi_m = TABLE_2["bullivant_fracture_matrix"]["phi_matrix"]
    h_hw = TABLE_2["houseworth"]["h"]

    def w_fracture_matrix(h, t_b, phi_m):
        return h**2 / (t_b * DAY * D_M * phi_m**2)

    # label: (function, parameter names, starting points, lower bounds, upper bounds)
    setups = {
        "Uniform porous - Sauty (1980)": (
            lambda t, t_c, L: sauty(t, t_c, L / alpha_L), ["t_c", "L"],
            [[t0, L] for t0 in times for L in lengths],
            [1.0, 10.0], [10.0 * t_end, 1e4]),
        "Single fracture - Bullivant & Sul.(1989)": (
            lambda t, t_m, L: bullivant_single_fracture(t, t_m, w_single_fracture(t_m, L)),
            ["t_m", "L"],
            [[t0, L] for t0 in times for L in lengths],
            [1.0, 10.0], [10.0 * t_end, 1e4]),
        "Fracture-matrix - Bullivant & Sul. (1989)": (
            lambda t, t_b, h: bullivant_fracture_matrix(t, t_b, w_fracture_matrix(h, t_b, phi_m)),
            ["t_b", "h"],
            [[f * t_p, h] for f in (0.2, 0.5, 0.8) for h in (5e-5, 2e-4, 1e-3)],
            [1.0, 1e-6], [t_p, 1e-2]),
        "Fracture-matrix - Houseworth et al. (2013)": (
            lambda t, t_b, k: houseworth(t, t_b, w_fracture_matrix(h_hw, t_b, phi_m), k),
            ["t_b", "k"],
            [[f * t_p, k] for f in (0.2, 0.5, 0.8) for k in (0.0, 0.1, 1.0, 10.0)],
            [1.0, 0.0], [t_p, 1e2]),
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


# func: permeability.
def permeability(L, t_adv, dP):
    """Effective fracture permeability from Darcy's law and the cubic law.

    The tracer moves with the fluid velocity in the fracture, v = L / t_adv.
    Inside an open fracture the Darcy flux equals that velocity, so
        k = mu v L / dP            (Darcy's law)
        h = sqrt(12 k)             (cubic law, k = h^2 / 12)
    L [m], t_adv [days], dP [MPa]. Returns v [m/s], k [m2], h [m].
    """
    v = L / (t_adv * DAY)
    k = MU * v * L / (dP * 1e6)
    return v, k, np.sqrt(12.0 * k)


# func: anisotropy.
def fault_plane(geometry):
    """Plane through the well-fault intersection points (least squares).

    Returns the strike direction e1 (horizontal), the down-dip direction e2
    and the plane normal, all unit vectors.
    """
    pts = np.array([g["point"] for g in geometry.values()])
    normal = np.linalg.svd(pts - pts.mean(axis=0))[2][-1]
    e1 = np.cross([0.0, 0.0, 1.0], normal)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    if e2[2] > 0:               # point e2 down-dip
        e2 = -e2
    return e1, e2, normal


def permeability_tensor(k, directions, e1, e2):
    """Fit a 2-D permeability tensor in the fault plane.

    Each well pair gives the permeability k_i along its direction n_i,
    projected onto the fault plane: k_i = n_i^T K n_i. With
    K = [[a, b], [b, c]] this is linear in (a, b, c) and is solved by least
    squares. Needs at least three directions that are not parallel.
    """
    n = np.array([[d @ e1, d @ e2] for d in directions])
    n /= np.linalg.norm(n, axis=1)[:, None]
    A = np.column_stack([n[:, 0] ** 2, 2.0 * n[:, 0] * n[:, 1], n[:, 1] ** 2])
    k = np.asarray(k)
    # Solve in relative terms so that every well counts equally.
    (a, b, c), *_ = np.linalg.lstsq(A / k[:, None], np.ones_like(k), rcond=None)
    K = np.array([[a, b], [b, c]])

    values, vectors = np.linalg.eigh(K)
    v_max = vectors[0, 1] * e1 + vectors[1, 1] * e2          # 3-D direction of k_max
    if v_max[2] > 0:
        v_max = -v_max
    azimuth = np.degrees(np.arctan2(v_max[0], v_max[1])) % 360.0
    plunge = np.degrees(np.arcsin(-v_max[2]))
    misfit = np.sqrt(np.mean((A @ [a, b, c] / k - 1.0) ** 2))   # relative RMS
    return {"K": K, "k_max": values[1], "k_min": values[0], "azimuth": azimuth,
            "plunge": plunge, "cond": np.linalg.cond(A), "misfit": misfit}


# Draw graphs for time (days) vs C/Cmax for every well, in the style of Figure 1.
STYLES = {
    "Uniform porous - Sauty (1980)": dict(ls="--"),
    "Single fracture - Bullivant & Sul.(1989)": dict(ls="-."),
    "Fracture-matrix - Bullivant & Sul. (1989)": dict(ls=":"),
    "Fracture-matrix - Houseworth et al. (2013)": dict(ls="-", marker="x", markevery=40, ms=4),
}


def plot_fits(well, btc, fits):
    t, y = btc["t"], btc["C"] / btc["C_peak"]
    t_fine = np.linspace(0.0, 1000.0, 2001)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    nonzero = y > 0                     # zero values are fitted but not plotted
    ax.plot(t[nonzero], y[nonzero], "o", ms=4, mfc="none", color="k", label="Measurement")
    for name in STYLES:
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
