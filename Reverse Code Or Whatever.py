import numpy as np
from scipy.optimize import least_squares

#---------Preprocessing-------------

def normalize_hologram(I):
    # divide by a smooth background estimate
    # (later replace with true background frame)
    from scipy.ndimage import gaussian_filter
    bg = gaussian_filter(I, sigma=20)
    return I / (bg + 1e-12)

#---------Forward Model-------------
def make_grid(N: int, fov_um: float):
    x = np.linspace(-fov_um/2, fov_um/2, N)
    X, Y = np.meshgrid(x, x)
    return X, Y

def hologram_model(params, X, Y, k):
    """
    params = [x0_um, y0_um, z_um, A, b0]
    """
    x0, y0, z, A, b0 = params
    r = np.sqrt((X - x0)**2 + (Y - y0)**2 + z**2)
    E0 = np.ones_like(r, dtype=np.complex128)
    Es = A * np.exp(1j * k * r) / r
    I = np.abs(E0 + Es)**2 + b0
    return I

def residuals(params, X, Y, k, I_meas, mask=None):
    I_pred = hologram_model(params, X, Y, k)
    if mask is None:
        return (I_pred - I_meas).ravel()
    return (I_pred[mask] - I_meas[mask]).ravel()

def estimate_center_by_centroid(I, X, Y):
    # crude guess: intensity-weighted centroid after removing background
    J = I - np.median(I)
    J[J < 0] = 0
    s = J.sum()
    if s <= 0:
        return 0.0, 0.0
    x0 = (X * J).sum() / s
    y0 = (Y * J).sum() / s
    return float(x0), float(y0)

def fit_one_hologram(I_meas, lambda0_um=0.632, n_medium=1.333, fov_um=60.0):


    I_meas = normalize_hologram(I_meas)

    N = I_meas.shape[0]
    assert I_meas.shape == (N, N), "I_meas must be square (crop around one particle)."

    k = 2*np.pi*n_medium/lambda0_um
    X, Y = make_grid(N, fov_um)

    # focus fitting on central region to reduce edge influence
    rho = np.sqrt(X**2 + Y**2)
    mask = rho < (0.45 * fov_um)

    # initial guesses
    x0g, y0g = estimate_center_by_centroid(I_meas, X, Y)
    zg  = 25.0
    Ag  = 0.1
    b0g = 0.0
    p0 = np.array([x0g, y0g, zg, Ag, b0g], dtype=float)

    # bounds
    lb = np.array([-fov_um/2, -fov_um/2,  1.0, -2.0, -2.0], dtype=float)
    ub = np.array([ fov_um/2,  fov_um/2, 200.0,  2.0,  2.0], dtype=float)

    res = least_squares(
        residuals, p0, bounds=(lb, ub),
        args=(X, Y, k, I_meas, mask),
        method="trf", max_nfev=250
    )
    return res.x, res

#---------Main Script------------

def main():
    lambda0_um = 0.632
    n_medium   = 1.333
    fov_um     = 60.0
    N          = 512
    k          = 2*np.pi*n_medium/lambda0_um
    X, Y       = make_grid(N, fov_um)

    # "True" parameters
    true = np.array([2.0, -1.5, 30.0, 0.15, 0.0])  # x0,y0,z,A,b0
    I_true = hologram_model(true, X, Y, k)

    # add a bit of noise to mimic reality
    rng = np.random.default_rng(0)
    I_meas = I_true + rng.normal(0, 0.002, I_true.shape)

    fit, res = fit_one_hologram(I_meas, lambda0_um, n_medium, fov_um)

    print("True: [x0, y0, z, A, b0] =", true)
    print("Fit : [x0, y0, z, A, b0] =", fit)

if __name__ == "__main__":
    main()