import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter1d, gaussian_filter


# -----------------------------
# 0. Helper Functions
# -----------------------------
def smooth_profile(profile, sigma=1.5):
    return gaussian_filter1d(profile, sigma=sigma)

def find_fit_mask(r_um, profile, threshold=0.08):
    """
    Keep fitting until the signal has decayed near baseline.
    threshold is relative to the normalized profile max.
    """
    p = normalize_profile(profile)
    peak_idx = np.argmax(p)

    end_idx = len(p) - 1
    for i in range(peak_idx + 1, len(p)):
        if p[i] < threshold:
            end_idx = i
            break

    mask = np.zeros_like(p, dtype=bool)
    mask[:end_idx + 1] = True
    return mask


# -----------------------------
# 1. Load real image
# -----------------------------
def load_hologram_image(path):
    img = Image.open(path).convert("L")   # grayscale
    return np.array(img, dtype=float)


def normalize_profile(profile):
    profile = profile - np.min(profile)
    if np.max(profile) > 0:
        profile = profile / np.max(profile)
    return profile


# -----------------------------
# 2. Click center manually
# -----------------------------
def click_center(I):
    fig, ax = plt.subplots(figsize=(6, 6))
    I_disp = (I - I.min()) / (I.max() - I.min())
    ax.imshow(I_disp, cmap="gray")
    ax.set_title("Click the center of the rings, then close the window")
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)

    if not pts:
        raise RuntimeError("No point clicked.")

    x0, y0 = pts[0]
    return x0, y0


# -----------------------------
# 3. Build grid for simulated model
# -----------------------------
def make_grid(N, fov_um):
    x = np.linspace(-fov_um / 2, fov_um / 2, N)
    X, Y = np.meshgrid(x, x)
    return X, Y


# -----------------------------
# 4. Forward hologram model
# -----------------------------
def hologram_model_2d(X, Y, z_um, k, A=0.15, x0_um=0.0, y0_um=0.0):
    r = np.sqrt((X - x0_um)**2 + (Y - y0_um)**2 + z_um**2)

    E0 = np.ones_like(r, dtype=np.complex128)
    Es = A * np.exp(1j * k * r) / r

    I = np.abs(E0 + Es)**2
    return I


# -----------------------------
# 5. Radial profile from REAL image (pixel space)
# -----------------------------
def radial_profile_pixels(I, x0_px, y0_px, nbins=200):
    h, w = I.shape
    Y, X = np.indices((h, w))
    R = np.sqrt((X - x0_px)**2 + (Y - y0_px)**2)

    r_max = R.max()
    bin_edges = np.linspace(0, r_max, nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    profile = np.zeros(nbins)
    counts = np.zeros(nbins)

    r_flat = R.ravel()
    I_flat = I.ravel()
    bin_index = np.digitize(r_flat, bin_edges) - 1

    for i in range(len(r_flat)):
        idx = bin_index[i]
        if 0 <= idx < nbins:
            profile[idx] += I_flat[i]
            counts[idx] += 1

    valid = counts > 0
    profile[valid] /= counts[valid]

    return bin_centers, profile


# -----------------------------
# 6. Radial profile from SIMULATED image
# -----------------------


#Add center finding function to look for intensity radially and center based on split difference of r.
def radial_profile_model(I, X, Y, x0_um=0.0, y0_um=0.0, nbins=200):
    R = np.sqrt((X - x0_um)**2 + (Y - y0_um)**2)

    r_max = R.max()
    bin_edges = np.linspace(0, r_max, nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    profile = np.zeros(nbins)
    counts = np.zeros(nbins)

    r_flat = R.ravel()
    I_flat = I.ravel()
    bin_index = np.digitize(r_flat, bin_edges) - 1

    for i in range(len(r_flat)):
        idx = bin_index[i]
        if 0 <= idx < nbins:
            profile[idx] += I_flat[i]
            counts[idx] += 1

    valid = counts > 0
    profile[valid] /= counts[valid]

    return bin_centers, profile


# -----------------------------
# 7. Brute-force z fitting
# -----------------------------
def estimate_z_from_profile(r_meas, profile_meas, X, Y, k, z_values, A=0.15, nbins=200):
    profile_meas = normalize_profile(profile_meas)

    best_z = None
    best_error = np.inf
    best_profile = None
    best_sigma = None

    blur_sigmas = [0.0, 0.5, 1.0, 1.5, 2.0]

    total = len(z_values)

    for i, z in enumerate(z_values):
        if i % 10 == 0 or i == total - 1:
            print(f"Checking z = {z:.2f} um   ({i+1}/{total})")

        I_model = hologram_model_2d(X, Y, z, k, A=A, x0_um=0.0, y0_um=0.0)

        for sigma in blur_sigmas:
            if sigma > 0:
                I_model_use = gaussian_filter(I_model, sigma=sigma)
            else:
                I_model_use = I_model

            r_model, profile_model = radial_profile_model(
                I_model_use, X, Y, x0_um=0.0, y0_um=0.0, nbins=nbins
            )

            profile_model_interp = np.interp(r_meas, r_model, profile_model)
            profile_model_interp = normalize_profile(profile_model_interp)

            error = np.sum((profile_model_interp - profile_meas) ** 2)

            if error < best_error:
                best_error = error
                best_z = z
                best_profile = profile_model_interp
                best_sigma = sigma

    return best_z, best_error, best_profile, best_sigma


# -----------------------------
# 8. Show image helper
# -----------------------------
def show_image(I, title=""):
    I_disp = (I - I.min()) / (I.max() - I.min())
    plt.imshow(I_disp, cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.show()


# -----------------------------
# 9. Main
# -----------------------------
def main():
    # -------------------------
    # Known constants from HoloGen dataset generation
    # -------------------------
    lambda0_um = 0.632
    n_medium = 1.4
    k = 2 * np.pi * n_medium / lambda0_um

    # -------------------------
    # Image path
    # -------------------------
    image_path = "RealTestPicture.png"

    # -------------------------
    # Load image
    # -------------------------
    I_meas = load_hologram_image(image_path)

    # -------------------------
    # Click center manually
    # -------------------------
    x0_px, y0_px = click_center(I_meas)
    print(f"Chosen center: x = {x0_px:.2f}, y = {y0_px:.2f}")

    # Show clicked center
    plt.figure(figsize=(6, 6))
    I_disp = (I_meas - I_meas.min()) / (I_meas.max() - I_meas.min())
    plt.imshow(I_disp, cmap="gray")
    plt.scatter([x0_px], [y0_px], c="red", s=40)
    plt.title("Chosen center")
    plt.show()

    # -------------------------
    # Get radial profile from real image
    # -------------------------
    r_meas_px, profile_meas = radial_profile_pixels(I_meas, x0_px, y0_px, nbins=200)

    # HoloGen default pixel pitch from generate_dataset.py
    pixel_size_um = 0.065
    r_meas_um = r_meas_px * pixel_size_um

    # Smooth only the measured 1D radial profile
    profile_meas_smooth = smooth_profile(profile_meas, sigma=1.5)

    # Automatically choose fitting region based on where rings die out
    fit_mask = find_fit_mask(r_meas_um, profile_meas_smooth, threshold=0.08)

    # Apply mask
    r_meas_fit = r_meas_um[fit_mask]
    profile_meas_fit = profile_meas_smooth[fit_mask]

    # Plot measured radial profile
    plt.figure(figsize=(7, 5))
    plt.plot(r_meas_um, normalize_profile(profile_meas_smooth), label="Measured radial profile")
    plt.axvline(r_meas_fit[-1], color="red", linestyle="--", label="Fit cutoff")
    plt.xlabel("Radius (um)")
    plt.ylabel("Normalized intensity")
    plt.title("Measured radial profile")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Simulated model grid
    # -------------------------
    h, w = I_meas.shape
    N = min(h, w)
    fov_um = N * pixel_size_um
    A = 0.15

    X, Y = make_grid(N, fov_um)

    # -------------------------
    # z search
    # -------------------------
    z_values = np.linspace(5.0, 30.0, 100)

    z_fit, err, best_profile, best_sigma = estimate_z_from_profile(
        r_meas=r_meas_fit,
        profile_meas=profile_meas_fit,
        X=X,
        Y=Y,
        k=k,
        z_values=z_values,
        A=A,
        nbins=200
    )

    print("\nDone.")
    print(f"Best-fit z = {z_fit:.3f} um")
    print(f"Best-fit blur sigma = {best_sigma}")
    print(f"Profile error = {err:.6e}")

    # -------------------------
    # Compare measured vs best-fit profile
    # -------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(r_meas_fit, normalize_profile(profile_meas_fit), label="Measured profile")
    plt.plot(r_meas_fit, best_profile, "--", label=f"Best-fit model (z = {z_fit:.2f} um)")
    plt.xlabel("Radius (um)")
    plt.ylabel("Normalized intensity")
    plt.title("Measured vs Best-Fit Radial Profile")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")