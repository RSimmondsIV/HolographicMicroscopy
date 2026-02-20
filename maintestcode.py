import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Physical parameters
# ----------------------------
lambda0 = 0.632     # wavelength in microns (HeNe)
n_medium = 1.333    # refractive index of medium (water)
k = 2 * np.pi * n_medium / lambda0

z_particle = 50.0     # particle distance from camera plane (microns)
A = 0.15              # scattering strength (dimensionless)

# ----------------------------
# Camera / image plane
# ----------------------------
N = 512             # pixels per side
fov = 60.0            # field of view in microns

x = np.linspace(-fov/2, fov/2, N)
X, Y = np.meshgrid(x, x)

# Distance from particle to each pixel
r = np.sqrt(X**2 + Y**2 + z_particle**2)

print(X,Y)
print(r)
# ----------------------------
# Optical fields
# ----------------------------
E0 = np.ones_like(r, dtype=complex)      # reference (collimated beam)
Es = A * np.exp(1j * k * r) / r           # scattered spherical wave

# ----------------------------
# Hologram intensity
# ----------------------------
I = np.abs(E0 + Es)**2

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(6, 6))
plt.imshow(I, extent=[-fov/2, fov/2, -fov/2, fov/2], cmap='gray')
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.title('Simulated In line Hologram')
plt.colorbar(label='Intensity')
plt.tight_layout()
plt.show()
