import os
import numpy as np
import h5py
import woma
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Number of particles
N = 10 ** 5
N_label = "n%d" % (10 * np.log10(N))

# Plotting options
font_size = 14
params = {
    "axes.labelsize": font_size,
    "font.size": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "font.family": "serif",
}
matplotlib.rcParams.update(params)

# Earth units
R_E = 6.3710e6  # m

def plot_profile_and_particles(profile, A1_r, A1_rho):
    """Plot the particles."""
    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    # Profile
    ax.plot(profile.A1_r / R_E, profile.A1_rho)

    # Particles
    ax.scatter(A1_r / R_E, A1_rho, c="k", marker=".", s=1 ** 2)

    ax.set_xlim(0, None)
    ax.set_xlabel(r"Radial distance ($R_\oplus$)")
    ax.set_ylabel(r"Density (kg m$^{-3}$)")

    plt.tight_layout()

def plot_spinning_profiles(sp):    
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
        
    ax[0].plot(sp.planet.A1_r / R_E, sp.planet.A1_rho, label="original spherical")
    ax[0].plot(sp.A1_R / R_E, sp.A1_rho, label="equatorial")
    ax[0].plot(sp.A1_Z / R_E, sp.A1_rho, label="polar")
    ax[0].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[0].set_ylabel(r"Density, $\rho$ [kg m$^{-3}$]")
    ax[0].set_yscale("log")
    ax[0].set_xlim(0, 1.1 * sp.R_eq / R_E)
    ax[0].legend()
    
    for i, e in enumerate([
        Ellipse(
            xy=[0, 0],
            width=2 * sp.A1_R[i] / R_E, 
            height=2 * sp.A1_Z[i] / R_E,
            zorder=-i,
        )
        for i in range(len(sp.A1_R))
    ]):
        ax[1].add_artist(e)
        e.set_clip_box(ax[1].bbox)
        e.set_facecolor(plt.get_cmap("viridis")(
            (sp.A1_rho[i] - sp.rho_s) / (sp.rho_0 - sp.rho_s)
        ))
    
    ax[1].set_xlabel(r"Equatorial Radius, $r_{xy}$ $[R_\oplus]$")
    ax[1].set_ylabel(r"Polar Radius, $z$ $[R_\oplus]$")    
    ax[1].set_xlim(0, 1.1 * sp.R_eq / R_E)
    ax[1].set_ylim(0, 1.1 * sp.R_po / R_E)
    ax[1].set_aspect("equal")
    ax[1].set_title(r"Density [kg m$^{-3}$]")
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Plot each snapshot
    for body in ["target", "impactor"]:
        # Load profiles
        profile = woma.Planet(load_file="demo_%s_profile.hdf5" % body)
        spin_profile = woma.SpinPlanet(load_file="demo_%s_spin_profile.hdf5" % body)

        # Load the data
        snapshot_id = 5
        filename = "snapshots/demo_%s_%s_%04d.hdf5" % (body, N_label, snapshot_id)
        with h5py.File(filename, "r") as f:
            # Units from file metadata
            file_to_SI = woma.Conversions(
                m=float(f["Units"].attrs["Unit mass in cgs (U_M)"]) * 1e-3,
                l=float(f["Units"].attrs["Unit length in cgs (U_L)"]) * 1e-2,
                t=float(f["Units"].attrs["Unit time in cgs (U_t)"]),
            )

            # Particle data
            A2_pos = (
                np.array(f["PartType0/Coordinates"][()])
                - 0.5 * f["Header"].attrs["BoxSize"]
            ) * file_to_SI.l
            A1_r = np.sqrt(np.sum(A2_pos ** 2, axis=1))
            A1_rho = np.array(f["PartType0/Densities"][()]) * file_to_SI.rho

        # Plot the density profile with particles
        plot_profile_and_particles(profile, A1_r, A1_rho)
        save1 = "profiles/demo_%s_%s_%04d_density.png" % (body, N_label, snapshot_id)
        plt.savefig(save1, dpi=200)
        plt.close()
        print("Saved %s" % save1)

        # Plot the spinning profiles
        fig = plot_spinning_profiles(spin_profile)
        save2 = "profiles/demo_%s_%s_%04d_spin.png" % (body, N_label, snapshot_id)
        plt.savefig(save2, dpi=200)
        plt.close()
        print("Saved %s" % save2)