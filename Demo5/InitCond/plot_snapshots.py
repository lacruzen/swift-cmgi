import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import h5py
import woma
from pathlib import Path

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

# Material colours
Di_mat_colour = {"ANEOS_Fe85Si15": "orangered", "ANEOS_forsterite": "gold"}
Di_id_colour = {woma.Di_mat_id[mat]: colour for mat, colour in Di_mat_colour.items()}

# Scale point size with resolution
size = (1 * np.cbrt(10 ** 6 / N)) ** 2

# Earth units
R_E = 6.3710e6  # m


def load_snapshot(filename):
    """
    Load and convert particle data from snapshot file.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 snapshot file
    
    Returns:
    --------
    tuple : (positions, internal_energies, material_ids)
        positions: np.array of shape (n_particles, 3)
        internal_energies: np.array of shape (n_particles,)
        material_ids: np.array of shape (n_particles,)
    """
    with h5py.File(filename, "r") as f:
        # Load unit conversions
        file_to_SI = woma.Conversions(
            m=float(f["Units"].attrs["Unit mass in cgs (U_M)"]) * 1e-3,
            l=float(f["Units"].attrs["Unit length in cgs (U_L)"]) * 1e-2,
            t=float(f["Units"].attrs["Unit time in cgs (U_t)"]),
        )

        # Load particle data
        A2_pos = (
            np.array(f["PartType0/Coordinates"][()])
            - 0.5 * f["Header"].attrs["BoxSize"]
        ) * file_to_SI.l
        A1_u = np.array(f["PartType0/InternalEnergies"][()]) * file_to_SI.u
        
    return A2_pos, A1_u


def plot_snapshot(A2_pos, A1_u):
    """Plot the particles, coloured by their internal energy."""
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_aspect("equal")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)

    # Plot
    scat = ax.scatter(
        A2_pos[:, 0] / R_E,
        A2_pos[:, 1] / R_E,
        c=A1_u,
        edgecolors="none",
        marker=".",
        s=size,
    )
    cbar = plt.colorbar(scat, cax=cax)
    cbar.set_label(r"Sp. Int. Energy (J kg$^{-1}$)")

    ax_lim = 1.5
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_yticks(ax.get_xticks())
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_xlabel(r"$x$ ($R_\oplus$)")
    ax.set_ylabel(r"$y$ ($R_\oplus$)")

    plt.tight_layout()


if __name__ == "__main__":
    # Plot each snapshot
    for body in ["target", "impactor"]:
        # Load the data
        snapshot_id = 5
        A2_pos, A1_u = load_snapshot(
            "snapshots/demo_%s_%s_%04d.hdf5" % (body, N_label, snapshot_id)
        )

        # Plot the data
        plot_snapshot(A2_pos, A1_u)

        # Save the figure
        save = "snapshots/demo_%s_%s_%04d.png" % (body, N_label, snapshot_id)
        plt.savefig(save, dpi=200)
        plt.close()

        print("\rSaved %s" % save)