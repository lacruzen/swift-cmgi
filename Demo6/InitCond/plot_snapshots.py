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
    """Load and convert particle data from snapshot file."""
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
        A1_rho = np.array(f["PartType0/Densities"][()]) * file_to_SI.rho
        A1_mat_id = np.array(f["PartType0/MaterialIDs"][()])
            
    return A2_pos, A1_u, A1_rho, A1_mat_id

def plot_internal_energies(A2_pos, A1_u):
    """Plot the particles, coloured by their internal energy."""
    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_aspect("equal")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    # fig.text(0.06, 0.9, '1e3', va='top', ha='left')
    # fig.text(0.72, 0.06, '1e3', va='bottom', ha='right')

    # Plot
    scat = ax.scatter(
        A2_pos[:, 0] / 1e6,
        A2_pos[:, 1] / 1e6,
        c=A1_u,
        cmap='jet',
        edgecolors="none",
        marker=".",
        s=size,
        alpha=0.8,
    )
    cbar = plt.colorbar(scat, cax=cax)
    cbar.set_label(r"Sp. Int. Energy [J kg$^{-1}$]")

    # ax_lim = 1.5
    # ax.set_xlim(-ax_lim, ax_lim)
    # ax.set_yticks(ax.get_xticks())
    # ax.set_ylim(-ax_lim, ax_lim)
    
    # Set axis limits to show -6 to +6 (thousands of km)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)

    # Set the exact tick marks you requested
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])

    # Add grid
    ax.grid(True, alpha=0.4, linestyle='-')
    
    ax.set_xlabel(r"X [km]")
    ax.set_ylabel(r"Y [km]")
    # ax.text(2, 0.5, '1e3', transform=ax.transAxes, va='bottom', ha='left')
    # ax.text(1, -0.5, '1e3', transform=ax.transAxes, va='top', ha='right')
    plt.tight_layout()

def plot_densities(A2_pos, A1_rho):
    """Plot the particles, coloured by their densities."""
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_aspect("equal")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)

    # Plot
    scat = ax.scatter(
        A2_pos[:, 0] / 1e6,
        A2_pos[:, 1] / 1e6,
        c=A1_rho / 1e3,
        cmap='jet',
        edgecolors="none",
        marker=".",
        s=size,
        alpha=0.8,
    )
    cbar = plt.colorbar(scat, cax=cax)
    cbar.set_label(r"Density [g/cm$^{3}$]")

    # ax_lim = 1.5
    # ax.set_xlim(-ax_lim, ax_lim)
    # ax.set_yticks(ax.get_xticks())
    # ax.set_ylim(-ax_lim, ax_lim)
    
    # Set axis limits to show -6 to +6 (thousands of km)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)

    # Set the exact tick marks you requested
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])

    # Add grid
    ax.grid(True, alpha=0.4, linestyle='-')
    
    ax.set_xlabel(r"X [km]")
    ax.set_ylabel(r"Y [km]") 
    plt.tight_layout()

if __name__ == "__main__":
    # Plot each snapshot
    for body in ["target", "impactor"]:
        # Load the data
        snapshot_id = 5
        A2_pos, A1_u, A1_rho, A1_mat_id = load_snapshot(
            "snapshots/cmgi_%s_%s_%04d.hdf5" % (body, N_label, snapshot_id)
        )

        # Plot the data and save the figures
        plot_internal_energies(A2_pos, A1_u)
        save = "snapshots/cmgi_%s_energy_%s_%04d.png" % (body, N_label, snapshot_id)
        plt.savefig(save, dpi=200)
        plt.close()

        plot_densities(A2_pos, A1_rho)
        save = "snapshots/cmgi_%s_density_%s_%04d.png" % (body, N_label, snapshot_id)
        plt.savefig(save, dpi=200)
        plt.close()

        print("\rSaved %s" % save)