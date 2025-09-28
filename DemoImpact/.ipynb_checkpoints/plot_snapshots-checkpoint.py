import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
import woma

# Number of particles
N = 10 ** 5
N_label = "n%d" % (10 * np.log10(N))

# Plotting options
font_size = 20
params = {
    "axes.labelsize": font_size,
    "font.size": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "font.family": "serif",
}
matplotlib.rcParams.update(params)

# Material colours
Di_mat_colour = {"ANEOS_Fe85Si15": "darkgray", "ANEOS_forsterite": "orangered"}
Di_id_colour = {woma.Di_mat_id[mat]: colour for mat, colour in Di_mat_colour.items()}

# Scale point size with resolution
size = (0.5 * np.cbrt(10 ** 6 / N)) ** 2

# Earth units
R_E = 6.3710e6  # m


def load_snapshot(filename):
    """Load and convert the particle data to plot."""
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
        A1_mat_id = np.array(f["PartType0/MaterialIDs"][()])
        
        # NEW: Load velocities for spin visualization
        A2_vel = np.array(f["PartType0/Velocities"][()]) * file_to_SI.v

    # Restrict to z < 0 for plotting (consider modifying this for spin)
    A1_sel = np.where(A2_pos[:, 2] < 0)[0]
    A2_pos = A2_pos[A1_sel]
    A1_mat_id = A1_mat_id[A1_sel]
    A2_vel = A2_vel[A1_sel]  # NEW

    return A2_pos, A1_mat_id, A2_vel  # MODIFIED: Return velocities


def plot_snapshot(A2_pos, A1_mat_id, A2_vel, snapshot_id):
    """Plot the particles, coloured by their material."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))  # MODIFIED: Two subplots
    
    # Earth units
    R_E = 6.3710e6  # m

    # Colour by material
    A1_colour = np.empty(len(A2_pos), dtype=object)
    for id_c, c in Di_id_colour.items():
        A1_colour[A1_mat_id == id_c] = c

    # Plot 1: XY position (original plot)
    ax1.set_aspect("equal")
    ax1.scatter(
        A2_pos[:, 0] / R_E,
        A2_pos[:, 1] / R_E,
        c=A1_colour,
        edgecolors="none",
        marker=".",
        s=size,
        alpha=0.5,
    )
    ax_lim = 10
    ax1.set_xlim(-ax_lim, ax_lim)
    ax1.set_ylim(-ax_lim, ax_lim)
    ax1.set_xlabel(r"$x$ ($R_\oplus$)")
    ax1.set_ylabel(r"$y$ ($R_\oplus$)")
    ax1.set_title("Position")

    # NEW: Plot 2: Velocity magnitude or specific angular momentum
    ax2.set_aspect("equal")
    
    # Option A: Color by velocity magnitude
    vel_mag = np.linalg.norm(A2_vel, axis=1)
    scatter2 = ax2.scatter(
        A2_pos[:, 0] / R_E,
        A2_pos[:, 1] / R_E,
        c=vel_mag,
        edgecolors="none",
        marker=".",
        s=size,
        alpha=0.5,
        cmap='viridis'
    )
    
    # Option B: Color by specific angular momentum (better for spin visualization)
    # A2_ang_mom = np.cross(A2_pos, A2_vel)
    # ang_mom_z = A2_ang_mom[:, 2]
    # scatter2 = ax2.scatter(
    #     A2_pos[:, 0] / R_E,
    #     A2_pos[:, 1] / R_E,
    #     c=ang_mom_z,
    #     edgecolors="none", 
    #     marker=".",
    #     s=size,
    #     alpha=0.5,
    #     cmap='RdBu_r'
    # )
    
    ax2.set_xlim(-ax_lim, ax_lim)
    ax2.set_ylim(-ax_lim, ax_lim)
    ax2.set_xlabel(r"$x$ ($R_\oplus$)")
    ax2.set_ylabel(r"$y$ ($R_\oplus$)")
    ax2.set_title("Velocity Magnitude")
    plt.colorbar(scatter2, ax=ax2, label='Velocity (m/s)')

    plt.tight_layout()


def plot_spin_profile_comparison(snapshot_id):
    """NEW: Compare with original spin profiles if available"""
    try:
        # Try to load the original spin profiles for comparison
        target_spin = woma.SpinPlanet(load_file="demo_target_spin_profile.hdf5")
        impactor_spin = woma.SpinPlanet(load_file="demo_impactor_spin_profile.hdf5")
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot equatorial profiles
        ax[0].plot(target_spin.A1_R / R_E, target_spin.A1_rho, 'b-', label='Target (original)')
        ax[0].plot(impactor_spin.A1_R / R_E, impactor_spin.A1_rho, 'r-', label='Impactor (original)')
        ax[0].set_xlabel(r"Equatorial Radius ($R_\oplus$)")
        ax[0].set_ylabel(r"Density (kg m$^{-3}$)")
        ax[0].set_yscale('log')
        ax[0].legend()
        
        # Plot polar profiles  
        ax[1].plot(target_spin.A1_Z / R_E, target_spin.A1_rho, 'b-', label='Target (original)')
        ax[1].plot(impactor_spin.A1_Z / R_E, impactor_spin.A1_rho, 'r-', label='Impactor (original)')
        ax[1].set_xlabel(r"Polar Radius ($R_\oplus$)")
        ax[1].set_ylabel(r"Density (kg m$^{-3}$)")
        ax[1].set_yscale('log')
        ax[1].legend()
        
        plt.tight_layout()
        
        # Save spin profile comparison
        if not os.path.exists("plots/"):
            os.makedirs("plots/")
        save = "plots/demo_impact_%s_%04d_spin_profiles.png" % (N_label, snapshot_id)
        plt.savefig(save, dpi=200)
        plt.close()
        print("Saved spin profile comparison: %s" % save)
        
    except Exception as e:
        print("Could not load spin profiles for comparison: %s" % str(e))


if __name__ == "__main__":
    print()

    # Plot each snapshot
    for snapshot_id in range(28):
        # Load the data
        A2_pos, A1_mat_id, A2_vel = load_snapshot(  # MODIFIED
            "snapshots/demo_impact_%s_%04d.hdf5" % (N_label, snapshot_id)
        )

        # Plot the data
        plot_snapshot(A2_pos, A1_mat_id, A2_vel, snapshot_id)  # MODIFIED

        # Save the figure
        if not os.path.exists("plots/"):
            os.makedirs("plots/")
        save = "plots/demo_impact_%s_%04d.png" % (N_label, snapshot_id)
        plt.savefig(save, dpi=200)
        plt.close()

        # NEW: Plot spin profile comparison
        plot_spin_profile_comparison(snapshot_id)

        print("\rProcessed snapshot %d/27" % snapshot_id, end="")

    print()