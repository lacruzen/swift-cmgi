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
size = (0.5 * np.cbrt(10 ** 5 / N)) ** 2

# Earth units
R_E = 6.3710e6  # m

# Simulation time parameters
SIMULATION_START_TIME = 0
SIMULATION_END_TIME = 54000
SNAPSHOT_INTERVAL = 2000


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


def calculate_simulation_time(snapshot_id):
    """
    Calculate simulation time for a given snapshot ID.
    
    Parameters:
    -----------
    snapshot_id : int
        Snapshot identifier (0, 1, 2, ...)
    
    Returns:
    --------
    float : Simulation time in seconds
    """
    return SIMULATION_START_TIME + (snapshot_id * SNAPSHOT_INTERVAL)


def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
    
    Returns:
    --------
    str : Formatted time string
    """
    hours = seconds / 3600
    minutes = (seconds % 3600) / 60
    secs = seconds % 60
    
    if hours >= 1:
        return f"{hours:.1f} h"
    elif minutes >= 1:
        return f"{minutes:.1f} min"
    else:
        return f"{secs:.0f} s"


def plot_snapshot(A2_pos, A1_mat_id, A2_vel, snapshot_id):
    """Plot the particles, coloured by their material."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Calculate simulation time
    sim_time = calculate_simulation_time(snapshot_id)
    time_str = format_time(sim_time)
    
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
    ax1.set_title(f"Position")
    
    # Add material legend to first subplot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=Di_mat_colour["ANEOS_Fe85Si15"], label='Iron core'),
        Patch(facecolor=Di_mat_colour["ANEOS_forsterite"], label='Silicate mantle')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=font_size*0.8)

    # Plot 2: Velocity magnitude
    ax2.set_aspect("equal")
    
    # Color by velocity magnitude
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
    
    ax2.set_xlim(-ax_lim, ax_lim)
    ax2.set_ylim(-ax_lim, ax_lim)
    ax2.set_xlabel(r"$x$ ($R_\oplus$)")
    ax2.set_ylabel(r"$y$ ($R_\oplus$)")
    ax2.set_title(f"Velocity Magnitude")
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar.set_label('Velocity (m/s)', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size*0.8)
    
    # Add progress information to the figure
    progress = (snapshot_id + 1) / 28 * 100
    fig.suptitle(f'Demo3 Impact Simulation\nSimulation Progress: {progress:.1f}% Complete\nTime: {time_str} (Snapshot {snapshot_id})', 
                 fontsize=font_size+2, y=0.98)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print()
    print("Demo Impact Simulation Plot")
    print("=" * 50)
    print(f"Simulation time: {SIMULATION_START_TIME} to {SIMULATION_END_TIME} s")
    print(f"Snapshot interval: {SNAPSHOT_INTERVAL} s")
    print(f"Total snapshots: 28")
    print("=" * 50)

    # Create output directory
    if not os.path.exists("demo_plots/"):
        os.makedirs("demo_plots/")
        print("Created directory: demo_plots/")

    # Plot each snapshot
    for snapshot_id in range(28):
        # Calculate current simulation time
        current_time = calculate_simulation_time(snapshot_id)
        
        print(f"Processing snapshot {snapshot_id:02d}/27 - Time: {format_time(current_time)}")
        
        try:
            # Load the data
            A2_pos, A1_mat_id, A2_vel = load_snapshot(
                "snapshots/demo_impact_%s_%04d.hdf5" % (N_label, snapshot_id)
            )

            # Plot the data
            fig = plot_snapshot(A2_pos, A1_mat_id, A2_vel, snapshot_id)

            # Save the figure
            save_path = "demo_plots/demo_impact_%s_%04d.png" % (N_label, snapshot_id)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

            print(f"  ✓ Saved: {save_path}")
            
        except FileNotFoundError:
            print(f"  ✗ Snapshot file not found: snapshots/demo_impact_{N_label}_{snapshot_id:04d}.hdf5")
        except Exception as e:
            print(f"  ✗ Error processing snapshot {snapshot_id}: {e}")

    print()
    print("=" * 50)
    print("Snapshot plotting completed!")
    print(f"All plots saved to: demo_plots/")