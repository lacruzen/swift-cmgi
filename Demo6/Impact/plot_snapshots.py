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
M_E = 5.9724e24  # kg

# ======================================================================
# UPDATED TIME PARAMETERS BASED ON REVISED .YML FILE
# ======================================================================
# From your parameter file:
# TimeIntegration:
#   time_begin: 0
#   time_end: 3600           # ~1 hour simulation time (not 54000!)
# Snapshots:
#   delta_time: 10           # Output every ~10 seconds (not 2000!)

# Convert internal units to physical units:
# UnitMass_in_cgs: 5.9724e27 (Earth mass)
# UnitLength_in_cgs: 6.3710e8 (Earth radius)
# UnitVelocity_in_cgs: 1e5 (1 km/s)

# Time unit = Length/velocity = (6.3710e8 cm) / (1e5 cm/s) = 6371 seconds
TIME_UNIT_SECONDS = 6371.0  # Internal time unit in seconds

# Simulation parameters from .yml
SIMULATION_START_TIME = 0 * TIME_UNIT_SECONDS  # 0 in internal units
SIMULATION_END_TIME = 3600 * TIME_UNIT_SECONDS  # 3600 in internal units
SNAPSHOT_INTERVAL = 10 * TIME_UNIT_SECONDS     # 10 in internal units

# Calculate total number of snapshots
TOTAL_SNAPSHOTS = int((SIMULATION_END_TIME - SIMULATION_START_TIME) / SNAPSHOT_INTERVAL) + 1


def load_snapshot(filename):
    """Load and convert the particle data to plot."""
    with h5py.File(filename, "r") as f:
        # Extract simulation time from snapshot
        time = f["Header"].attrs["Time"] * TIME_UNIT_SECONDS
        
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
        
        # Load velocities for visualization
        A2_vel = np.array(f["PartType0/Velocities"][()]) * file_to_SI.v
        
        # Load densities for additional diagnostics
        A1_rho = np.array(f["PartType0/Densities"][()]) * file_to_SI.rho
        
        # Load internal energies for temperature estimation
        A1_u = np.array(f["PartType0/InternalEnergies"][()]) * file_to_SI.u

    # For impact visualization, consider all particles (not just z < 0)
    # But restrict to central region for better visualization
    dist_from_center = np.linalg.norm(A2_pos, axis=1) / R_E
    A1_sel = dist_from_center < 20  # Show particles within 20 Earth radii
    
    A2_pos = A2_pos[A1_sel]
    A1_mat_id = A1_mat_id[A1_sel]
    A2_vel = A2_vel[A1_sel]
    A1_rho = A1_rho[A1_sel]
    A1_u = A1_u[A1_sel]

    return A2_pos, A1_mat_id, A2_vel, A1_rho, A1_u, time


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
    if seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} h"


def calculate_impact_energy(masses, velocities):
    """Calculate kinetic energy of particles."""
    # Assuming equal masses for simplicity (or use actual masses if available)
    # KE = 0.5 * m * v^2
    vel_mag_sq = np.sum(velocities**2, axis=1)
    avg_mass = 0.887 * M_E / (0.887 * N)  # Approximate mass per particle for target
    KE = 0.5 * avg_mass * np.sum(vel_mag_sq)
    return KE


def plot_snapshot(A2_pos, A1_mat_id, A2_vel, A1_rho, A1_u, time, snapshot_id):
    """Plot the particles with multiple visualizations."""
    fig = plt.figure(figsize=(18, 12))
    
    # Create subplots
    ax1 = plt.subplot(2, 3, 1)  # XY position (material colors)
    ax2 = plt.subplot(2, 3, 2)  # XY position (velocity magnitude)
    ax3 = plt.subplot(2, 3, 3)  # XZ position (density)
    ax4 = plt.subplot(2, 3, 4)  # Velocity vectors
    ax5 = plt.subplot(2, 3, 5)  # Internal energy
    ax6 = plt.subplot(2, 3, 6)  # Statistics text
    
    # Time string
    time_str = format_time(time)
    
    # Earth units
    R_E = 6.3710e6  # m
    
    # ------------------------------------------------------------------
    # Plot 1: XY position colored by material
    # ------------------------------------------------------------------
    ax1.set_aspect("equal")
    A1_colour = np.empty(len(A2_pos), dtype=object)
    for id_c, c in Di_id_colour.items():
        A1_colour[A1_mat_id == id_c] = c
    
    scatter1 = ax1.scatter(
        A2_pos[:, 0] / R_E,
        A2_pos[:, 1] / R_E,
        c=A1_colour,
        edgecolors="none",
        marker=".",
        s=size,
        alpha=0.7,
    )
    
    ax_lim = 8  # Reduced from 10 for better visualization
    ax1.set_xlim(-ax_lim, ax_lim)
    ax1.set_ylim(-ax_lim, ax_lim)
    ax1.set_xlabel(r"$x$ ($R_\oplus$)")
    ax1.set_ylabel(r"$y$ ($R_\oplus$)")
    ax1.set_title("Material Distribution")
    ax1.grid(True, alpha=0.3)
    
    # Add material legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=Di_mat_colour["ANEOS_Fe85Si15"], label='Iron core'),
        Patch(facecolor=Di_mat_colour["ANEOS_forsterite"], label='Silicate mantle')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=font_size*0.7)
    
    # ------------------------------------------------------------------
    # Plot 2: XY position colored by velocity magnitude
    # ------------------------------------------------------------------
    ax2.set_aspect("equal")
    vel_mag = np.linalg.norm(A2_vel, axis=1) / 1000  # Convert to km/s
    
    scatter2 = ax2.scatter(
        A2_pos[:, 0] / R_E,
        A2_pos[:, 1] / R_E,
        c=vel_mag,
        edgecolors="none",
        marker=".",
        s=size,
        alpha=0.7,
        cmap='viridis',
        vmin=0,
        vmax=20  # 20 km/s max for color scale
    )
    
    ax2.set_xlim(-ax_lim, ax_lim)
    ax2.set_ylim(-ax_lim, ax_lim)
    ax2.set_xlabel(r"$x$ ($R_\oplus$)")
    ax2.set_ylabel(r"$y$ ($R_\oplus$)")
    ax2.set_title("Velocity Magnitude (km/s)")
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('Velocity (km/s)', fontsize=font_size*0.8)
    cbar2.ax.tick_params(labelsize=font_size*0.7)
    
    # ------------------------------------------------------------------
    # Plot 3: XZ position colored by density
    # ------------------------------------------------------------------
    ax3.set_aspect("equal")
    
    # Log scale for density (rock/iron densities are ~3000-8000 kg/m³)
    log_rho = np.log10(A1_rho)
    
    scatter3 = ax3.scatter(
        A2_pos[:, 0] / R_E,
        A2_pos[:, 2] / R_E,
        c=log_rho,
        edgecolors="none",
        marker=".",
        s=size,
        alpha=0.7,
        cmap='plasma',
        vmin=2,   # 100 kg/m³
        vmax=4.5  # ~30,000 kg/m³
    )
    
    ax3.set_xlim(-ax_lim, ax_lim)
    ax3.set_ylim(-ax_lim, ax_lim)
    ax3.set_xlabel(r"$x$ ($R_\oplus$)")
    ax3.set_ylabel(r"$z$ ($R_\oplus$)")
    ax3.set_title("Log Density (log₁₀ kg/m³)")
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
    cbar3.set_label('log₁₀(ρ) [kg/m³]', fontsize=font_size*0.8)
    cbar3.ax.tick_params(labelsize=font_size*0.7)
    
    # ------------------------------------------------------------------
    # Plot 4: Velocity vectors (subsampled for clarity)
    # ------------------------------------------------------------------
    ax4.set_aspect("equal")
    
    # Subsample for cleaner vector plot
    subsample = len(A2_pos) // 500  # Show ~500 vectors
    if subsample > 0:
        A2_pos_sub = A2_pos[::subsample]
        A2_vel_sub = A2_vel[::subsample]
        
        # Plot background particles
        ax4.scatter(
            A2_pos[:, 0] / R_E,
            A2_pos[:, 1] / R_E,
            c='lightgray',
            edgecolors="none",
            marker=".",
            s=size*0.5,
            alpha=0.3,
        )
        
        # Plot velocity vectors
        quiver = ax4.quiver(
            A2_pos_sub[:, 0] / R_E,
            A2_pos_sub[:, 1] / R_E,
            A2_vel_sub[:, 0] / 1000,  # Convert to km/s for scaling
            A2_vel_sub[:, 1] / 1000,
            color='red',
            alpha=0.7,
            scale=100,  # Adjust scale for visibility
            width=0.003,
        )
    
    ax4.set_xlim(-ax_lim, ax_lim)
    ax4.set_ylim(-ax_lim, ax_lim)
    ax4.set_xlabel(r"$x$ ($R_\oplus$)")
    ax4.set_ylabel(r"$y$ ($R_\oplus$)")
    ax4.set_title("Velocity Vectors")
    ax4.grid(True, alpha=0.3)
    
    # Add scale arrow annotation
    ax4.annotate('10 km/s', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=font_size*0.7, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8))
    
    # ------------------------------------------------------------------
    # Plot 5: Internal energy (temperature proxy)
    # ------------------------------------------------------------------
    ax5.set_aspect("equal")
    
    # Internal energy in MJ/kg (1e6 J/kg)
    u_mj_per_kg = A1_u / 1e6
    
    scatter5 = ax5.scatter(
        A2_pos[:, 0] / R_E,
        A2_pos[:, 1] / R_E,
        c=u_mj_per_kg,
        edgecolors="none",
        marker=".",
        s=size,
        alpha=0.7,
        cmap='hot',
        vmin=0,
        vmax=10  # 10 MJ/kg
    )
    
    ax5.set_xlim(-ax_lim, ax_lim)
    ax5.set_ylim(-ax_lim, ax_lim)
    ax5.set_xlabel(r"$x$ ($R_\oplus$)")
    ax5.set_ylabel(r"$y$ ($R_\oplus$)")
    ax5.set_title("Specific Internal Energy (MJ/kg)")
    ax5.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar5 = plt.colorbar(scatter5, ax=ax5, shrink=0.8)
    cbar5.set_label('u (MJ/kg)', fontsize=font_size*0.8)
    cbar5.ax.tick_params(labelsize=font_size*0.7)
    
    # ------------------------------------------------------------------
    # Plot 6: Statistics text
    # ------------------------------------------------------------------
    ax6.axis('off')
    
    # Calculate statistics
    total_particles = len(A2_pos)
    avg_vel = np.mean(vel_mag)  # Already in km/s
    max_vel = np.max(vel_mag)
    avg_density = np.mean(A1_rho) / 1000  # Convert to g/cm³
    total_ke = calculate_impact_energy(None, A2_vel)  # Approximate
    
    # Text box with statistics
    stats_text = (
        f"CMGI Impact Simulation\n"
        f"Time: {time_str}\n"
        f"Snapshot: {snapshot_id:04d}\n"
        f"\n"
        f"Statistics:\n"
        f"Total particles shown: {total_particles:,}\n"
        f"Avg velocity: {avg_vel:.1f} km/s\n"
        f"Max velocity: {max_vel:.1f} km/s\n"
        f"Avg density: {avg_density:.2f} g/cm³\n"
        f"≈ Kinetic energy: {total_ke/1e30:.2f} ×10³⁰ J\n"
        f"\n"
        f"Simulation Progress:\n"
        f"{time/SIMULATION_END_TIME*100:.1f}% complete"
    )
    
    ax6.text(0.1, 0.5, stats_text, fontsize=font_size*0.9,
             verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # ------------------------------------------------------------------
    # Main title
    # ------------------------------------------------------------------
    fig.suptitle(f'Core-Merging Giant Impact Simulation\n'
                 f'Target: 4-hour rotation | Impact: 2×v_esc | Time: {time_str}', 
                 fontsize=font_size+4, y=0.98)

    plt.tight_layout()
    return fig


def create_movie_info_file():
    """Create a text file with information for making a movie from snapshots."""
    info = f"""CMGI Impact Simulation Movie Information
===========================================
Simulation Parameters:
- Total simulation time: {format_time(SIMULATION_END_TIME)}
- Snapshot interval: {format_time(SNAPSHOT_INTERVAL)}
- Total snapshots: {TOTAL_SNAPSHOTS}
- Particles: {N:,}
- Target rotation: 4 hours
- Impact velocity: 2× escape velocity

To create a movie using ffmpeg:
ffmpeg -framerate 10 -i cmgi_plots/cmgi_impact_{N_label}_%04d.png \\
       -c:v libx264 -pix_fmt yuv420p \\
       -vf "scale=1920:-2" \\
       cmgi_impact_{N_label}.mp4

Or using ImageMagick:
convert -delay 10 -loop 0 cmgi_plots/cmgi_impact_{N_label}_*.png \\
        cmgi_impact_{N_label}.gif

Plot information:
- Each frame shows 6 panels:
  1. Material distribution (core=mantle colors)
  2. Velocity magnitude (km/s)
  3. Density distribution
  4. Velocity vectors
  5. Internal energy (MJ/kg)
  6. Simulation statistics

Created: {os.path.basename(__file__)}
"""
    
    with open("cmgi_plots/movie_info.txt", "w") as f:
        f.write(info)
    
    print("✓ Created movie information file: cmgi_plots/movie_info.txt")


if __name__ == "__main__":
    print()
    print("CMGI Impact Simulation Plotter")
    print("=" * 60)
    print(f"Particles: {N:,} ({N_label})")
    print(f"Simulation time: {format_time(SIMULATION_START_TIME)} to {format_time(SIMULATION_END_TIME)}")
    print(f"Snapshot interval: {format_time(SNAPSHOT_INTERVAL)}")
    print(f"Expected snapshots: {TOTAL_SNAPSHOTS}")
    print("=" * 60)

    # Create output directory
    if not os.path.exists("cmgi_plots/"):
        os.makedirs("cmgi_plots/")
        print("✓ Created directory: cmgi_plots/")

    # Create movie info file
    create_movie_info_file()
    
    print("\nProcessing snapshots:")
    print("-" * 60)
    
    # Process each expected snapshot
    processed_count = 0
    skipped_count = 0
    
    for snapshot_id in range(TOTAL_SNAPSHOTS):
        # Expected filename based on your parameter file
        expected_filename = f"snapshots/cmgi_impact_{snapshot_id:04d}.hdf5"
        
        # Alternative naming pattern (from your code)
        alt_filename = f"snapshots/cmgi_impact_{N_label}_{snapshot_id:04d}.hdf5"
        
        # Try both naming patterns
        filename = None
        if os.path.exists(expected_filename):
            filename = expected_filename
        elif os.path.exists(alt_filename):
            filename = alt_filename
        
        if filename:
            try:
                print(f"Processing snapshot {snapshot_id:04d}/{TOTAL_SNAPSHOTS-1:04d}...")
                
                # Load the data
                A2_pos, A1_mat_id, A2_vel, A1_rho, A1_u, time = load_snapshot(filename)

                # Plot the data
                fig = plot_snapshot(A2_pos, A1_mat_id, A2_vel, A1_rho, A1_u, time, snapshot_id)

                # Save the figure
                save_path = f"cmgi_plots/cmgi_impact_{N_label}_{snapshot_id:04d}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Print progress
                progress = (snapshot_id + 1) / TOTAL_SNAPSHOTS * 100
                print(f"  ✓ Time: {format_time(time)} | Saved: {save_path}")
                print(f"    Progress: {progress:.1f}%")
                
                processed_count += 1
                
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {e}")
                skipped_count += 1
        else:
            # Try to find any snapshot with this ID
            import glob
            possible_files = glob.glob(f"snapshots/*{snapshot_id:04d}*.hdf5")
            if possible_files:
                print(f"⚠  Snapshot {snapshot_id:04d} has non-standard name: {possible_files[0]}")
            else:
                # Only warn for first few missing snapshots
                if snapshot_id < 10:
                    print(f"⚠  Snapshot {snapshot_id:04d} not found (expected: {expected_filename})")
            skipped_count += 1
    
    print("\n" + "=" * 60)
    print("Snapshot plotting completed!")
    print(f"Processed: {processed_count} snapshots")
    print(f"Skipped: {skipped_count} snapshots")
    print(f"All plots saved to: cmgi_plots/")
    print("\nTo create a movie, run:")
    print(f"  ffmpeg -framerate 10 -i cmgi_plots/cmgi_impact_{N_label}_%04d.png \\")
    print(f"         -c:v libx264 -pix_fmt yuv420p cmgi_impact_{N_label}.mp4")