import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import h5py
import woma
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration parameters for the snapshot plotting script."""
    # Simulation parameters
    N_PARTICLES = 10 ** 5
    N_LABEL = f"n{int(10 * np.log10(10 ** 5))}"
    
    # Plotting parameters
    FONT_SIZE = 12
    PLOT_PARAMS = {
        "axes.labelsize": FONT_SIZE,
        "font.size": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "font.family": "serif",
    }
    
    # Physical constants
    R_EARTH = 6.3710e6  # m
    
    # Plot styling
    MATERIAL_COLORS = {
        "ANEOS_Fe85Si15": "orangered", 
        "ANEOS_forsterite": "gold"
    }
    POINT_SIZE = (1 * np.cbrt(10 ** 6 / N_PARTICLES)) ** 2
    
    # Data selection
    Z_SELECTION = "negative"  # Options: "negative", "midplane", "all"
    SNAPSHOT_ID = 2
    AXIS_LIMIT = 1.5  # Plot limits in Earth radii


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def setup_plotting():
    """Initialize matplotlib with custom parameters."""
    matplotlib.rcParams.update(Config.PLOT_PARAMS)


def load_snapshot_data(filename):
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
        positions = (
            np.array(f["PartType0/Coordinates"][()]) 
            - 0.5 * f["Header"].attrs["BoxSize"]
        ) * file_to_SI.l
        
        internal_energies = np.array(f["PartType0/InternalEnergies"][()]) * file_to_SI.u
        
        # Load material IDs if available
        material_ids = None
        if "PartType0/MaterialIDs" in f:
            material_ids = np.array(f["PartType0/MaterialIDs"][()])

    return positions, internal_energies, material_ids


def select_particles_by_z(positions, internal_energies, material_ids=None):
    """
    Select particles based on z-coordinate criteria.
    
    Parameters:
    -----------
    positions : np.array
        Particle positions
    internal_energies : np.array  
        Particle internal energies
    material_ids : np.array, optional
        Particle material IDs
    
    Returns:
    --------
    tuple : Filtered (positions, internal_energies, material_ids)
    """
    if Config.Z_SELECTION == "negative":
        # Select particles with z < 0 (original behavior)
        selector = positions[:, 2] < 0
    elif Config.Z_SELECTION == "midplane":
        # Select particles near midplane (|z| < 0.1 R_Earth)
        selector = np.abs(positions[:, 2]) < 0.1 * Config.R_EARTH
    elif Config.Z_SELECTION == "all":
        # Use all particles
        selector = np.ones(len(positions), dtype=bool)
    else:
        raise ValueError(f"Unknown Z_SELECTION: {Config.Z_SELECTION}")
    
    filtered_positions = positions[selector]
    filtered_energies = internal_energies[selector]
    filtered_materials = material_ids[selector] if material_ids is not None else None
    
    return filtered_positions, filtered_energies, filtered_materials


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_snapshot_internal_energy(positions, internal_energies, body_name, snapshot_id):
    """
    Plot particles colored by internal energy.
    
    Parameters:
    -----------
    positions : np.array
        Particle positions (x, y, z)
    internal_energies : np.array
        Particle internal energies
    body_name : str
        Name of the body being plotted
    snapshot_id : int
        Snapshot identifier
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal")
    
    # Create colorbar axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Create scatter plot
    scatter = ax.scatter(
        positions[:, 0] / Config.R_EARTH,
        positions[:, 1] / Config.R_EARTH,
        c=internal_energies,
        edgecolors="none",
        marker=".",
        s=Config.POINT_SIZE,
        cmap="viridis"
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label(r"Specific Internal Energy (J kg$^{-1}$)")
    
    # Set plot limits and labels
    ax.set_xlim(-Config.AXIS_LIMIT, Config.AXIS_LIMIT)
    ax.set_ylim(-Config.AXIS_LIMIT, Config.AXIS_LIMIT)
    ax.set_xlabel(r"$x$ ($R_\oplus$)")
    ax.set_ylabel(r"$y$ ($R_\oplus$)")
    ax.set_title(f"{body_name.title()} - Snapshot {snapshot_id:04d}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_snapshot_by_material(positions, material_ids, body_name, snapshot_id):
    """
    Plot particles colored by material type.
    
    Parameters:
    -----------
    positions : np.array
        Particle positions (x, y, z)
    material_ids : np.array
        Particle material IDs
    body_name : str
        Name of the body being plotted
    snapshot_id : int
        Snapshot identifier
    """
    if material_ids is None:
        print(f"  ⚠ Material IDs not available for {body_name}")
        return None
        
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal")
    
    # Create color mapping
    material_colors = {}
    for mat_name, color in Config.MATERIAL_COLORS.items():
        if mat_name in woma.Di_mat_id:
            material_colors[woma.Di_mat_id[mat_name]] = color
    
    # Assign colors to particles
    particle_colors = np.array([material_colors.get(mid, 'gray') 
                               for mid in material_ids])
    
    # Create scatter plot by material
    for material_id, color in material_colors.items():
        mask = material_ids == material_id
        if np.any(mask):
            material_name = next(name for name, mid in woma.Di_mat_id.items() 
                               if mid == material_id)
            ax.scatter(
                positions[mask, 0] / Config.R_EARTH,
                positions[mask, 1] / Config.R_EARTH,
                c=color,
                edgecolors="none",
                marker=".",
                s=Config.POINT_SIZE,
                label=material_name
            )
    
    # Set plot properties
    ax.set_xlim(-Config.AXIS_LIMIT, Config.AXIS_LIMIT)
    ax.set_ylim(-Config.AXIS_LIMIT, Config.AXIS_LIMIT)
    ax.set_xlabel(r"$x$ ($R_\oplus$)")
    ax.set_ylabel(r"$y$ ($R_\oplus$)")
    ax.set_title(f"{body_name.title()} - Materials - Snapshot {snapshot_id:04d}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def process_snapshot(body, snapshot_id, output_dir="snapshot_plots"):
    """
    Process a single snapshot and generate plots.
    
    Parameters:
    -----------
    body : str
        Body name ('target' or 'impactor')
    snapshot_id : int
        Snapshot identifier
    output_dir : str
        Directory to save output plots
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    filename = f"snapshots/demo_{body}_{Config.N_LABEL}_{snapshot_id:04d}.hdf5"
    
    try:
        # Load data
        positions, internal_energies, material_ids = load_snapshot_data(filename)
        
        # Apply z-selection
        positions, internal_energies, material_ids = select_particles_by_z(
            positions, internal_energies, material_ids
        )
        
        # Generate internal energy plot
        fig1 = plot_snapshot_internal_energy(
            positions, internal_energies, body, snapshot_id
        )
        save_path1 = f"{output_dir}/demo_{body}_{Config.N_LABEL}_{snapshot_id:04d}_energy.png"
        fig1.savefig(save_path1, dpi=200, bbox_inches='tight')
        plt.close(fig1)
        print(f"  ✓ Saved {save_path1}")
        
        # Generate material plot (if material IDs available)
        if material_ids is not None:
            fig2 = plot_snapshot_by_material(
                positions, material_ids, body, snapshot_id
            )
            if fig2 is not None:
                save_path2 = f"{output_dir}/demo_{body}_{Config.N_LABEL}_{snapshot_id:04d}_materials.png"
                fig2.savefig(save_path2, dpi=200, bbox_inches='tight')
                plt.close(fig2)
                print(f"  ✓ Saved {save_path2}")
                
    except FileNotFoundError:
        print(f"  ✗ Snapshot file not found: {filename}")
    except Exception as e:
        print(f"  ✗ Error processing {body} snapshot {snapshot_id}: {e}")


def main():
    """Main function to generate all snapshot plots."""
    setup_plotting()
    
    bodies = ["target", "impactor"]
    
    print("Generating snapshot plots...")
    print(f"Z selection mode: {Config.Z_SELECTION}")
    print(f"Snapshot ID: {Config.SNAPSHOT_ID}")
    print("-" * 50)
    
    for body in bodies:
        print(f"Processing {body}...")
        process_snapshot(body, Config.SNAPSHOT_ID)
    
    print("-" * 50)
    print("Snapshot plotting complete!")


if __name__ == "__main__":
    main()