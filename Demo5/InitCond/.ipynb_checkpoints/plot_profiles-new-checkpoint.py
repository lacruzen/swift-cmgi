import os
import numpy as np
import h5py
import woma
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration parameters for the plotting script."""
    # Simulation parameters
    N_PARTICLES = 10 ** 5
    N_LABEL = f"n{int(10 * np.log10(10 ** 5))}"  # Fixed: proper calculation
    
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
    R_EARTH = 6.3710e6  # m (Earth radius)
    
    # File paths
    SNAPSHOT_ID = 2
    OUTPUT_DIR = "profiles"


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def setup_plotting():
    """Initialize matplotlib with custom parameters."""
    matplotlib.rcParams.update(Config.PLOT_PARAMS)
    Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)


def plot_density_profile_with_particles(profile, particle_radii, particle_densities, body_name):
    """
    Plot density profile with overlaid particles.
    
    Parameters:
    -----------
    profile : woma.Planet
        Planet profile object
    particle_radii : np.array
        Radial positions of particles
    particle_densities : np.array  
        Densities of particles
    body_name : str
        Name of the body for title
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Plot theoretical profile
    ax.plot(profile.A1_r / Config.R_EARTH, profile.A1_rho, 
            'b-', linewidth=2, label='Theoretical profile')
    
    # Plot particles
    ax.scatter(particle_radii / Config.R_EARTH, particle_densities, 
               c="k", marker=".", s=4, alpha=0.6, label='SPH particles')
    
    ax.set_xlim(0, None)
    ax.set_xlabel(r"Radial distance ($R_\oplus$)")
    ax.set_ylabel(r"Density (kg m$^{-3}$)")
    ax.set_title(f"Density Profile: {body_name.title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_spinning_profiles(spin_planet, body_name):
    """
    Plot spinning planet profiles with ellipses showing density structure.
    
    Parameters:
    -----------
    spin_planet : woma.SpinPlanet
        Spinning planet object
    body_name : str
        Name of the body for title
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    
    # Left panel: Radial density profiles
    _plot_radial_profiles(axes[0], spin_planet, body_name)
    
    # Right panel: Elliptical density structure
    _plot_density_ellipses(axes[1], spin_planet, body_name)
    
    plt.tight_layout()
    return fig


def _plot_radial_profiles(ax, spin_planet, body_name):
    """Plot radial density profiles for spinning planet."""
    # Plot different profile types
    ax.plot(spin_planet.planet.A1_r / Config.R_EARTH, spin_planet.planet.A1_rho, 
            'k--', label="Original spherical", alpha=0.7)
    ax.plot(spin_planet.A1_R / Config.R_EARTH, spin_planet.A1_rho, 
            'r-', label="Equatorial", linewidth=2)
    ax.plot(spin_planet.A1_Z / Config.R_EARTH, spin_planet.A1_rho, 
            'b-', label="Polar", linewidth=2)
    
    ax.set_xlabel(r"Radius ($R_\oplus$)")
    ax.set_ylabel(r"Density (kg m$^{-3}$)")
    ax.set_yscale("log")
    ax.set_xlim(0, 1.1 * spin_planet.R_eq / Config.R_EARTH)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{body_name.title()} - Radial Profiles")


def _plot_density_ellipses(ax, spin_planet, body_name):
    """Plot density structure using ellipses."""
    # Create colormap normalization
    norm = plt.Normalize(spin_planet.rho_s, spin_planet.rho_0)
    cmap = plt.get_cmap("viridis")
    
    # Plot ellipses for each density layer
    for i in range(len(spin_planet.A1_R)):
        ellipse = Ellipse(
            xy=(0, 0),
            width=2 * spin_planet.A1_R[i] / Config.R_EARTH, 
            height=2 * spin_planet.A1_Z[i] / Config.R_EARTH,
            zorder=-i,
            facecolor=cmap(norm(spin_planet.A1_rho[i])),
            edgecolor='none',
            alpha=0.8
        )
        ax.add_artist(ellipse)
        ellipse.set_clip_box(ax.bbox)
    
    ax.set_xlabel(r"Equatorial Radius ($R_\oplus$)")
    ax.set_ylabel(r"Polar Radius ($R_\oplus$)")    
    ax.set_xlim(0, 1.1 * spin_planet.R_eq / Config.R_EARTH)
    ax.set_ylim(0, 1.1 * spin_planet.R_po / Config.R_EARTH)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{body_name.title()} - Density Structure")
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(r"Density (kg m$^{-3}$)")


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_particle_data(body, snapshot_id):
    """
    Load particle data from snapshot file.
    
    Parameters:
    -----------
    body : str
        Body name ('target' or 'impactor')
    snapshot_id : int
        Snapshot identifier
    
    Returns:
    --------
    tuple : (particle_radii, particle_densities)
    """
    filename = f"snapshots/demo_{body}_{Config.N_LABEL}_{snapshot_id:04d}.hdf5"
    
    with h5py.File(filename, "r") as f:
        # Load unit conversions
        file_to_SI = woma.Conversions(
            m=float(f["Units"].attrs["Unit mass in cgs (U_M)"]) * 1e-3,
            l=float(f["Units"].attrs["Unit length in cgs (U_L)"]) * 1e-2,
            t=float(f["Units"].attrs["Unit time in cgs (U_t)"]),
        )

        # Load particle data
        positions = (
            np.array(f["PartType0/Coordinates"][()]) - 0.5 * f["Header"].attrs["BoxSize"]
        ) * file_to_SI.l
        
        densities = np.array(f["PartType0/Densities"][()]) * file_to_SI.rho
        
        # Calculate radial distances
        radii = np.sqrt(np.sum(positions ** 2, axis=1))
    
    return radii, densities


def load_profiles(body):
    """
    Load planet and spin profiles.
    
    Parameters:
    -----------
    body : str
        Body name ('target' or 'impactor')
    
    Returns:
    --------
    tuple : (planet_profile, spin_profile)
    """
    planet_profile = woma.Planet(load_file=f"demo_{body}_profile.hdf5")
    spin_profile = woma.SpinPlanet(load_file=f"demo_{body}_spin_profile.hdf5")
    
    return planet_profile, spin_profile


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main function to generate all plots."""
    setup_plotting()
    
    bodies = ["target", "impactor"]
    
    for body in bodies:
        print(f"Processing {body}...")
        
        try:
            # Load profiles
            planet_profile, spin_profile = load_profiles(body)
            
            # Load particle data
            particle_radii, particle_densities = load_particle_data(body, Config.SNAPSHOT_ID)
            
            # Generate and save density profile plot
            fig1 = plot_density_profile_with_particles(
                planet_profile, particle_radii, particle_densities, body
            )
            save_path1 = f"{Config.OUTPUT_DIR}/demo_{body}_{Config.N_LABEL}_{Config.SNAPSHOT_ID:04d}_density.png"
            fig1.savefig(save_path1, dpi=200, bbox_inches='tight')
            plt.close(fig1)
            print(f"  ✓ Saved {save_path1}")
            
            # Generate and save spin profile plot
            fig2 = plot_spinning_profiles(spin_profile, body)
            save_path2 = f"{Config.OUTPUT_DIR}/demo_{body}_{Config.N_LABEL}_{Config.SNAPSHOT_ID:04d}_spin.png"
            fig2.savefig(save_path2, dpi=200, bbox_inches='tight')
            plt.close(fig2)
            print(f"  ✓ Saved {save_path2}")
            
        except FileNotFoundError as e:
            print(f"  ✗ Error loading files for {body}: {e}")
        except Exception as e:
            print(f"  ✗ Unexpected error processing {body}: {e}")
    
    print("Plotting complete!")


if __name__ == "__main__":
    main()