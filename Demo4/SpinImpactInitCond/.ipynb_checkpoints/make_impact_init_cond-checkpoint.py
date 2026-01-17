import numpy as np
import h5py
import woma

# Number of particles
N = 10 ** 5
N_label = "n%d" % (10 * np.log10(N))

# Earth units
M_E = 5.9724e24  # kg
R_E = 6.3710e6  # m

# Set profile inputs
# M_t = 0.887 * M_E
# M_i = 0.133 * M_E

# Load the settled particle planets
def load_snapshot_data(filename):
    with h5py.File(filename, "r") as f:
        # Units from file metadata
        file_to_SI = woma.Conversions(
            m=float(f["Units"].attrs["Unit mass in cgs (U_M)"]) * 1e-3,
            l=float(f["Units"].attrs["Unit length in cgs (U_L)"]) * 1e-2,
            t=float(f["Units"].attrs["Unit time in cgs (U_t)"]),
        )

        # Particle data (converted to SI)
        A2_pos = (
            np.array(f["PartType0/Coordinates"][()])
            - 0.5 * f["Header"].attrs["BoxSize"]
        ) * file_to_SI.l
        A1_m = np.array(f["PartType0/Masses"][()]) * file_to_SI.m
        A1_h = np.array(f["PartType0/SmoothingLengths"][()]) * file_to_SI.l
        A1_rho = np.array(f["PartType0/Densities"][()]) * file_to_SI.rho
        A1_P = np.array(f["PartType0/Pressures"][()]) * file_to_SI.P
        A1_u = np.array(f["PartType0/InternalEnergies"][()]) * file_to_SI.u
        A1_mat_id = np.array(f["PartType0/MaterialIDs"][()])
        A2_vel = np.array(f["PartType0/Velocities"][()]) * file_to_SI.v  # NEW: Load velocities

        return A2_pos, A1_m, A1_h, A1_rho, A1_P, A1_u, A1_mat_id, A2_vel


snapshot_id = 5
A2_pos_t, A1_m_t, A1_h_t, A1_rho_t, A1_P_t, A1_u_t, A1_mat_id_t, A2_vel_t = load_snapshot_data(
    "snapshots/demo_target_%s_%04d.hdf5" % (N_label, snapshot_id)
)
A2_pos_i, A1_m_i, A1_h_i, A1_rho_i, A1_P_i, A1_u_i, A1_mat_id_i, A2_vel_i = load_snapshot_data(
    "snapshots/demo_impactor_%s_%04d.hdf5" % (N_label, snapshot_id)
)

G = 6.67408e-11  # m^3 kg^-1 s^-2
R_t = 0.96 * R_E
R_i = 0.57 * R_E

# Calculate actual masses from particles
M_t_actual = np.sum(A1_m_t)
M_i_actual = np.sum(A1_m_i)
print(f"\nMass summary:")
print(f"  Target mass: {M_t_actual:.2e} kg ({M_t_actual/M_E:.3f} M_E)")
print(f"  Impactor mass: {M_i_actual:.2e} kg ({M_i_actual/M_E:.3f} M_E)")

# Estimate radii from particle distributions
R_t = np.max(np.linalg.norm(A2_pos_t, axis=1))
R_i = np.max(np.linalg.norm(A2_pos_i, axis=1))
print(f"Radius summary:")
print(f"  Target radius: {R_t:.2e} m ({R_t/R_E:.3f} R_E)")
print(f"  Impactor radius: {R_i:.2e} m ({R_i/R_E:.3f} R_E)")

# Calculate angular momentum of spinning target
def calculate_angular_momentum(positions, velocities, masses, reference_point):
    """Calculate angular momentum relative to reference point."""
    positions_rel = positions - reference_point
    angular_momentum = np.sum(np.cross(positions_rel, masses[:, np.newaxis] * velocities), axis=0)
    return angular_momentum

target_center = np.sum(A2_pos_t * A1_m_t[:, np.newaxis], axis=0) / M_t_actual
target_angular_momentum = calculate_angular_momentum(A2_pos_t, A2_vel_t, A1_m_t, target_center)
print(f"\nTarget spin properties:")
print(f"  Angular momentum: {target_angular_momentum} kg·m²/s")
print(f"  Angular momentum magnitude: {np.linalg.norm(target_angular_momentum):.2e} kg·m²/s")

# Impact initial conditions (target rest frame)
# Collide at 20 degrees, at the mutual escape speed, start right before contact
G = 6.67408e-11  # m^3 kg^-1 s^-2
v_esc = np.sqrt(2 * G * (M_t_actual + M_i_actual) / (R_t + R_i))
print(f"Escape velocity at contact: {v_esc:.2f} m/s")

# Calculate impact trajectory (target starts at origin with its spin)
A1_pos_t, A1_vel_t = np.zeros(3), np.zeros(3)  # Target at rest at origin
A1_pos_i, A1_vel_i = woma.impact_pos_vel_b_v_c_r(
    b       = np.sin(20 * np.pi/180), 
    v_c     = 2 * v_esc,  # Slightly super-escape velocity
    r       = R_t + R_i,  # Start further away for stability
    R_t     = R_t, 
    R_i     = R_i, 
    M_t     = M_t_actual,
    M_i     = M_i_actual,
)

print(f"\nImpact configuration:")
print(f"  Target position: {A1_pos_t / R_E} R_E")
print(f"  Impactor position: {A1_pos_i / R_E} R_E")
print(f"  Target velocity: {A1_vel_t} m/s")
print(f"  Impactor velocity: {A1_vel_i} m/s")
print(f"  Relative velocity: {np.linalg.norm(A1_vel_i - A1_vel_t):.2f} m/s")

# Shift to centre-of-mass frame
A1_pos_com = (M_t_actual * A1_pos_t + M_i_actual * A1_pos_i) / (M_t_actual + M_i_actual)
A1_vel_com = (M_t_actual * A1_vel_t + M_i_actual * A1_vel_i) / (M_t_actual + M_i_actual)

A1_pos_t -= A1_pos_com
A1_vel_t -= A1_vel_com
A1_pos_i -= A1_pos_com
A1_vel_i -= A1_vel_com

print(f"COM frame adjustment:")
print(f"  COM position: {A1_pos_com / R_E} R_E")
print(f"  COM velocity: {A1_vel_com} m/s")

# Update particle positions and velocities
print(f"\nUpdating particle positions...")
A2_pos_t[:] += A1_pos_t
# Target velocities already contain spin, just add bulk motion
A2_vel_t[:] += A1_vel_t

A2_pos_i[:] += A1_pos_i
A2_vel_i[:] += A1_vel_i  # Impactor gets bulk motion

# Final angular momentum check
final_target_angular_momentum = calculate_angular_momentum(A2_pos_t, A2_vel_t, A1_m_t, A1_pos_com)
print(f"Final target angular momentum: {np.linalg.norm(final_target_angular_momentum):.2e} kg·m²/s")

# Combine and save the particle data
print(f"\nSaving impact initial conditions...")
with h5py.File("demo_impact_%s.hdf5" % N_label, "w") as f:
    woma.save_particle_data(
        f,
        A2_pos=np.append(A2_pos_t, A2_pos_i, axis=0),
        A2_vel=np.append(A2_vel_t, A2_vel_i, axis=0),
        A1_m=np.append(A1_m_t, A1_m_i),
        A1_h=np.append(A1_h_t, A1_h_i),
        A1_rho=np.append(A1_rho_t, A1_rho_i),
        A1_P=np.append(A1_P_t, A1_P_i),
        A1_u=np.append(A1_u_t, A1_u_i),
        A1_mat_id=np.append(A1_mat_id_t, A1_mat_id_i),
        boxsize=100 * R_E,
        file_to_SI=woma.Conversions(m=1e24, l=1e6, t=1),
    )

print(f"✅ Saved: demo_impact_{N_label}.hdf5")