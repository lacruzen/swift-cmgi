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
        A2_vel = np.array(f["PartType0/Velocities"][()]) * file_to_SI.v

        return A2_pos, A1_m, A1_h, A1_rho, A1_P, A1_u, A1_mat_id, A2_vel


snapshot_id = 5
A2_pos_t, A1_m_t, A1_h_t, A1_rho_t, A1_P_t, A1_u_t, A1_mat_id_t, A2_vel_t = load_snapshot_data(
    "snapshots/cmgi_target_%s_%04d.hdf5" % (N_label, snapshot_id)
)
A2_pos_i, A1_m_i, A1_h_i, A1_rho_i, A1_P_i, A1_u_i, A1_mat_id_i, A2_vel_i = load_snapshot_data(
    "snapshots/cmgi_impactor_%s_%04d.hdf5" % (N_label, snapshot_id)
)

G = 6.67408e-11  # m^3 kg^-1 s^-2
# R_t = 0.96 * R_E
# R_i = 0.57 * R_E

# Calculate actual masses from particles
M_t_actual = np.sum(A1_m_t)
M_i_actual = np.sum(A1_m_i)

# Estimate radii from particle distributions
R_t = np.max(np.linalg.norm(A2_pos_t, axis=1))
R_i = np.max(np.linalg.norm(A2_pos_i, axis=1))

# Impact initial conditions (target rest frame)
v_esc = np.sqrt(2 * G * (M_t_actual + M_i_actual) / (R_t + R_i))
print(f"Escape velocity: {v_esc/1000:.1f} km/s")

# ======================================================================
def calculate_angular_momentum(positions, velocities, masses, center=None):
    """Calculate total angular momentum relative to a center."""
    if center is not None:
        positions = positions - center
    L = np.sum(np.cross(positions, velocities) * masses[:, np.newaxis], axis=0)
    return L

target_spin = woma.SpinPlanet(load_file="cmgi_target_profile.hdf5")
impactor_spin = woma.SpinPlanet(load_file="cmgi_impactor_profile.hdf5")

# Calculate expected angular momentum (already provided by SpinPlanet.L)
# I = I_MR2 * M * R_eq^2
L_t_expected = target_spin.L
L_i_expected = impactor_spin.L

# Calculate expected moment of inertia from I_MR2
I_t_expected = target_spin.I_MR2 * target_spin.M * target_spin.R_eq**2
I_i_expected = impactor_spin.I_MR2 * impactor_spin.M * impactor_spin.R_eq**2

# Convert period from hours to seconds and calculate angular velocity
hours_to_sec = 3600.0
target_omega = 2.0 * np.pi / (target_spin.period * hours_to_sec)
impactor_omega = 2.0 * np.pi / (impactor_spin.period * hours_to_sec)

# Verify: L = I * ω
L_t_verify = I_t_expected * target_omega
L_i_verify = I_i_expected * impactor_omega

print(f"\nVerification L = I*ω:")
print(f"Target: {L_t_expected:.2e} vs {L_t_verify:.2e} (ratio: {L_t_expected/L_t_verify:.3f})")
print(f"Impactor: {L_i_expected:.2e} vs {L_i_verify:.2e} (ratio: {L_i_expected/L_i_verify:.3f})")

# ======================================================================
# Calculate spin properties from loaded particles
center_t = np.average(A2_pos_t, weights=A1_m_t, axis=0)
center_i = np.average(A2_pos_i, weights=A1_m_i, axis=0)

# Calculate current angular momentum
L_t_current = calculate_angular_momentum(A2_pos_t - center_t, A2_vel_t, A1_m_t)
L_i_current = calculate_angular_momentum(A2_pos_i - center_i, A2_vel_i, A1_m_i)

# ======================================================================
# We'll apply solid-body rotation with the angular velocity from SpinPlanet
# The SpinPlanet uses z-axis as rotation axis by default

print(f"\n=== Applying Spin ===")
# Calculate bulk velocity (center of mass velocity)
v_com_t = np.average(A2_vel_t, weights=A1_m_t, axis=0)
v_com_i = np.average(A2_vel_i, weights=A1_m_i, axis=0)


# Center positions for rotation calculation
A2_pos_t_centered = A2_pos_t - center_t
A2_pos_i_centered = A2_pos_i - center_i

# Apply solid-body rotation around z-axis
rotation_axis = np.array([0.0, 0.0, 1.0])  # z-axis

# Calculate rotational velocities using the calculated angular velocity
A2_vel_rot_t = np.cross(rotation_axis * target_omega, A2_pos_t_centered)
A2_vel_rot_i = np.cross(rotation_axis * impactor_omega, A2_pos_i_centered)

# Combine rotational velocity with bulk velocity
A2_vel_t = A2_vel_rot_t + v_com_t
A2_vel_i = A2_vel_rot_i + v_com_i

# Verify corrected spin
L_t_corrected = calculate_angular_momentum(A2_pos_t_centered, A2_vel_rot_t, A1_m_t)
L_i_corrected = calculate_angular_momentum(A2_pos_i_centered, A2_vel_rot_i, A1_m_i)


# SET UP IMPACT TRAJECTORY
# ======================================================================
# Target starts at origin with its spin
A1_pos_t = np.zeros(3)  # Target at origin
A1_vel_t = np.zeros(3)  # Target's center of mass velocity

# Impactor trajectory relative to spinning target
# Collide at 0 degrees, at the mutual escape speed, start right before contact
A1_pos_i, A1_vel_i = woma.impact_pos_vel_b_v_c_r(
    b       = np.sin(0 * np.pi/180),  # Head-on collision (b=0)
    v_c     = 2 * v_esc,              # Slightly super-escape velocity
    r       = R_t + R_i,                    # Start further away for stability
    R_t     = R_t, 
    R_i     = R_i, 
    M_t     = M_t_actual,
    M_i     = M_i_actual,
)

# CENTER OF MASS FRAME
# ======================================================================
# Shift to centre-of-mass frame
A1_pos_com = (M_t_actual * A1_pos_t + M_i_actual * A1_pos_i) / (M_t_actual + M_i_actual)
A1_vel_com = (M_t_actual * A1_vel_t + M_i_actual * A1_vel_i) / (M_t_actual + M_i_actual)

A1_pos_t -= A1_pos_com
A1_vel_t -= A1_vel_com
A1_pos_i -= A1_pos_com
A1_vel_i -= A1_vel_com

# UPDATE PARTICLE POSITIONS AND VELOCITIES
# ======================================================================
# Rotate particle positions to account for body spin orientation
# Create rotation matrix around z-axis (rotation axis)
def rotation_matrix_z(angle):
    """Create a rotation matrix around the z-axis."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])

# Apply rotation to centered positions before translation
# Rotate positions around the center to account for spin orientation
# (The rotation phase can be adjusted if needed for specific impact geometry)
rotation_angle_t = 0.0  # Initial rotation phase for target (can be adjusted)
rotation_angle_i = 0.0  # Initial rotation phase for impactor (can be adjusted)

R_t = rotation_matrix_z(rotation_angle_t)
R_i = rotation_matrix_z(rotation_angle_i)

# Rotate the centered positions
A2_pos_t_rotated = (R_t @ A2_pos_t_centered.T).T
A2_pos_i_rotated = (R_i @ A2_pos_i_centered.T).T

# Rotate the rotational velocities to match the rotated positions
A2_vel_rot_t_rotated = (R_t @ A2_vel_rot_t.T).T
A2_vel_rot_i_rotated = (R_i @ A2_vel_rot_i.T).T

# Update positions: rotate then translate
A2_pos_t[:] = A2_pos_t_rotated + A1_pos_t
# Combine rotated rotational velocity with bulk velocity, then add bulk motion
A2_vel_t[:] = A2_vel_rot_t_rotated + v_com_t + A1_vel_t

A2_pos_i[:] = A2_pos_i_rotated + A1_pos_i
# Combine rotated rotational velocity with bulk velocity, then add bulk motion
A2_vel_i[:] = A2_vel_rot_i_rotated + v_com_i + A1_vel_i

## FINAL VERIFICATION
# ======================================================================
# Calculate final angular momentum in the CoM frame
L_t_final = calculate_angular_momentum(A2_pos_t - A1_pos_com, A2_vel_t, A1_m_t)
L_i_final = calculate_angular_momentum(A2_pos_i - A1_pos_com, A2_vel_i, A1_m_i)

# Calculate orbital angular momentum
orbital_angular_momentum = np.cross(A1_pos_i - A1_pos_t, A1_vel_i - A1_vel_t) * M_i_actual

print(f"\n=== Final Angular Momentum ===")
print(f"Target spin L: {np.linalg.norm(L_t_final):.2e} kg·m²/s")
print(f"Impactor spin L: {np.linalg.norm(L_i_final):.2e} kg·m²/s")
print(f"Orbital L: {np.linalg.norm(orbital_angular_momentum):.2e} kg·m²/s")
print(f"Total L: {np.linalg.norm(L_t_final + L_i_final + orbital_angular_momentum):.2e} kg·m²/s")

# SAVE THE DATA
# ======================================================================
# Combine and save the particle data
output_filename = "cmgi_impact_%s.hdf5" % N_label
with h5py.File(output_filename, "w") as f:
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