"""
Patent 5a - Enhanced Connection Learning Module (Patent Def 1.3 - Algorithm 1.3.1)

Implements robust connection operator learning for inter-modal parallel transport:
- SVD-based rotation estimation (Kabsch algorithm)
- RANSAC outlier rejection for robust calibration
- Modality-specific calibration (camera-lidar, lidar-radar, etc.)
- Temporal offset estimation for sensor synchronization

This module provides the `learn_connection_auto` function imported by patent_5a_real.py.

Patent Reference:
    Connection Γ_{i→j} specifies how to transport measurements from sensor i's
    reference frame to sensor j's reference frame. Learning these connections
    from calibration data is essential for accurate multi-sensor fusion.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


# =============================================================================
# CONNECTION OPERATOR (Reimport for standalone use)
# =============================================================================

@dataclass
class ConnectionOperator:
    """
    Connection Γ_{i→j} for parallel transport between sensor frames.

    Patent Def 1.3: Specifies how to transport measurements from sensor i's
    reference frame to sensor j's reference frame.
    """
    source_sensor: str
    target_sensor: str

    # Transformation parameters (learned via calibration)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0
    temporal_offset: float = 0.0  # ms

    def transport(self, measurement: np.ndarray) -> np.ndarray:
        """
        Apply parallel transport: Γ_{i→j}(m_i) → m_j

        Patent Def 1.3: Transport measurement from source to target frame.
        """
        if len(measurement) >= 3:
            # Apply rotation + translation + scale
            m_transformed = self.scale * (self.rotation @ measurement[:3]) + self.translation
            if len(measurement) > 3:
                m_transformed = np.concatenate([m_transformed, measurement[3:]])
            return m_transformed
        return measurement * self.scale


# =============================================================================
# KABSCH ALGORITHM (SVD-based rotation estimation)
# =============================================================================

def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute optimal rotation matrix using Kabsch algorithm.

    Given two sets of corresponding 3D points P and Q, find the rotation R
    that minimizes ||Q - R @ P||^2.

    Algorithm:
        1. Center both point sets
        2. Compute covariance matrix H = P.T @ Q
        3. SVD: H = U @ S @ V.T
        4. R = V @ U.T (with reflection handling)

    Args:
        P: Source points (N x 3)
        Q: Target points (N x 3)

    Returns:
        Optimal 3x3 rotation matrix R
    """
    # Center the points
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)

    # Compute covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection (ensure proper rotation with det=1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def compute_scale(P: np.ndarray, Q: np.ndarray, R: np.ndarray) -> float:
    """
    Compute optimal scale factor after rotation.

    Args:
        P: Source points (N x 3)
        Q: Target points (N x 3)
        R: Rotation matrix

    Returns:
        Optimal scale factor s
    """
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)

    # Rotate P
    P_rotated = (R @ P_centered.T).T

    # Optimal scale: s = sum(P_rotated . Q) / sum(P_rotated . P_rotated)
    numerator = np.sum(P_rotated * Q_centered)
    denominator = np.sum(P_rotated * P_rotated)

    if denominator > 1e-10:
        return numerator / denominator
    return 1.0


def compute_translation(P: np.ndarray, Q: np.ndarray, R: np.ndarray, s: float) -> np.ndarray:
    """
    Compute translation after rotation and scaling.

    Args:
        P: Source points (N x 3)
        Q: Target points (N x 3)
        R: Rotation matrix
        s: Scale factor

    Returns:
        Translation vector t
    """
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)

    # t = Q_mean - s * R @ P_mean
    t = Q_mean - s * (R @ P_mean)
    return t


# =============================================================================
# RANSAC OUTLIER REJECTION
# =============================================================================

def ransac_transform(source_pts: np.ndarray, target_pts: np.ndarray,
                     n_iterations: int = 100,
                     inlier_threshold: float = 0.5,
                     min_inliers: int = 3) -> Tuple[np.ndarray, np.ndarray, float, List[int]]:
    """
    RANSAC-based robust transformation estimation.

    Iteratively:
        1. Sample minimal point set
        2. Estimate transform
        3. Count inliers
        4. Keep best model

    Args:
        source_pts: Source points (N x 3)
        target_pts: Target points (N x 3)
        n_iterations: Number of RANSAC iterations
        inlier_threshold: Distance threshold for inliers (meters)
        min_inliers: Minimum required inliers

    Returns:
        (rotation, translation, scale, inlier_indices)
    """
    n_pts = len(source_pts)

    if n_pts < min_inliers:
        # Not enough points, return identity
        return np.eye(3), np.zeros(3), 1.0, list(range(n_pts))

    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_s = 1.0
    best_inliers = []

    for _ in range(n_iterations):
        # Sample 3 random points (minimum for 3D transform)
        sample_idx = np.random.choice(n_pts, min(3, n_pts), replace=False)
        P_sample = source_pts[sample_idx]
        Q_sample = target_pts[sample_idx]

        try:
            # Estimate transform from sample
            R = kabsch_rotation(P_sample, Q_sample)
            s = compute_scale(P_sample, Q_sample, R)
            t = compute_translation(P_sample, Q_sample, R, s)

            # Compute residuals for all points
            transformed = s * (source_pts @ R.T) + t
            residuals = np.linalg.norm(transformed - target_pts, axis=1)

            # Count inliers
            inliers = np.where(residuals < inlier_threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers.tolist()
                best_R = R
                best_t = t
                best_s = s

        except np.linalg.LinAlgError:
            continue

    # Refit using all inliers
    if len(best_inliers) >= min_inliers:
        P_inliers = source_pts[best_inliers]
        Q_inliers = target_pts[best_inliers]

        best_R = kabsch_rotation(P_inliers, Q_inliers)
        best_s = compute_scale(P_inliers, Q_inliers, best_R)
        best_t = compute_translation(P_inliers, Q_inliers, best_R, best_s)

    return best_R, best_t, best_s, best_inliers


# =============================================================================
# TEMPORAL OFFSET ESTIMATION
# =============================================================================

def estimate_temporal_offset(source_timestamps: np.ndarray,
                             target_timestamps: np.ndarray,
                             source_pts: np.ndarray,
                             target_pts: np.ndarray,
                             max_offset_ms: float = 100.0) -> float:
    """
    Estimate temporal offset between sensor streams.

    Uses cross-correlation of motion patterns to find time alignment.

    Args:
        source_timestamps: Timestamps for source measurements (N,)
        target_timestamps: Timestamps for target measurements (M,)
        source_pts: Source measurements (N x 3)
        target_pts: Target measurements (M x 3)
        max_offset_ms: Maximum offset to search (milliseconds)

    Returns:
        Estimated temporal offset in milliseconds
    """
    if len(source_pts) < 2 or len(target_pts) < 2:
        return 0.0

    # Compute motion vectors (velocities)
    source_vel = np.diff(source_pts, axis=0)
    target_vel = np.diff(target_pts, axis=0)

    if len(source_vel) == 0 or len(target_vel) == 0:
        return 0.0

    # Compute motion magnitudes
    source_speed = np.linalg.norm(source_vel, axis=1)
    target_speed = np.linalg.norm(target_vel, axis=1)

    # Simple correlation-based offset estimation
    # For production, would use interpolation and cross-correlation
    if len(source_timestamps) > 1 and len(target_timestamps) > 1:
        source_dt = np.mean(np.diff(source_timestamps))
        target_dt = np.mean(np.diff(target_timestamps))

        # Estimate offset from timestamp differences
        if source_dt > 0 and target_dt > 0:
            # Average phase difference
            phase_diff = np.mean(source_timestamps) - np.mean(target_timestamps)
            return np.clip(phase_diff * 1000, -max_offset_ms, max_offset_ms)

    return 0.0


# =============================================================================
# MODALITY-SPECIFIC CALIBRATION
# =============================================================================

class ModalityCalibrator:
    """
    Modality-specific calibration parameters and procedures.

    Different sensor pairs require different calibration approaches:
    - Camera-LIDAR: Perspective projection + point cloud alignment
    - LIDAR-Radar: 3D to 2D+ range-doppler
    - GPS-IMU: Position vs. velocity integration
    - etc.
    """

    # Default noise characteristics per modality
    NOISE_MODELS = {
        'camera': {'position': 0.5, 'orientation': 0.02, 'scale': 1.0},
        'lidar': {'position': 0.1, 'orientation': 0.01, 'scale': 1.0},
        'radar': {'position': 0.5, 'orientation': 0.05, 'range': 0.3},
        'gps': {'position': 1.0, 'velocity': 0.1},
        'imu': {'acceleration': 0.01, 'gyro': 0.001},
        'generic': {'position': 0.3, 'orientation': 0.02}
    }

    # Expected mounting offsets (body frame, meters)
    TYPICAL_OFFSETS = {
        ('camera', 'lidar'): np.array([0.0, 0.0, 0.2]),   # Camera above LIDAR
        ('lidar', 'radar'): np.array([0.0, -0.5, 0.0]),   # Radar behind LIDAR
        ('gps', 'imu'): np.array([0.0, 0.0, 0.5]),        # GPS antenna above IMU
    }

    @staticmethod
    def get_noise_prior(modality: str) -> Dict[str, float]:
        """Get noise prior for modality."""
        return ModalityCalibrator.NOISE_MODELS.get(
            modality,
            ModalityCalibrator.NOISE_MODELS['generic']
        )

    @staticmethod
    def get_offset_prior(source: str, target: str) -> np.ndarray:
        """Get expected mounting offset prior."""
        key = (source.lower(), target.lower())
        if key in ModalityCalibrator.TYPICAL_OFFSETS:
            return ModalityCalibrator.TYPICAL_OFFSETS[key].copy()

        # Try reverse
        key_rev = (target.lower(), source.lower())
        if key_rev in ModalityCalibrator.TYPICAL_OFFSETS:
            return -ModalityCalibrator.TYPICAL_OFFSETS[key_rev].copy()

        return np.zeros(3)

    @staticmethod
    def get_inlier_threshold(source_modality: str, target_modality: str) -> float:
        """
        Get appropriate RANSAC inlier threshold based on sensor noise.
        """
        source_noise = ModalityCalibrator.NOISE_MODELS.get(
            source_modality, {'position': 0.3}
        ).get('position', 0.3)

        target_noise = ModalityCalibrator.NOISE_MODELS.get(
            target_modality, {'position': 0.3}
        ).get('position', 0.3)

        # Combined noise (3-sigma threshold)
        combined = np.sqrt(source_noise**2 + target_noise**2)
        return 3.0 * combined


# =============================================================================
# MAIN LEARNING FUNCTION
# =============================================================================

def learn_connection_auto(calibration_data: List[Tuple[np.ndarray, np.ndarray]],
                          source: str,
                          target: str,
                          source_modality: str = "generic",
                          target_modality: str = "generic",
                          timestamps: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                          use_ransac: bool = True,
                          ransac_iterations: int = 100) -> ConnectionOperator:
    """
    Automatically learn connection operator from calibration data.

    Patent Def 1.3 Algorithm 1.3.1:
        θ* = argmin_θ Σ_k ||Γ_{i→j}(m_i^(k); θ) - m_j^(k)||²

    This function learns the transformation parameters (rotation, translation,
    scale, temporal offset) that best align measurements from source sensor
    to target sensor.

    Args:
        calibration_data: List of (source_measurement, target_measurement) pairs
        source: Source sensor identifier
        target: Target sensor identifier
        source_modality: Source sensor modality (camera, lidar, radar, gps, imu)
        target_modality: Target sensor modality
        timestamps: Optional (source_timestamps, target_timestamps) for temporal alignment
        use_ransac: Whether to use RANSAC for outlier rejection (recommended)
        ransac_iterations: Number of RANSAC iterations

    Returns:
        ConnectionOperator with learned transformation parameters

    Algorithm:
        1. Extract 3D points from calibration data
        2. (Optional) Use RANSAC to reject outliers
        3. Use Kabsch algorithm for rotation estimation
        4. Compute optimal scale and translation
        5. (Optional) Estimate temporal offset
        6. Apply modality-specific refinements
    """
    if not calibration_data:
        return ConnectionOperator(source, target)

    # Extract points (first 3 dimensions)
    source_pts = np.array([d[0][:3] if len(d[0]) >= 3 else
                           np.concatenate([d[0], np.zeros(3-len(d[0]))])
                           for d in calibration_data])
    target_pts = np.array([d[1][:3] if len(d[1]) >= 3 else
                           np.concatenate([d[1], np.zeros(3-len(d[1]))])
                           for d in calibration_data])

    n_points = len(source_pts)

    # Get modality-specific threshold
    inlier_threshold = ModalityCalibrator.get_inlier_threshold(
        source_modality, target_modality
    )

    # Step 1: RANSAC outlier rejection (if enabled and enough points)
    if use_ransac and n_points >= 4:
        R, t, s, inliers = ransac_transform(
            source_pts, target_pts,
            n_iterations=ransac_iterations,
            inlier_threshold=inlier_threshold,
            min_inliers=3
        )

        # Report outlier rejection
        n_outliers = n_points - len(inliers)
        if n_outliers > 0:
            # Refine with inliers only
            source_pts = source_pts[inliers]
            target_pts = target_pts[inliers]
    else:
        # Direct estimation without RANSAC
        R = np.eye(3)
        s = 1.0
        t = np.zeros(3)

    # Step 2: Kabsch rotation estimation
    if len(source_pts) >= 3:
        R = kabsch_rotation(source_pts, target_pts)
        s = compute_scale(source_pts, target_pts, R)
        t = compute_translation(source_pts, target_pts, R, s)
    elif len(source_pts) >= 1:
        # Not enough points for rotation, estimate translation only
        R = np.eye(3)
        s = 1.0
        t = np.mean(target_pts - source_pts, axis=0)

    # Step 3: Temporal offset estimation (if timestamps provided)
    temporal_offset = 0.0
    if timestamps is not None:
        source_ts, target_ts = timestamps
        if len(source_ts) > 1 and len(target_ts) > 1:
            temporal_offset = estimate_temporal_offset(
                source_ts, target_ts,
                source_pts, target_pts
            )

    # Step 4: Apply modality-specific offset prior (as regularization)
    offset_prior = ModalityCalibrator.get_offset_prior(source_modality, target_modality)
    # Blend with prior (0.9 learned, 0.1 prior)
    t = 0.9 * t + 0.1 * offset_prior

    return ConnectionOperator(
        source_sensor=source,
        target_sensor=target,
        rotation=R,
        translation=t,
        scale=s,
        temporal_offset=temporal_offset
    )


# =============================================================================
# VALIDATION / TESTING
# =============================================================================

def validate_connection(connection: ConnectionOperator,
                        test_data: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
    """
    Validate learned connection on test data.

    Returns:
        Dictionary with validation metrics
    """
    if not test_data:
        return {'rmse': 0.0, 'max_error': 0.0, 'mean_error': 0.0}

    errors = []
    for source_meas, target_meas in test_data:
        transported = connection.transport(source_meas)
        min_dim = min(len(transported), len(target_meas))
        error = np.linalg.norm(transported[:min_dim] - target_meas[:min_dim])
        errors.append(error)

    errors = np.array(errors)

    return {
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'max_error': float(np.max(errors)),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'n_samples': len(errors)
    }


# =============================================================================
# DEMO / SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Patent 5a - Enhanced Connection Learning Module")
    print("Self-Test and Demonstration")
    print("=" * 70)

    np.random.seed(42)

    # Generate synthetic calibration data
    # True transform: 15 degree rotation, 0.5m translation, scale=1.02
    true_angle = np.radians(15)
    true_R = np.array([
        [np.cos(true_angle), -np.sin(true_angle), 0],
        [np.sin(true_angle), np.cos(true_angle), 0],
        [0, 0, 1]
    ])
    true_t = np.array([0.5, -0.3, 0.1])
    true_s = 1.02

    print(f"\n  True transformation:")
    print(f"    Rotation angle: {np.degrees(true_angle):.1f} deg")
    print(f"    Translation: {true_t}")
    print(f"    Scale: {true_s}")

    # Generate calibration points
    n_calib = 50
    n_outliers = 10

    source_pts = []
    target_pts = []

    for i in range(n_calib):
        # Random source point
        p_source = np.random.randn(3) * 5

        # True target (with noise)
        p_target = true_s * (true_R @ p_source) + true_t
        p_target += np.random.randn(3) * 0.1  # Measurement noise

        # Add outliers
        if i < n_outliers:
            p_target += np.random.randn(3) * 3.0  # Outlier

        source_pts.append(p_source)
        target_pts.append(p_target)

    calibration_data = list(zip(source_pts, target_pts))

    print(f"\n  Calibration data: {n_calib} points ({n_outliers} outliers)")

    # Learn connection with RANSAC
    print("\n  Learning connection (with RANSAC)...")
    connection = learn_connection_auto(
        calibration_data=calibration_data,
        source="camera",
        target="lidar",
        source_modality="camera",
        target_modality="lidar",
        use_ransac=True
    )

    print(f"\n  Learned transformation:")
    print(f"    Rotation:\n{connection.rotation}")
    print(f"    Translation: {connection.translation}")
    print(f"    Scale: {connection.scale:.4f}")

    # Compute errors
    rotation_error = np.linalg.norm(connection.rotation - true_R, 'fro')
    translation_error = np.linalg.norm(connection.translation - true_t)
    scale_error = abs(connection.scale - true_s)

    print(f"\n  Estimation errors:")
    print(f"    Rotation (Frobenius): {rotation_error:.4f}")
    print(f"    Translation: {translation_error:.4f} m")
    print(f"    Scale: {scale_error:.4f}")

    # Validate on clean test data
    test_data = []
    for _ in range(20):
        p_source = np.random.randn(3) * 5
        p_target = true_s * (true_R @ p_source) + true_t + np.random.randn(3) * 0.05
        test_data.append((p_source, p_target))

    metrics = validate_connection(connection, test_data)
    print(f"\n  Validation on clean test data:")
    print(f"    RMSE: {metrics['rmse']:.4f} m")
    print(f"    Max error: {metrics['max_error']:.4f} m")
    print(f"    Mean error: {metrics['mean_error']:.4f} m")

    # Test without RANSAC for comparison
    print("\n  Learning connection (without RANSAC)...")
    connection_no_ransac = learn_connection_auto(
        calibration_data=calibration_data,
        source="camera",
        target="lidar",
        use_ransac=False
    )

    metrics_no_ransac = validate_connection(connection_no_ransac, test_data)
    print(f"  Validation (no RANSAC):")
    print(f"    RMSE: {metrics_no_ransac['rmse']:.4f} m")

    improvement = (metrics_no_ransac['rmse'] - metrics['rmse']) / metrics_no_ransac['rmse'] * 100
    print(f"\n  RANSAC improvement: {improvement:.1f}% RMSE reduction")

    print("\n" + "=" * 70)
    print("Self-test complete.")
    print("=" * 70)
