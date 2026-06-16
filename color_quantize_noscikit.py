

from pathlib import Path
import math

import numpy as np
from PIL import Image


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB colors in [0, 255] to CIE Lab using D65 white point.

    Args:
        rgb: RGB array with shape (..., 3), dtype uint8 or numeric.

    Returns:
        Lab array with shape (..., 3).
    """
    srgb = rgb.astype(np.float64) / 255.0

    # Convert gamma-encoded sRGB to linear RGB.
    linear_rgb = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )

    # Linear RGB to XYZ, D65.
    xyz = linear_rgb @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    ).T

    # Normalize by D65 reference white.
    white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xyz_scaled = xyz / white

    epsilon = 216 / 24389
    kappa = 24389 / 27

    f_xyz = np.where(
        xyz_scaled > epsilon,
        np.cbrt(xyz_scaled),
        (kappa * xyz_scaled + 16) / 116,
    )

    lab = np.empty_like(f_xyz)
    lab[..., 0] = 116 * f_xyz[..., 1] - 16
    lab[..., 1] = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    lab[..., 2] = 200 * (f_xyz[..., 1] - f_xyz[..., 2])

    return lab


def pairwise_euclidean_distance(points: np.ndarray) -> np.ndarray:
    """
    Compute a pairwise Euclidean distance matrix.

    Args:
        points: Array with shape (n, d).

    Returns:
        Distance matrix with shape (n, n).
    """
    delta = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(delta, axis=2)


def complete_linkage_agglomerative_labels(
    distance_matrix: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    """
    Perform agglomerative clustering with complete linkage.

    Args:
        distance_matrix: Pairwise distance matrix with shape (n, n).
        distance_threshold: Maximum complete-linkage distance for merging clusters.

    Returns:
        Cluster label array with shape (n,).
    """
    if distance_threshold < 0:
        raise ValueError("Distance threshold must be non-negative.")

    n = distance_matrix.shape[0]
    clusters = [[index] for index in range(n)]
    active = set(range(n))

    cluster_distance = distance_matrix.astype(np.float64, copy=True)
    np.fill_diagonal(cluster_distance, np.inf)

    while True:
        active_indices = sorted(active)
        best_pair = None
        best_distance = math.inf

        for position, left in enumerate(active_indices):
            right_candidates = active_indices[position + 1 :]
            if not right_candidates:
                continue

            distances = cluster_distance[left, right_candidates]
            min_position = int(np.argmin(distances))
            candidate_distance = float(distances[min_position])

            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_pair = (left, right_candidates[min_position])

        if best_pair is None or best_distance > distance_threshold:
            break

        left, right = best_pair

        clusters[left].extend(clusters[right])
        active.remove(right)

        # Complete linkage: distance between two clusters is the maximum
        # pairwise distance between their member colors.
        for other in active:
            if other == left:
                continue

            merged_distance = max(
                cluster_distance[left, other],
                cluster_distance[right, other],
            )
            cluster_distance[left, other] = merged_distance
            cluster_distance[other, left] = merged_distance

        cluster_distance[right, :] = np.inf
        cluster_distance[:, right] = np.inf

    labels = np.empty(n, dtype=np.int32)
    for label, cluster_index in enumerate(sorted(active)):
        labels[clusters[cluster_index]] = label

    return labels


def simplify_colors_by_lab_threshold(
    input_path: str,
    output_path: str,
    distance_threshold: float = 2.5,
) -> None:
    """
    Simplify visually similar colors in an image by clustering unique colors
    in CIE Lab color space using agglomerative clustering with complete linkage.

    Args:
        input_path: Path to the input image.
        output_path: Path to save the processed image.
        distance_threshold: Maximum Lab-space distance threshold for merging clusters.
            Smaller values preserve more colors.
    """
    image = Image.open(input_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)

    flat_rgb = rgb.reshape(-1, 3)
    unique_rgb, inverse = np.unique(flat_rgb, axis=0, return_inverse=True)

    unique_lab = rgb_to_lab(unique_rgb)
    distance_matrix = pairwise_euclidean_distance(unique_lab)

    labels = complete_linkage_agglomerative_labels(
        distance_matrix=distance_matrix,
        distance_threshold=distance_threshold,
    )

    simplified_rgb = np.zeros_like(unique_rgb)

    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_lab = unique_lab[cluster_indices]

        # Use the medoid: the actual original color closest to the cluster center.
        center_lab = cluster_lab.mean(axis=0)
        distances = np.linalg.norm(cluster_lab - center_lab, axis=1)
        medoid_index = cluster_indices[np.argmin(distances)]

        simplified_rgb[cluster_indices] = unique_rgb[medoid_index]

    mapped_flat_rgb = simplified_rgb[inverse]
    mapped_rgb = mapped_flat_rgb.reshape(rgb.shape)

    output = Image.fromarray(mapped_rgb, mode="RGB")
    output.save(output_path)


if __name__ == "__main__":
    simplify_colors_by_lab_threshold(
        input_path="output.png",
        output_path="output_colormerged.png",
        distance_threshold=15,
    )