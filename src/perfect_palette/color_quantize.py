from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering


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

    # Convert linear RGB to XYZ using the standard sRGB D65 matrix.
    xyz = linear_rgb @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    ).T

    # Normalize by the D65 reference white point.
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
    image = Image.open(input_path).convert("RGBA")
    rgba = np.asarray(image, dtype=np.uint8)

    rgb = rgba[:, :, :3]
    flat_rgb = rgb.reshape(-1, 3)

    unique_rgb, inverse = np.unique(flat_rgb, axis=0, return_inverse=True)

    unique_lab = rgb_to_lab(unique_rgb)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="complete",
        metric="euclidean",
    ).fit(unique_lab)

    labels = clustering.labels_
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