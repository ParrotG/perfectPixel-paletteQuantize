from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from sklearn.cluster import AgglomerativeClustering


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

    # Convert RGB values from [0, 255] to [0, 1] before Lab conversion.
    unique_rgb_float = unique_rgb.astype(np.float32) / 255.0
    unique_lab = rgb2lab(unique_rgb_float.reshape(1, -1, 3)).reshape(-1, 3)

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