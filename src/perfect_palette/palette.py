from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

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

    linear_rgb = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )

    xyz = linear_rgb @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    ).T

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


ImageInput = Union[str, Path, Image.Image, np.ndarray]
PaletteInput = Union[Iterable[Union[str, Iterable[int]]], np.ndarray]


def _load_rgb_and_alpha(image: ImageInput) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGBA")
        rgba = np.asarray(pil_image, dtype=np.uint8)
        return rgba[..., :3], rgba[..., 3]

    if isinstance(image, Image.Image):
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        return rgba[..., :3], rgba[..., 3]

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError("Image must have shape (H, W, 3) or (H, W, 4).")

    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)

    if arr.shape[-1] == 4:
        return arr[..., :3], arr[..., 3]

    return arr[..., :3], None


def normalize_palette(palette: PaletteInput) -> np.ndarray:
    """
    Normalize a palette to a uint8 RGB array with shape (N, 3).

    Args:
        palette: RGB tuples/lists, a numpy array, or hex strings such as "#ffcc00".

    Returns:
        Palette colors as uint8 RGB values.
    """
    colors = []
    for color in palette:
        if isinstance(color, str):
            hex_color = color.strip()
            if hex_color.startswith("#"):
                hex_color = hex_color[1:]
            if len(hex_color) != 6:
                raise ValueError(f"Invalid hex color: {color}")
            colors.append([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)])
            continue

        rgb = list(color)
        if len(rgb) < 3:
            raise ValueError(f"Palette color must contain at least 3 channels: {color}")
        colors.append(rgb[:3])

    palette_rgb = np.asarray(colors, dtype=np.float64)
    if palette_rgb.ndim != 2 or palette_rgb.shape[1] != 3 or len(palette_rgb) == 0:
        raise ValueError("Palette must contain at least one RGB color.")

    return np.clip(np.rint(palette_rgb), 0, 255).astype(np.uint8)


def extract_palette(
    image: ImageInput,
    max_colors: Optional[int] = None,
    sort_by: str = "count",
    ignore_transparent: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract unique RGB colors and pixel counts from an image.

    This function is intended for pixel-perfect or already-quantized images. For
    noisy AI output, run color simplification first, then extract the palette.

    Args:
        image: Image path, PIL image, or RGB/RGBA numpy array.
        max_colors: Optional maximum number of colors to return.
        sort_by: "count" for most-used colors first, or "rgb" for lexicographic RGB.
        ignore_transparent: Exclude fully transparent pixels for RGBA images.

    Returns:
        A tuple of (palette_rgb, counts), where palette_rgb has shape (N, 3).
    """
    rgb, alpha = _load_rgb_and_alpha(image)
    flat_rgb = rgb.reshape(-1, 3)

    if ignore_transparent and alpha is not None:
        visible = alpha.reshape(-1) > 0
        flat_rgb = flat_rgb[visible]

    if len(flat_rgb) == 0:
        return np.empty((0, 3), dtype=np.uint8), np.empty((0,), dtype=np.int64)

    palette_rgb, counts = np.unique(flat_rgb, axis=0, return_counts=True)

    if sort_by == "count":
        order = np.argsort(-counts, kind="stable")
    elif sort_by == "rgb":
        order = np.lexsort((palette_rgb[:, 2], palette_rgb[:, 1], palette_rgb[:, 0]))
    else:
        raise ValueError('sort_by must be "count" or "rgb".')

    palette_rgb = palette_rgb[order]
    counts = counts[order]

    if max_colors is not None:
        if max_colors <= 0:
            raise ValueError("max_colors must be positive.")
        palette_rgb = palette_rgb[:max_colors]
        counts = counts[:max_colors]

    return palette_rgb.astype(np.uint8, copy=False), counts.astype(np.int64, copy=False)


def save_palette_image(
    palette: PaletteInput,
    output_path: Union[str, Path],
    swatch_size: int = 32,
    columns: Optional[int] = None,
    padding: int = 0,
) -> Image.Image:
    """
    Save a palette as a swatch image.

    Args:
        palette: Palette colors as RGB values or hex strings.
        output_path: Path to save the palette image.
        swatch_size: Width and height of each color swatch.
        columns: Number of columns. Defaults to a near-square layout.
        padding: Pixel gap between swatches.

    Returns:
        The generated PIL image.
    """
    palette_rgb = normalize_palette(palette)
    if swatch_size <= 0:
        raise ValueError("swatch_size must be positive.")
    if padding < 0:
        raise ValueError("padding must be non-negative.")

    color_count = len(palette_rgb)
    if columns is None:
        columns = int(np.ceil(np.sqrt(color_count)))
    if columns <= 0:
        raise ValueError("columns must be positive.")

    rows = int(np.ceil(color_count / columns))
    width = columns * swatch_size + max(columns - 1, 0) * padding
    height = rows * swatch_size + max(rows - 1, 0) * padding

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for index, color in enumerate(palette_rgb):
        row = index // columns
        col = index % columns
        y0 = row * (swatch_size + padding)
        x0 = col * (swatch_size + padding)
        canvas[y0 : y0 + swatch_size, x0 : x0 + swatch_size] = color

    palette_image = Image.fromarray(canvas, mode="RGB")
    palette_image.save(output_path)
    return palette_image


def map_image_to_palette(
    image: ImageInput,
    palette: PaletteInput,
    output_path: Optional[Union[str, Path]] = None,
    color_space: str = "lab",
    preserve_alpha: bool = True,
) -> Image.Image:
    """
    Map every image color to the nearest color in a limited palette.

    Args:
        image: Image path, PIL image, or RGB/RGBA numpy array.
        palette: Target palette as RGB values or hex strings.
        output_path: Optional path to save the mapped image.
        color_space: "lab" for perceptual distance, or "rgb" for raw RGB distance.
        preserve_alpha: Preserve the original alpha channel for RGBA inputs.

    Returns:
        The mapped PIL image.
    """
    rgb, alpha = _load_rgb_and_alpha(image)
    palette_rgb = normalize_palette(palette)

    flat_rgb = rgb.reshape(-1, 3)
    unique_rgb, inverse = np.unique(flat_rgb, axis=0, return_inverse=True)

    if color_space == "lab":
        source_points = rgb_to_lab(unique_rgb)
        palette_points = rgb_to_lab(palette_rgb)
    elif color_space == "rgb":
        source_points = unique_rgb.astype(np.float64)
        palette_points = palette_rgb.astype(np.float64)
    else:
        raise ValueError('color_space must be "lab" or "rgb".')

    distances = np.sum((source_points[:, None, :] - palette_points[None, :, :]) ** 2, axis=2)
    nearest_indices = np.argmin(distances, axis=1)
    mapped_rgb = palette_rgb[nearest_indices][inverse].reshape(rgb.shape)

    if preserve_alpha and alpha is not None:
        mapped_rgba = np.dstack([mapped_rgb, alpha])
        output = Image.fromarray(mapped_rgba, mode="RGBA")
    else:
        output = Image.fromarray(mapped_rgb, mode="RGB")

    if output_path is not None:
        output.save(output_path)

    return output
