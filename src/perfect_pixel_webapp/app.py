import base64
import hashlib
from io import BytesIO
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
ASSETS_DIR = SRC_DIR.parent / "assets"

from perfect_palette import extract_palette, map_image_to_palette, normalize_palette
from perfect_palette.color_quantize import simplify_colors_by_lab_threshold_image
from perfect_pixel import get_perfect_pixel


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def resize_nearest(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).NEAREST
    return image.resize(size, resampling)


def show_pixelated_image(image: Image.Image) -> None:
    encoded = base64.b64encode(image_to_png_bytes(image)).decode("ascii")
    st.markdown(
        f"""
        <img
            src="data:image/png;base64,{encoded}"
            style="
                width: 100%;
                height: auto;
                image-rendering: pixelated;
                image-rendering: crisp-edges;
            "
        />
        """,
        unsafe_allow_html=True,
    )


def bytes_to_data_uri(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_to_data_uri(image: Image.Image) -> str:
    return bytes_to_data_uri(image_to_png_bytes(image), "image/png")


def asset_to_data_uri(path: Path, mime: str) -> str:
    return bytes_to_data_uri(path.read_bytes(), mime)


def inject_upload_surface_css(source_image: Optional[Image.Image]) -> None:
    if source_image is None:
        background_uri = asset_to_data_uri(ASSETS_DIR / "upload.svg", "image/svg+xml")
        background_size = "72px 72px"
        aspect_ratio = "1 / 1"
        min_height = "320px"
        hint_text = "点击上传原图"
        hint_display = "block"
    else:
        background_uri = image_to_data_uri(source_image)
        background_size = "contain"
        aspect_ratio = f"{source_image.width} / {source_image.height}"
        min_height = "0"
        hint_text = "点击图片重新上传"
        hint_display = "block"

    st.markdown(
        f"""
        <style>
        .st-key-source_upload_surface [data-testid="stFileUploaderDropzone"] {{
            min-height: {min_height};
            aspect-ratio: {aspect_ratio};
            position: relative;
            overflow: hidden;
            border: 1px dashed rgba(49, 51, 63, 0.35);
            border-radius: 8px;
            background-color: rgba(250, 250, 250, 0.75);
            background-image: url("{background_uri}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: {background_size};
            cursor: pointer;
        }}

        .st-key-source_upload_surface [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: rgb(255, 75, 75);
            background-color: rgba(255, 255, 255, 0.92);
        }}

        .st-key-source_upload_surface [data-testid="stFileUploaderDropzone"] > div {{
            opacity: 0;
        }}

        .st-key-source_upload_surface [data-testid="stFileUploaderDropzone"]::after {{
            content: "{hint_text}";
            display: {hint_display};
            position: absolute;
            left: 50%;
            bottom: 1rem;
            transform: translateX(-50%);
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.88);
            color: rgba(49, 51, 63, 0.82);
            font-size: 0.875rem;
            pointer-events: none;
            white-space: nowrap;
            box-shadow: 0 1px 8px rgba(0, 0, 0, 0.08);
        }}

        .st-key-source_upload_surface [data-testid="stFileUploader"] small {{
            display: none;
        }}

        .st-key-source_upload_surface [data-testid="stFileUploaderFile"] {{
            display: none;
        }}

        .st-key-source_upload_surface [data-testid="stFileUploaderDeleteBtn"] {{
            display: none;
        }}

        .source-upload-placeholder {{
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 320px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def uploaded_image_to_rgb(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def image_bytes_to_rgb(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def uploaded_file_key(uploaded_file, file_bytes: bytes) -> Tuple[str, int, str]:
    digest = hashlib.sha256(file_bytes).hexdigest()
    return uploaded_file.name, len(file_bytes), digest


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def palette_to_swatch_image(
    palette: np.ndarray,
    swatch_size: int = 28,
    columns: Optional[int] = None,
    padding: int = 1,
) -> Image.Image:
    palette_rgb = normalize_palette(palette)
    if len(palette_rgb) == 0:
        return Image.new("RGB", (swatch_size, swatch_size), "white")

    if columns is None:
        columns = min(16, max(1, int(np.ceil(np.sqrt(len(palette_rgb))))))

    rows = int(np.ceil(len(palette_rgb) / columns))
    width = columns * swatch_size + max(columns - 1, 0) * padding
    height = rows * swatch_size + max(rows - 1, 0) * padding
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for index, color in enumerate(palette_rgb):
        row = index // columns
        col = index % columns
        y0 = row * (swatch_size + padding)
        x0 = col * (swatch_size + padding)
        canvas[y0 : y0 + swatch_size, x0 : x0 + swatch_size] = color

    return Image.fromarray(canvas, mode="RGB")


def align_pixels(
    image: Image.Image,
    sample_method: str,
    refine_intensity: float,
    fix_square: bool,
) -> Tuple[int, int, Image.Image]:
    rgb = np.asarray(image, dtype=np.uint8)
    width, height, aligned = get_perfect_pixel(
        rgb,
        sample_method=sample_method,
        refine_intensity=refine_intensity,
        fix_square=fix_square,
        debug=False,
    )
    if width is None or height is None:
        return image.width, image.height, image

    return width, height, Image.fromarray(aligned.astype(np.uint8), mode="RGB")


def calibrate_colors(
    aligned_image: Image.Image,
    palette_image: Optional[Image.Image],
    distance_threshold: float,
    color_space: str,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    if palette_image is not None:
        target_palette, _ = extract_palette(palette_image)
        mapped = map_image_to_palette(
            aligned_image,
            target_palette,
            color_space=color_space,
        )
        return mapped.convert("RGB"), target_palette

    quantized = simplify_colors_by_lab_threshold_image(
        image=aligned_image,
        distance_threshold=distance_threshold,
    )
    return quantized.convert("RGB"), None


def main() -> None:
    st.set_page_config(page_title="Perfect Pixel Palette Workflow", layout="wide")
    inject_upload_surface_css(st.session_state.get("source_image"))
    st.title("Perfect Pixel Palette Workflow")

    with st.sidebar:
        st.header("输入")
        palette_upload = st.file_uploader(
            "上传限定色谱图片（可选）",
            type=("png", "jpg", "jpeg", "webp"),
        )

        st.header("像素对齐")
        sample_method = st.selectbox(
            "Sample 策略",
            options=("center", "median", "majority"),
            index=0,
        )
        refine_intensity = st.slider(
            "网格线修正强度",
            min_value=0.0,
            max_value=0.5,
            value=0.3,
            step=0.05,
        )
        fix_square = st.checkbox("接近正方形时自动修正尺寸", value=True)

        st.header("色彩校准")
        distance_threshold = st.slider(
            "Lab 颜色合并阈值",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=0.5,
            disabled=palette_upload is not None,
        )
        color_space = st.radio(
            "限定色谱最近邻空间",
            options=("lab", "rgb"),
            index=0,
            disabled=palette_upload is None,
            horizontal=True,
        )
        max_palette_preview = st.slider(
            "色谱预览最大颜色数",
            min_value=8,
            max_value=256,
            value=128,
            step=8,
        )

    palette_image = uploaded_image_to_rgb(palette_upload) if palette_upload is not None else None

    original_col, result_col = st.columns(2)
    with original_col:
        st.subheader("原图")
        source_image = None
        source_upload = None

        with st.container(key="source_upload_surface"):
            source_upload = st.file_uploader(
                "上传或替换原图",
                type=("png", "jpg", "jpeg", "webp"),
                key="source_upload",
                label_visibility="collapsed",
            )

        if source_upload is not None:
            source_bytes = source_upload.getvalue()
            source_key = uploaded_file_key(source_upload, source_bytes)
            if st.session_state.get("source_upload_key") != source_key:
                st.session_state["source_upload_key"] = source_key
                st.session_state.pop("workflow_result", None)
                st.session_state["source_image"] = image_bytes_to_rgb(source_bytes)
                rerun_app()
            source_image = image_bytes_to_rgb(source_bytes)
            st.session_state["source_image"] = source_image
        else:
            source_image = st.session_state.get("source_image")

        process_clicked = st.button(
            "启动处理",
            type="primary",
            disabled=source_image is None,
            use_container_width=True,
        )

    if process_clicked and source_image is not None:
        with st.spinner("正在进行像素对齐和色彩校准..."):
            aligned_width, aligned_height, aligned_image = align_pixels(
                image=source_image,
                sample_method=sample_method,
                refine_intensity=refine_intensity,
                fix_square=fix_square,
            )
            processed_image, target_palette = calibrate_colors(
                aligned_image=aligned_image,
                palette_image=palette_image,
                distance_threshold=distance_threshold,
                color_space=color_space,
            )

            before_palette, before_counts = extract_palette(
                aligned_image,
                max_colors=max_palette_preview,
            )
            if target_palette is not None:
                after_palette = target_palette[:max_palette_preview]
            else:
                after_palette, _ = extract_palette(
                    processed_image,
                    max_colors=max_palette_preview,
                )

            st.session_state["workflow_result"] = {
                "aligned_size": (aligned_width, aligned_height),
                "source_size": source_image.size,
                "processed_image": processed_image,
                "before_palette": before_palette,
                "before_color_count": int(len(before_counts)),
                "after_palette": after_palette,
                "used_uploaded_palette": target_palette is not None,
            }

    result = st.session_state.get("workflow_result")
    with result_col:
        st.subheader("处理后图")
        if result is None:
            st.caption("点击“启动处理”后显示结果。")
        else:
            processed_image = result["processed_image"]
            preview_image = resize_nearest(processed_image, result["source_size"])
            show_pixelated_image(preview_image)
            width, height = result["aligned_size"]
            preview_width, preview_height = result["source_size"]
            st.caption(
                f"输出网格尺寸：{width} x {height}；预览尺寸：{preview_width} x {preview_height}"
            )
            st.download_button(
                "下载处理后图片",
                data=image_to_png_bytes(processed_image),
                file_name="perfect_pixel_palette.png",
                mime="image/png",
                use_container_width=True,
            )

    st.divider()
    palette_before_col, palette_after_col = st.columns(2)
    with palette_before_col:
        st.subheader("处理前色谱")
        if result is None:
            st.caption("点击“启动处理”后显示像素对齐后的色谱。")
        else:
            st.image(
                palette_to_swatch_image(result["before_palette"]),
                use_container_width=False,
            )
            st.caption(f"预览颜色数：{len(result['before_palette'])}")

    with palette_after_col:
        st.subheader("处理后色谱")
        if result is None:
            if palette_image is not None:
                preview_palette, _ = extract_palette(
                    palette_image,
                    max_colors=max_palette_preview,
                )
                st.image(palette_to_swatch_image(preview_palette), use_container_width=False)
                st.caption("已上传限定色谱，点击处理后将用于最近邻映射。")
            else:
                st.caption("未上传限定色谱时，将显示自动合并后的输出色谱。")
        else:
            st.image(
                palette_to_swatch_image(result["after_palette"]),
                use_container_width=False,
            )
            if result["used_uploaded_palette"]:
                st.caption(f"使用上传限定色谱，预览颜色数：{len(result['after_palette'])}")
            else:
                st.caption(f"自动合并后预览颜色数：{len(result['after_palette'])}")


if __name__ == "__main__":
    main()
