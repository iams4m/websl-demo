import os
import math
import tempfile
import numpy as np
import gradio as gr
import torch

from PIL import Image
from pytorch_msssim import ms_ssim
from sklearn.cluster import MiniBatchKMeans
from skimage.metrics import structural_similarity as ssim


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def img01_to_uint8(img01):
    return (clip01(img01) * 255.0 + 0.5).astype(np.uint8)


def fmt_kb(n):
    return f"{n / 1024:.1f} KB"


def save_webp(path, img01, quality=80, method=6):
    img8 = img01_to_uint8(img01)
    Image.fromarray(img8).save(path, format="WEBP", quality=int(quality), method=int(method))
    return os.path.getsize(path)


def load_webp(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def psnr(a, b):
    e = rmse(a, b)
    if e <= 1e-12:
        return 999.0
    return float(20.0 * np.log10(1.0 / e))


def compute_ms_ssim(img_a, img_b):
    ta = torch.from_numpy(img_a).permute(2, 0, 1).unsqueeze(0).float()
    tb = torch.from_numpy(img_b).permute(2, 0, 1).unsqueeze(0).float()
    return float(ms_ssim(ta, tb, data_range=1.0, size_average=True))


def compute_ssim(img_a, img_b):
    return float(ssim(img_a, img_b, channel_axis=2, data_range=1.0))


def rgb_to_ycbcr(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 0.5 + (0.5 * r - 0.418688 * g - 0.081312 * b)
    return clip01(np.stack([y, cb, cr], axis=-1))


def ycbcr_to_rgb(ycbcr):
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1] - 0.5
    cr = ycbcr[..., 2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return clip01(np.stack([r, g, b], axis=-1))


def resize_if_needed(img, max_side=768):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pil = Image.fromarray(img01_to_uint8(img))
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(pil).astype(np.float32) / 255.0


def make_grain_colors(img3, grain_size):
    h, w, _ = img3.shape
    gh = math.ceil(h / grain_size)
    gw = math.ceil(w / grain_size)
    grid = np.zeros((gh, gw, 3), dtype=np.float32)

    for gy in range(gh):
        y0, y1 = gy * grain_size, min((gy + 1) * grain_size, h)
        for gx in range(gw):
            x0, x1 = gx * grain_size, min((gx + 1) * grain_size, w)
            grid[gy, gx] = np.mean(img3[y0:y1, x0:x1], axis=(0, 1))

    return grid


def project_grid3(grid3, out_h, out_w):
    gh, gw, _ = grid3.shape
    out = np.zeros((out_h, out_w, 3), dtype=np.float32)
    sy, sx = out_h / gh, out_w / gw

    for gy in range(gh):
        y0 = int(round(gy * sy))
        y1 = min(max(int(round((gy + 1) * sy)), y0 + 1), out_h)
        for gx in range(gw):
            x0 = int(round(gx * sx))
            x1 = min(max(int(round((gx + 1) * sx)), x0 + 1), out_w)
            out[y0:y1, x0:x1] = grid3[gy, gx]

    return out


def reorder_palette_and_indices(palette, indices_grid, color_space="ycbcr"):
    if color_space == "ycbcr":
        keys = np.stack(
            [
                palette[:, 0],
                np.abs(palette[:, 1] - 0.5),
                np.abs(palette[:, 2] - 0.5),
            ],
            axis=1,
        )
    else:
        lum = 0.299 * palette[:, 0] + 0.587 * palette[:, 1] + 0.114 * palette[:, 2]
        hue_like = np.arctan2(palette[:, 1] - palette[:, 2], palette[:, 0] - palette[:, 1])
        sat_like = np.max(palette, axis=1) - np.min(palette, axis=1)
        keys = np.stack([lum, hue_like, sat_like], axis=1)

    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    inv = np.zeros_like(order)
    inv[order] = np.arange(len(order))

    palette2 = palette[order]
    indices2 = inv[indices_grid]
    return palette2, indices2


def pipeline_websl(img, grain_size=2, n_colors=448, color_space="ycbcr"):
    h, w = img.shape[:2]

    if color_space == "ycbcr":
        work = rgb_to_ycbcr(img)
    else:
        work = img.copy()

    grain_colors = make_grain_colors(work, grain_size)
    gh, gw, _ = grain_colors.shape
    flat_colors = grain_colors.reshape(-1, 3)

    n_clusters = min(n_colors, len(flat_colors))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(4096, len(flat_colors)),
        n_init=5,
    )
    labels = kmeans.fit_predict(flat_colors)
    palette = kmeans.cluster_centers_.astype(np.float32)
    indices_grid = labels.reshape(gh, gw)

    palette, indices_grid = reorder_palette_and_indices(
        palette,
        indices_grid,
        color_space=color_space,
    )

    grain_rec = palette[indices_grid]
    out = project_grid3(grain_rec, h, w)

    if color_space == "ycbcr":
        img_rec = ycbcr_to_rgb(out)
    else:
        img_rec = clip01(out)

    palette_q = (clip01(palette) * 255.0 + 0.5).astype(np.uint8)
    support_bytes = len(palette_q.tobytes()) + indices_grid.size

    return img_rec, support_bytes, n_clusters


def build_report(webp_ref, best):
    lines = []
    lines.append("WebP direct")
    lines.append(f"- Taille: {fmt_kb(webp_ref['bytes'])}")
    lines.append(f"- RMSE: {webp_ref['rmse']:.4f}")
    lines.append(f"- PSNR: {webp_ref['psnr']:.2f}")
    lines.append(f"- SSIM: {webp_ref['ssim']:.4f}")
    lines.append(f"- MS-SSIM: {webp_ref['ms_ssim']:.4f}")
    lines.append("")
    lines.append("Meilleur WEBSL -> WebP")
    lines.append(f"- Grain size: {best['grain_size']}")
    lines.append(f"- Couleur: {best['color_space']}")
    lines.append(f"- K: {best['K']}")
    lines.append(f"- Support: {fmt_kb(best['support'])}")
    lines.append(f"- Sortie WebP: {fmt_kb(best['bytes'])}")
    lines.append(f"- Gain vs WebP direct: {(1 - best['bytes'] / webp_ref['bytes']) * 100:.1f}%")
    lines.append(f"- RMSE: {best['rmse']:.4f}")
    lines.append(f"- PSNR: {best['psnr']:.2f}")
    lines.append(f"- SSIM: {best['ssim']:.4f}")
    lines.append(f"- MS-SSIM: {best['ms_ssim']:.4f}")
    return "\n".join(lines)


def run_demo(input_image):
    if input_image is None:
        return None, None, "Aucune image fournie."

    img = np.asarray(input_image.convert("RGB")).astype(np.float32) / 255.0
    img = resize_if_needed(img, max_side=768)

    with tempfile.TemporaryDirectory() as td:
        ref_path = os.path.join(td, "direct.webp")
        ref_bytes = save_webp(ref_path, img, quality=80, method=6)
        ref_img = load_webp(ref_path)

        webp_ref = {
            "img": ref_img,
            "bytes": ref_bytes,
            "rmse": rmse(img, ref_img),
            "psnr": psnr(img, ref_img),
            "ssim": compute_ssim(img, ref_img),
            "ms_ssim": compute_ms_ssim(img, ref_img),
        }

        best = None

        for grain_size in [1, 2]:
            for color_space in ["ycbcr", "rgb"]:
                for K in [256, 320, 384, 448, 512]:
                    img_rec, support, n_clusters = pipeline_websl(
                        img,
                        grain_size=grain_size,
                        n_colors=K,
                        color_space=color_space,
                    )

                    out_path = os.path.join(td, f"tmp_g{grain_size}_{color_space}_{K}.webp")
                    out_bytes = save_webp(out_path, img_rec, quality=80, method=6)
                    out_img = load_webp(out_path)

                    candidate = {
                        "img": out_img,
                        "support": support,
                        "bytes": out_bytes,
                        "grain_size": grain_size,
                        "color_space": color_space,
                        "K": n_clusters,
                        "rmse": rmse(img, out_img),
                        "psnr": psnr(img, out_img),
                        "ssim": compute_ssim(img, out_img),
                        "ms_ssim": compute_ms_ssim(img, out_img),
                    }

                    if best is None:
                        best = candidate
                    else:
                        better_size = candidate["bytes"] < best["bytes"]
                        similar_quality = candidate["ms_ssim"] >= best["ms_ssim"] - 0.01
                        if better_size and similar_quality:
                            best = candidate

        orig_pil = Image.fromarray(img01_to_uint8(img))
        webp_pil = Image.fromarray(img01_to_uint8(webp_ref["img"]))
        websl_pil = Image.fromarray(img01_to_uint8(best["img"]))

        report = build_report(webp_ref, best)
        return orig_pil, webp_pil, websl_pil, report


with gr.Blocks() as demo:
    gr.Markdown("# WEBSL → WebP demo")
    gr.Markdown("Upload an image, then compare direct WebP with the best WEBSL → WebP result found by the sweep.")

    with gr.Row():
        inp = gr.Image(type="pil", label="Input image")

    run_btn = gr.Button("Run comparison")

    with gr.Row():
        out_orig = gr.Image(label="Original")
        out_webp = gr.Image(label="WebP direct")
        out_websl = gr.Image(label="Best WEBSL → WebP")

    out_text = gr.Textbox(label="Results", lines=18)

    run_btn.click(
        fn=run_demo,
        inputs=inp,
        outputs=[out_orig, out_webp, out_websl, out_text],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
