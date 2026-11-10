"""
app.py — Gradio UI: metric 3D reconstruction from a single image with UniDepth V2.

Run:  python app.py            (then open http://127.0.0.1:7860)
      python app.py --preload  (load the model before the UI starts)
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from engine import EngineError, PointCloud, Reconstruction, UniDepthEngine, export_ply

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
CANVAS = "#16181d"
GRID = "#232834"  # WebGL axes ignore alpha, so use an opaque subtle grey
ENGINE: UniDepthEngine  # set in main()


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _cam_to_plot(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenCV camera frame (x right, y down, z fwd) -> plot frame (x right, y depth, z up)."""
    return xyz[:, 0], xyz[:, 2], -xyz[:, 1]


def _axis(title: str) -> dict[str, Any]:
    return dict(
        title=dict(text=title, font=dict(size=12, color="#9aa4b2")),
        gridcolor=GRID, zerolinecolor=GRID, showbackground=False,
        tickfont=dict(size=10, color="#6b7280"), ticksuffix=" m",
    )


def _layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        height=720, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=CANVAS, plot_bgcolor=CANVAS, showlegend=False, uirevision="keep",
        font=dict(family="system-ui, sans-serif", color="#d1d5db"),
        hoverlabel=dict(bgcolor="#1f2937", bordercolor="#374151", font=dict(color="#f9fafb", size=12)),
        scene=dict(
            aspectmode="data", bgcolor=CANVAS,
            xaxis=_axis("X"), yaxis=_axis("Depth"), zaxis=_axis("Up"),
            camera=dict(eye=dict(x=1.0, y=-1.8, z=0.8), up=dict(x=0, y=0, z=1)),
        ),
    )
    return fig


def placeholder_figure() -> go.Figure:
    fig = go.Figure(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=0.1, opacity=0), hoverinfo="skip"))
    fig = _layout(fig)
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        annotations=[dict(
            text="No reconstruction yet", showarrow=False,
            x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=14, color="#6b7280"),
        )],
    )
    return fig


def point_cloud_figure(cloud: PointCloud, point_size: float, color_mode: str) -> go.Figure:
    x, y, z = _cam_to_plot(cloud.xyz)
    distance = np.linalg.norm(cloud.xyz, axis=1)

    if color_mode == "Depth":
        lo, hi = np.percentile(y, [2, 98])
        marker = dict(
            size=point_size, color=y, colorscale="Turbo", cmin=lo, cmax=hi,
            colorbar=dict(
                title=dict(text="depth [m]", font=dict(size=12, color="#9aa4b2")), thickness=12, len=0.6,
                x=0.98, tickfont=dict(size=10, color="#9aa4b2"), outlinewidth=0,
            ),
        )
    else:
        marker = dict(size=point_size, color=[f"rgb({r},{g},{b})" for r, g, b in cloud.rgb])

    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z, mode="markers", marker=marker, customdata=distance,
        hovertemplate="x %{x:.3f} m<br>depth %{y:.3f} m<br>up %{z:.3f} m<br>"
                      "range %{customdata:.3f} m<extra></extra>",
    ))
    return _layout(fig)


def stats_html(rec: Reconstruction, cloud: PointCloud) -> str:
    d = rec.depth[rec.valid]
    w, h, depth = cloud.extent_m()
    iw, ih = rec.size
    image = f"{iw} x {ih}" + (f" (resized from {rec.original_size[0]} x {rec.original_size[1]})"
                              if rec.original_size != rec.size else "")
    rows = [
        ("Extent (w x h x d)", f"{w:.2f} x {h:.2f} x {depth:.2f} m"),
        ("Depth min / median / max", f"{d.min():.2f} / {np.median(d):.2f} / {d.max():.2f} m"),
        ("Points", f"{len(cloud):,}"),
        ("Image", image),
        ("Inference", f"{rec.inference_seconds:.1f} s ({rec.device})"),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="stats">{body}</table>'


EMPTY_STATS = '<p class="stats-empty">Measurements are shown here after a run.</p>'


# --------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------


def render(state: dict[str, Any] | None, max_points: int, point_size: float, remove_edges: bool,
           color_mode: str) -> tuple[Any, Any, Any]:
    """Redraw from the cached reconstruction (no inference)."""
    if not state:
        return gr.skip(), gr.skip(), gr.skip()
    rec: Reconstruction = state["rec"]
    cloud = rec.point_cloud(remove_edges=remove_edges)
    suffix = "clean" if remove_edges else "raw"
    ply = export_ply(cloud, OUTPUT_DIR / f"{state['stem']}_{state['run_id']}_{suffix}.ply")
    return (
        point_cloud_figure(cloud.subsample(int(max_points)), point_size, color_mode),
        stats_html(rec, cloud),
        gr.DownloadButton(value=str(ply), label=f"Download PLY ({len(cloud):,} points)", interactive=True),
    )


def reconstruct(image_path: str | None, max_points: int, point_size: float, remove_edges: bool,
                color_mode: str) -> tuple[Any, ...]:
    if not image_path:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    try:
        rec = ENGINE.reconstruct(image_path)
    except EngineError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:  # never leak a raw traceback into the UI
        logger.exception("Unexpected failure")
        raise gr.Error(f"Unexpected error: {exc}") from exc

    state = {"rec": rec, "stem": Path(image_path).stem or "scene", "run_id": datetime.now().strftime("%Y%m%d_%H%M%S")}
    return (state, *render(state, max_points, point_size, remove_edges, color_mode))


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------

CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
#header { padding: 4px 2px 10px; border-bottom: 1px solid var(--border-color-primary); margin-bottom: 6px; }
#header h1 { font-size: 1.35rem; font-weight: 600; margin: 0 0 2px; }
#header p { margin: 0; font-size: .9rem; color: var(--body-text-color-subdued); }
#viewer { border: 1px solid var(--border-color-primary); border-radius: 4px; overflow: hidden; background: #16181d; }
#viewer .wrap, #viewer > div { background: #16181d; }
footer { display: none !important; }
table.stats { width: 100%; border-collapse: collapse; font-size: .85rem; border: none !important; margin: 0; }
.prose table.stats tr, .prose table.stats tbody, table.stats tr { border: none !important; }
.prose table.stats td:first-child, .prose table.stats td:last-child { border-left: none !important; border-right: none !important; }
table.stats td {
  padding: 6px 2px !important; border: none !important; border-bottom: 1px solid var(--border-color-primary) !important;
  background: transparent !important;
}
table.stats td:first-child { color: var(--body-text-color-subdued); }
table.stats td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
.stats-empty { font-size: .85rem; color: var(--body-text-color-subdued); margin: 4px 2px; }
#footer { font-size: .8rem; color: var(--body-text-color-subdued); padding-top: 4px; }
"""

HEADER = """
<div id="header">
  <h1>Single-image metric 3D reconstruction</h1>
  <p>UniDepth V2 (ViT-L/14). Coordinates are in metres, relative to the camera.</p>
</div>
"""

FOOTER = """
<div id="footer">Left-drag to rotate, scroll to zoom, right-drag to pan. Hover a point to read its coordinates.</div>
"""


def build_ui() -> gr.Blocks:
    theme = gr.themes.Default(
        primary_hue="slate", neutral_hue="gray", radius_size="sm",
        font=["system-ui", "Segoe UI", "Helvetica", "Arial", "sans-serif"],
    )
    with gr.Blocks(title="Metric 3D Reconstruction", theme=theme, css=CSS) as demo:
        state = gr.State(None)
        gr.HTML(HEADER)
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                image_in = gr.Image(type="filepath", label="Input image", sources=["upload", "clipboard"], height=300)
                if EXAMPLES:
                    gr.Examples(EXAMPLES, inputs=image_in, label="Example")
                run_btn = gr.Button("Reconstruct", variant="primary")
                with gr.Accordion("Display options", open=False):
                    color_mode = gr.Radio(["Photo", "Depth"], value="Photo", label="Colour")
                    point_size = gr.Slider(0.5, 4.0, value=2.0, step=0.1, label="Point size")
                    max_points = gr.Slider(20_000, 300_000, value=120_000, step=10_000,
                                           label="Max points displayed", info="Display only; the PLY contains all points")
                    remove_edges = gr.Checkbox(value=True, label="Filter depth discontinuities",
                                               info="Removes points with a >4% depth jump to a neighbour")
                stats = gr.HTML(EMPTY_STATS)
                download = gr.DownloadButton("Download PLY", interactive=False, variant="secondary")
            with gr.Column(scale=8):
                viewer = gr.Plot(placeholder_figure(), show_label=False, elem_id="viewer")
                gr.HTML(FOOTER)

        settings = [max_points, point_size, remove_edges, color_mode]
        for trigger in (image_in.change, run_btn.click):
            trigger(reconstruct, inputs=[image_in, *settings], outputs=[state, viewer, stats, download])
        for control in (color_mode, remove_edges):
            control.change(render, inputs=[state, *settings], outputs=[viewer, stats, download])
        for slider in (point_size, max_points):
            slider.release(render, inputs=[state, *settings], outputs=[viewer, stats, download])
    return demo


def main() -> None:
    global ENGINE
    parser = argparse.ArgumentParser(description="UniDepth V2 metric 3D reconstruction demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--resolution-level", type=int, default=None, choices=range(0, 10),
                        help="Override UniDepth V2 internal resolution, 0 = fastest, 9 = most detailed "
                             "(default: model's full range)")
    parser.add_argument("--preload", action="store_true", help="Load the model before serving")
    args = parser.parse_args()

    ENGINE = UniDepthEngine(device=args.device, resolution_level=args.resolution_level)
    if args.preload:
        ENGINE.load()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_ui().queue(max_size=8).launch(
        server_name=args.host, server_port=args.port, share=args.share, allowed_paths=[str(OUTPUT_DIR)]
    )


if __name__ == "__main__":
    main()
