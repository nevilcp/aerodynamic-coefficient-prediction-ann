"""Export figures for publication and display a smaller notebook preview."""

import base64
import logging
import re
from html import escape
from io import BytesIO

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import HTML, Image, display

PUBLICATION_DPI = 600
PREVIEW_DPI = 200


def _keep_font_log(record):
    # FontTools already converts these legacy timestamps during PDF embedding.
    return record.getMessage() not in {
        "'created' timestamp seems very low; regarding as unix timestamp",
        "'modified' timestamp seems very low; regarding as unix timestamp",
    }


def finish_figure(
    fig, directory, filename, *, table=None, caption=None, preserve_canvas=False
):
    """Write a PDF and a 600 dpi PNG, then display the figure and optional table."""
    try:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", filename):
            raise ValueError(
                "Use lowercase letters, numbers, and single hyphens without an extension"
            )

        destination = directory / f"{filename}.png"
        bbox = None if preserve_canvas else "tight"
        with mpl.rc_context(
            {"pdf.fonttype": 42, "pdf.use14corefonts": False, "savefig.bbox": None}
        ):
            fig.savefig(destination, dpi=PUBLICATION_DPI, bbox_inches=bbox)
            font_log = logging.getLogger("fontTools.ttLib.tables._h_e_a_d")
            font_log.addFilter(_keep_font_log)
            try:
                fig.savefig(
                    destination.with_suffix(".pdf"),
                    dpi=PUBLICATION_DPI,
                    bbox_inches=bbox,
                )
            finally:
                font_log.removeFilter(_keep_font_log)
            preview = BytesIO()
            fig.savefig(preview, format="png", dpi=PREVIEW_DPI, bbox_inches=bbox)

        if caption:
            destination.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")

        preview_bytes = preview.getvalue()
        if table is None:
            display(Image(data=preview_bytes, format="png", retina=True))
        else:
            encoded = base64.b64encode(preview_bytes).decode("ascii")
            display(
                HTML(
                    '<div style="display:flex;align-items:center;gap:24px;'
                    'flex-wrap:wrap">'
                    '<div style="flex:1 1 480px;min-width:0">'
                    f'<img src="data:image/png;base64,{encoded}" '
                    'alt="Training and validation MSE versus epoch" '
                    'style="width:100%;height:auto"></div>'
                    '<div style="flex:0 1 auto;overflow-x:auto">'
                    f"{table.to_html()}</div></div>"
                )
            )
        if caption:
            display(HTML(f"<p>{escape(caption)}</p>"))
    finally:
        plt.close(fig)
