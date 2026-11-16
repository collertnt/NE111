"""
NE111 Term Project - Plot Digitizer
===================================
A single-page Streamlit app for extracting x-y data from an image of a graph.

How it works
------------
1. Upload an image (png, jpg, gif, tiff; colour or greyscale).
2. Calibrate: click the origin (where the axes meet), then a known point on
   the x-axis, then a known point on the y-axis, and type in their values.
3. Pick data: click on the data points. Each dataset gets its own colour.
4. Edit the table if needed and download everything as a CSV file.

Run locally with:   streamlit run plot_digitizer.py
Required packages:  streamlit, streamlit-image-coordinates, pillow, numpy, pandas
"""

import csv
import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DISPLAY_WIDTH = 800  # width (in screen pixels) of the clickable image

# One colour per dataset (colour-blind friendly palette)
PALETTE = [
    "#E6194B", "#0072B2", "#009E73", "#F58231", "#911EB4",
    "#00B5D8", "#F032E6", "#7A5C00", "#808000", "#000075",
]
CALIB_COLOUR = "#FF00FF"

MODE_CALIBRATE = "1. Calibrate axes"
MODE_PICK = "2. Pick data points"

CALIB_STEPS = [
    "the ORIGIN (the corner where the x and y axes meet)",
    "a known point on the X-AXIS (far from the origin is best)",
    "a known point on the Y-AXIS (far from the origin is best)",
]
CALIB_TAGS = ["O", "X", "Y"]

st.set_page_config(page_title="Plot Digitizer", page_icon="📈", layout="wide")
ss = st.session_state


# ----------------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------------
def load_image(uploaded_file):
    """Open any supported image and return it as an 8-bit RGB PIL image.

    Handles colour, greyscale, palette (gif), transparent and 16/32-bit
    (tiff) images. For multi-frame gif/tiff files the first frame is used.
    """
    im = Image.open(uploaded_file)
    im.seek(0)  # first frame of animated gif / multi-page tiff

    # High bit-depth greyscale (common in scientific tiffs): rescale to 0-255
    if im.mode in ("I", "F") or im.mode.startswith("I;16"):
        arr = np.asarray(im, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

    # Everything else: flatten any transparency onto a white background
    rgba = im.convert("RGBA")
    background = Image.new("RGB", rgba.size, "white")
    background.paste(rgba, mask=rgba.split()[3])
    return background


# ----------------------------------------------------------------------------
# Calibration maths
# ----------------------------------------------------------------------------
# The three calibration clicks define two pixel vectors:
#     u = (x-axis point) - (origin)        v = (y-axis point) - (origin)
# Any pixel p can be written as   p = origin + a*u + b*v.
# 'a' is the fraction of the way along the x-axis and 'b' along the y-axis.
# Because u and v are the *real* axis directions in the image, this works even
# if the scan is rotated (or slightly skewed). For log axes the interpolation
# is simply done on log10 of the values.
def build_calibration(points, x0, x1, y0, y1, x_log, y_log):
    """Return (calibration_dict, None) on success or (None, error_message)."""
    if len(points) < 3:
        return None, None
    if None in (x0, x1, y0, y1):
        return None, "Enter a number for all four calibration values."
    if x_log and (x0 <= 0 or x1 <= 0):
        return None, "A log x-axis needs positive x values (the origin cannot be 0)."
    if y_log and (y0 <= 0 or y1 <= 0):
        return None, "A log y-axis needs positive y values (the origin cannot be 0)."

    tx0, tx1 = (math.log10(x0), math.log10(x1)) if x_log else (x0, x1)
    ty0, ty1 = (math.log10(y0), math.log10(y1)) if y_log else (y0, y1)
    if tx0 == tx1:
        return None, "The two x calibration values must be different."
    if ty0 == ty1:
        return None, "The two y calibration values must be different."

    (ox, oy), (px, py), (qx, qy) = points
    ux, uy = px - ox, py - oy
    vx, vy = qx - ox, qy - oy
    det = ux * vy - uy * vx
    if abs(det) < 1e-9:
        return None, "The calibration points lie on one line - please re-pick them."

    return {
        "o": (ox, oy), "u": (ux, uy), "v": (vx, vy), "det": det,
        "tx0": tx0, "tx1": tx1, "ty0": ty0, "ty1": ty1,
        "x_log": x_log, "y_log": y_log,
    }, None


def pixel_to_data(cal, px, py):
    """Convert an image pixel position to graph (x, y) values."""
    dx, dy = px - cal["o"][0], py - cal["o"][1]
    (ux, uy), (vx, vy) = cal["u"], cal["v"]
    a = (dx * vy - dy * vx) / cal["det"]
    b = (ux * dy - uy * dx) / cal["det"]
    tx = cal["tx0"] + a * (cal["tx1"] - cal["tx0"])
    ty = cal["ty0"] + b * (cal["ty1"] - cal["ty0"])
    x = 10.0 ** tx if cal["x_log"] else tx
    y = 10.0 ** ty if cal["y_log"] else ty
    return x, y


def data_to_pixel(cal, x, y):
    """Convert graph (x, y) values back to an image pixel position."""
    tx = math.log10(x) if cal["x_log"] else x
    ty = math.log10(y) if cal["y_log"] else y
    a = (tx - cal["tx0"]) / (cal["tx1"] - cal["tx0"])
    b = (ty - cal["ty0"]) / (cal["ty1"] - cal["ty0"])
    (ux, uy), (vx, vy) = cal["u"], cal["v"]
    return cal["o"][0] + a * ux + b * vx, cal["o"][1] + a * uy + b * vy


# ----------------------------------------------------------------------------
# Drawing the (zoomed) image with all the click marks on top
# ----------------------------------------------------------------------------
def render_view(image, zoom, pan_x, pan_y):
    """Crop + scale the image for display.

    Returns (view_image, left, top, scale) where (left, top) is the image
    pixel shown at the view's top-left corner and scale = screen px / image px.
    """
    w, h = image.size
    view_w, view_h = w / zoom, h / zoom
    left = min(max(pan_x * w - view_w / 2, 0), w - view_w)
    top = min(max(pan_y * h - view_h / 2, 0), h - view_h)
    scale = DISPLAY_WIDTH / view_w
    out_size = (DISPLAY_WIDTH, max(1, round(view_h * scale)))
    view = image.resize(
        out_size, resample=Image.BICUBIC,
        box=(left, top, left + view_w, top + view_h),
    )
    return view, left, top, scale


def draw_overlays(view, left, top, scale, datasets, calib_points):
    """Draw calibration marks and every dataset's points onto the view."""
    draw = ImageDraw.Draw(view)

    def to_screen(p):
        return (p[0] - left) * scale, (p[1] - top) * scale

    # Calibration: lines along the picked axes, then a labelled cross per point
    if len(calib_points) >= 2:
        draw.line([to_screen(calib_points[0]), to_screen(calib_points[1])],
                  fill=CALIB_COLOUR, width=2)
    if len(calib_points) >= 3:
        draw.line([to_screen(calib_points[0]), to_screen(calib_points[2])],
                  fill=CALIB_COLOUR, width=2)
    for tag, p in zip(CALIB_TAGS, calib_points):
        sx, sy = to_screen(p)
        r = 10
        draw.line([(sx - r, sy), (sx + r, sy)], fill=CALIB_COLOUR, width=3)
        draw.line([(sx, sy - r), (sx, sy + r)], fill=CALIB_COLOUR, width=3)
        draw.text((sx + 8, sy + 6), tag, fill=CALIB_COLOUR)

    # Data points: same colour within a dataset, new colour per dataset
    r = 6
    for ds in datasets:
        for p in ds["points"]:
            sx, sy = to_screen(p)
            draw.ellipse([sx - r, sy - r, sx + r, sy + r],
                         fill=ds["color"], outline="white", width=2)
            draw.ellipse([sx - 1, sy - 1, sx + 1, sy + 1], fill="white")
    return view


# ----------------------------------------------------------------------------
# Table / CSV helpers
# ----------------------------------------------------------------------------
def build_table(datasets, cal, x_label, y_label):
    """Wide table: two columns (x, y) per dataset, padded with blanks."""
    n_rows = max((len(ds["points"]) for ds in datasets), default=0)
    columns, used = {}, set()

    def unique(name):
        # st.data_editor needs unique column names
        candidate, k = name, 2
        while candidate in used:
            candidate, k = f"{name} ({k})", k + 1
        used.add(candidate)
        return candidate

    for ds in datasets:
        values = [pixel_to_data(cal, *p) for p in ds["points"]]
        pad = [np.nan] * (n_rows - len(values))
        columns[unique(f"{ds['name']} | {x_label}")] = [v[0] for v in values] + pad
        columns[unique(f"{ds['name']} | {y_label}")] = [v[1] for v in values] + pad
    return pd.DataFrame(columns, dtype=float)


def build_csv(datasets, cal, x_label, y_label):
    """CSV in the required layout:
    row 1 = dataset names, row 2 = axis labels, then the x,y values."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    row1, row2 = [], []
    for ds in datasets:
        row1 += [ds["name"], ""]
        row2 += [x_label, y_label]
    writer.writerow(row1)
    writer.writerow(row2)
    n_rows = max((len(ds["points"]) for ds in datasets), default=0)
    for i in range(n_rows):
        row = []
        for ds in datasets:
            if i < len(ds["points"]):
                x, y = pixel_to_data(cal, *ds["points"][i])
                row += [f"{x:.6g}", f"{y:.6g}"]
            else:
                row += ["", ""]
        writer.writerow(row)
    return buffer.getvalue()


def parse_number(text):
    """Turn user text such as '10', '0.5' or '1e-3' into a float (or None)."""
    try:
        value = float(text.strip())
        return value if math.isfinite(value) else None
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# Session state + button callbacks
# ----------------------------------------------------------------------------
def add_dataset():
    ds_id = ss.next_id
    ss.next_id += 1
    ss.datasets.append({
        "id": ds_id,
        "name": f"Dataset {ds_id + 1}",
        "color": PALETTE[ds_id % len(PALETTE)],
        "points": [],  # stored as image pixel positions
    })
    ss.active_id = ds_id
    ss.table_version += 1


def reset_for_new_image():
    ss.datasets = []
    ss.next_id = 0
    ss.calib_points = []
    ss.table_version = ss.get("table_version", 0) + 1
    ss.mode = MODE_CALIBRATE
    add_dataset()


def get_active():
    for ds in ss.datasets:
        if ds["id"] == ss.active_id:
            return ds
    return ss.datasets[0]


def delete_active_dataset():
    ss.datasets = [ds for ds in ss.datasets if ds["id"] != ss.active_id]
    if ss.datasets:
        ss.active_id = ss.datasets[-1]["id"]
    else:
        add_dataset()  # always keep one dataset to click into
    ss.table_version += 1


def undo_last_point():
    ds = get_active()
    if ds["points"]:
        ds["points"].pop()
        ss.table_version += 1


def clear_active_points():
    get_active()["points"] = []
    ss.table_version += 1


def undo_calibration_point():
    if ss.calib_points:
        ss.calib_points.pop()
        ss.mode = MODE_CALIBRATE


def reset_calibration():
    ss.calib_points = []
    ss.mode = MODE_CALIBRATE


if "datasets" not in ss:
    ss.image = None
    ss.image_key = None
    ss.last_click = None
    reset_for_new_image()

# A mode change requested during the previous run (see click handling below)
if "pending_mode" in ss:
    ss.mode = ss.pop("pending_mode")

# Keep dataset names in sync with their text boxes before anything is drawn
for ds in ss.datasets:
    ds["name"] = ss.get(f"name_{ds['id']}", ds["name"]).strip() or f"Dataset {ds['id'] + 1}"


# ----------------------------------------------------------------------------
# Sidebar: upload, axis setup, calibration values, zoom
# ----------------------------------------------------------------------------
st.title("📈 Plot Digitizer")

with st.sidebar:
    st.header("Image")
    uploaded = st.file_uploader(
        "Upload a graph image",
        type=["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp"],
    )

    st.header("Axes")
    x_label = st.text_input("x-axis name", "x").strip() or "x"
    y_label = st.text_input("y-axis name", "y").strip() or "y"
    col_a, col_b = st.columns(2)
    x_log = col_a.radio("x scale", ["Linear", "Log"], key="x_scale") == "Log"
    y_log = col_b.radio("y scale", ["Linear", "Log"], key="y_scale") == "Log"

    st.header("Calibration values")
    st.caption("The graph values at the three points you click on the image. "
               "For a log axis the origin value must be positive (e.g. 1 or 0.1).")
    col_a, col_b = st.columns(2)
    x0 = parse_number(col_a.text_input("x at origin", "0"))
    y0 = parse_number(col_b.text_input("y at origin", "0"))
    x1 = parse_number(col_a.text_input("x at x-axis point", "1"))
    y1 = parse_number(col_b.text_input("y at y-axis point", "1"))

    st.header("Zoom")
    zoom = st.slider("Zoom", 1.0, 10.0, 1.0, 0.5)
    if zoom > 1:
        pan_x = st.slider("Pan left ↔ right", 0.0, 1.0, 0.5, 0.01)
        pan_y = st.slider("Pan top ↔ bottom", 0.0, 1.0, 0.5, 0.01)
    else:
        pan_x = pan_y = 0.5

if uploaded is None:
    st.info(
        "Upload an image of a graph in the sidebar to begin.\n\n"
        "1. **Calibrate** - click the origin, a point on the x-axis and a point "
        "on the y-axis, and enter their values in the sidebar.\n"
        "2. **Pick data** - click each data point. Add a new dataset for each line.\n"
        "3. **Download** the results as a CSV file."
    )
    st.stop()

# Load the image once per upload, and start fresh when a new file arrives
image_key = f"{uploaded.name}-{uploaded.size}"
if ss.image_key != image_key:
    try:
        ss.image = load_image(uploaded)
    except Exception as err:  # unreadable / corrupt file
        st.error(f"Could not read that image: {err}")
        st.stop()
    ss.image_key = image_key
    reset_for_new_image()

image = ss.image
calibration, calib_error = build_calibration(
    ss.calib_points, x0, x1, y0, y1, x_log, y_log
)

# ----------------------------------------------------------------------------
# Controls above the image
# ----------------------------------------------------------------------------
st.radio("Clicking on the image will:", [MODE_CALIBRATE, MODE_PICK],
         key="mode", horizontal=True)

if ss.mode == MODE_CALIBRATE:
    n_done = len(ss.calib_points)
    if n_done < 3:
        st.info(f"**Calibration click {n_done + 1} of 3:** click {CALIB_STEPS[n_done]}.")
    else:
        st.success("All three calibration points are set. Check the values in the "
                   "sidebar, then switch to picking data points.")
    col_a, col_b, _ = st.columns([1, 1, 3])
    col_a.button("↩ Undo calibration click", on_click=undo_calibration_point,
                 disabled=n_done == 0)
    col_b.button("Reset calibration", on_click=reset_calibration,
                 disabled=n_done == 0)
else:
    col_a, col_b, col_c = st.columns([2, 2, 3])
    col_a.selectbox(
        "Active dataset (clicks are added to this one)",
        options=[ds["id"] for ds in ss.datasets],
        format_func=lambda i: next(d["name"] for d in ss.datasets if d["id"] == i),
        key="active_id",
    )
    active = get_active()
    col_b.text_input("Dataset name", value=active["name"], key=f"name_{active['id']}")
    col_c.markdown(
        f"<div style='margin-top:1.9rem'><span style='display:inline-block;"
        f"width:1rem;height:1rem;border-radius:50%;background:{active['color']};"
        f"vertical-align:middle'></span> &nbsp;{len(active['points'])} point(s) "
        f"in this dataset</div>",
        unsafe_allow_html=True,
    )
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.button("➕ New dataset", on_click=add_dataset)
    col_b.button("↩ Undo last point", on_click=undo_last_point,
                 disabled=not active["points"])
    col_c.button("Clear this dataset's points", on_click=clear_active_points,
                 disabled=not active["points"])
    col_d.button("🗑 Delete this dataset", on_click=delete_active_dataset)

if calib_error:
    st.error(calib_error)

# ----------------------------------------------------------------------------
# The clickable image
# ----------------------------------------------------------------------------
view, left, top, scale = render_view(image, zoom, pan_x, pan_y)
view = draw_overlays(view, left, top, scale, ss.datasets, ss.calib_points)
click = streamlit_image_coordinates(view, key="canvas")

# The component keeps returning the most recent click on every rerun, so only
# act when the click is different from the one already handled.
if click is not None and click != ss.last_click:
    ss.last_click = click
    # Convert from screen position back to a position on the original image
    shown_w = click.get("width") or view.width
    shown_h = click.get("height") or view.height
    img_x = left + click["x"] * (view.width / shown_w) / scale
    img_y = top + click["y"] * (view.height / shown_h) / scale

    if ss.mode == MODE_CALIBRATE:
        if len(ss.calib_points) < 3:
            ss.calib_points.append((img_x, img_y))
            if len(ss.calib_points) == 3:
                ss.pending_mode = MODE_PICK  # move on automatically
            st.rerun()
    elif calibration is not None:
        get_active()["points"].append((img_x, img_y))
        ss.table_version += 1
        st.rerun()

if ss.mode == MODE_PICK and calibration is None and not calib_error:
    st.warning("Finish the 3 calibration clicks before picking data points.")

# ----------------------------------------------------------------------------
# Live, editable data table + CSV download
# ----------------------------------------------------------------------------
st.subheader("Data")
if calibration is None:
    st.caption("The data table appears once the axes are calibrated.")
    st.stop()

table = build_table(ss.datasets, calibration, x_label, y_label)
st.caption(
    "This table updates as you click. You can also edit it: change a value to "
    "move its mark, add a row to add a point, or blank out both cells of a "
    "point to remove it. The marks on the image follow the table."
)
edited = st.data_editor(
    table,
    key=f"table_{ss.table_version}",  # new key = fresh editor after each change
    num_rows="dynamic",
    hide_index=True,
    column_config={
        col: st.column_config.NumberColumn(format="%.6g") for col in table.columns
    },
)

# Work out what the user changed and push it back to the stored click positions
new_points, incomplete, bad_value, changed = {}, False, False, False
for i, ds in enumerate(ss.datasets):
    xs = pd.to_numeric(edited.iloc[:, 2 * i], errors="coerce").tolist()
    ys = pd.to_numeric(edited.iloc[:, 2 * i + 1], errors="coerce").tolist()
    old_values = [pixel_to_data(calibration, *p) for p in ds["points"]]
    values = []
    for x, y in zip(xs, ys):
        x_blank, y_blank = math.isnan(x), math.isnan(y)
        if x_blank and y_blank:
            continue  # empty padding row
        if x_blank or y_blank:
            incomplete = True  # user is half-way through typing a point
            continue
        if (x_log and x <= 0) or (y_log and y <= 0):
            bad_value = True
            continue
        values.append((x, y))
    if len(values) != len(old_values) or (
        values and not np.allclose(values, old_values, rtol=1e-9, atol=0)
    ):
        changed = True
        # Keep the original pixel position for any point that was not edited
        pixels = []
        for k, (x, y) in enumerate(values):
            if k < len(old_values) and np.allclose((x, y), old_values[k], rtol=1e-9, atol=0):
                pixels.append(ds["points"][k])
            else:
                pixels.append(data_to_pixel(calibration, x, y))
        new_points[ds["id"]] = pixels

if bad_value:
    st.error("Values on a log axis must be greater than zero.")
elif incomplete:
    st.caption("Fill in both the x and y cell to add the new point.")
elif changed:
    for ds in ss.datasets:
        if ds["id"] in new_points:
            ds["points"] = new_points[ds["id"]]
    ss.table_version += 1
    st.rerun()

st.download_button(
    "⬇ Download CSV",
    data=build_csv(ss.datasets, calibration, x_label, y_label),
    file_name="digitized_data.csv",
    mime="text/csv",
)
