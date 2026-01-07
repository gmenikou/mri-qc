import streamlit as st
import numpy as np
import pydicom
from skimage import filters, measure
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os

# -------------------- CONFIG --------------------
st.set_page_config(page_title="MRI ACR QC", layout="wide")

ACTION_LIMITS = {
    "B0_ppm": 0.5,
    "Geometry_mm": 2.0,
    "SliceThickness_mm": 0.7,
    "Uniformity_percent": 82,
    "Ghosting_percent": 0.025
}

if "metrics" not in st.session_state:
    st.session_state.metrics = {}

# -------------------- UTILS --------------------
def load_dicom_stack(files):
    slices = []
    for f in files:
        ds = pydicom.dcmread(f)
        slices.append((getattr(ds, "InstanceNumber", 0),
                       ds.pixel_array.astype(np.float32)))
    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices])

def phantom_mask(slice_img):
    smooth = filters.gaussian(slice_img, sigma=1)
    thresh = filters.threshold_otsu(smooth)
    binary = smooth > thresh
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    if not regions:
        return np.zeros_like(slice_img, dtype=bool)
    largest = max(regions, key=lambda r: r.area)
    return labeled == largest.label

def shrink_to_85(mask):
    dist = distance_transform_edt(mask)
    if dist.max() == 0:
        return mask
    return dist >= 0.15 * dist.max()

def compute_b0(te1_stack, te2_stack, delta_te):
    field_maps = []
    max_ppm_slices = []

    for i in range(te1_stack.shape[0]):
        m = phantom_mask(te1_stack[i])
        roi = shrink_to_85(m)

        phase_diff = np.angle(te2_stack[i] / (te1_stack[i] + 1e-12))
        ppm = phase_diff / (2 * np.pi * delta_te * 1e6)

        ppm_roi = ppm * roi
        field_maps.append(ppm_roi)

        if roi.sum():
            max_ppm_slices.append(np.max(np.abs(ppm_roi)))

    return field_maps, max(max_ppm_slices) if max_ppm_slices else None

def plot_ppm(ppm):
    fig, ax = plt.subplots()
    im = ax.imshow(ppm, cmap="RdBu", origin="lower")
    plt.colorbar(im, ax=ax, label="ppm")
    ax.axis("off")
    return fig

# -------------------- PDF --------------------
def generate_pdf(metrics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "MRI ACR QC Report", ln=True)

    for test, data in metrics.items():
        pdf.ln(5)
        pdf.set_font("Arial", style="B", size=11)
        pdf.cell(0, 8, test, ln=True)
        pdf.set_font("Arial", size=10)

        if data is None:
            pdf.cell(0, 6, "NOT PERFORMED", ln=True)
        else:
            for k, v in data.items():
                pdf.cell(0, 6, f"{k}: {v}", ln=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# -------------------- UI --------------------
st.title("MRI ACR Quality Control")

tabs = st.tabs([
    "B0 Homogeneity",
    "Geometry",
    "Slice Thickness",
    "Image Uniformity",
    "Ghosting",
    "Coils / Gains",
    "Final Report"
])

# ======================================================
# B0 TAB
# ======================================================
with tabs[0]:
    st.subheader("B0 Field Homogeneity (Dual-TE)")

    te1 = st.file_uploader("Upload TE1 DICOM stack", accept_multiple_files=True, type="dcm")
    te2 = st.file_uploader("Upload TE2 DICOM stack", accept_multiple_files=True, type="dcm")
    delta_te = st.number_input("ΔTE (seconds)", value=0.01, format="%.4f")

    if st.button("Compute B0"):
        if not te1 or not te2:
            st.warning("Please upload both TE1 and TE2 stacks.")
        else:
            s1 = load_dicom_stack(te1)
            s2 = load_dicom_stack(te2)
            maps, max_ppm = compute_b0(s1, s2, delta_te)

            status = "PASS" if max_ppm <= ACTION_LIMITS["B0_ppm"] else "FAIL"
            st.session_state.metrics["B0 Homogeneity"] = {
                "Max ppm": round(max_ppm, 3),
                "Limit": ACTION_LIMITS["B0_ppm"],
                "Status": status
            }

            st.success(f"Max B0 shift: {max_ppm:.3f} ppm ({status})")

            for i, m in enumerate(maps):
                if np.any(m):
                    st.text(f"Slice {i+1}")
                    st.pyplot(plot_ppm(m))

# ======================================================
# MANUAL ACR TABS
# ======================================================
def manual_tab(tab, name, unit, limit):
    with tab:
        val = st.number_input(f"Measured value ({unit})", value=0.0)
        performed = st.checkbox("Test performed", value=True)

        if st.button(f"Save {name}"):
            if not performed:
                st.session_state.metrics[name] = None
                st.info("Marked as NOT PERFORMED")
            else:
                status = "PASS" if val <= limit else "FAIL"
                st.session_state.metrics[name] = {
                    "Value": val,
                    "Limit": limit,
                    "Status": status
                }
                st.success(f"{status}")

manual_tab(tabs[1], "Geometry", "mm", ACTION_LIMITS["Geometry_mm"])
manual_tab(tabs[2], "Slice Thickness", "mm", ACTION_LIMITS["SliceThickness_mm"])
manual_tab(tabs[3], "Image Uniformity", "%", ACTION_LIMITS["Uniformity_percent"])
manual_tab(tabs[4], "Ghosting", "%", ACTION_LIMITS["Ghosting_percent"])

# ======================================================
# COILS TAB
# ======================================================
with tabs[5]:
    st.subheader("Coils / Gains")
    notes = st.text_area("CF / Gain notes")
    if st.button("Save Coils"):
        st.session_state.metrics["Coils"] = {"Notes": notes}
        st.success("Saved")

# ======================================================
# REPORT
# ======================================================
with tabs[6]:
    st.subheader("Final Report")

    if st.button("Generate PDF"):
        pdf_path = generate_pdf(st.session_state.metrics)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", f, file_name="MRI_ACR_QC_Report.pdf")
