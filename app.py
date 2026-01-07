# ===================== IMPORTS =====================
import streamlit as st
import numpy as np
import pandas as pd
import pydicom
import os, hashlib, subprocess
from datetime import datetime
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from fpdf import FPDF

# ===================== CONFIG =====================
DATA_PATH = "./ACR_QC_Data"
REPORT_PATH = os.path.join(DATA_PATH, "Reports")
os.makedirs(REPORT_PATH, exist_ok=True)

USERS = {
    "physicist1": hashlib.sha256("acr123".encode()).hexdigest(),
    "physicist2": hashlib.sha256("acr456".encode()).hexdigest()
}

ACTION_LIMITS = {
    "B0_ppm": 0.5,
    "Uniformity": 87,
    "HCR": 1.0,
    "LowContrast": 3,
    "SNR": 40,
    "SliceThickness": 1.0,
    "Distortion": 1.0
}

ALL_TESTS = [
    "B0", "Uniformity", "HCR", "LowContrast",
    "SNR", "SliceThickness", "Distortion", "CF_Gain"
]

# ===================== AUTH =====================
def check_password(u, p):
    return u in USERS and hashlib.sha256(p.encode()).hexdigest() == USERS[u]

if "auth" not in st.session_state:
    st.session_state.auth = False
if "metrics" not in st.session_state:
    st.session_state.metrics = {}

if not st.session_state.auth:
    st.title("ACR MRI QC Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_password(u, p):
            st.session_state.auth = True
            st.session_state.user = u
            st.success(f"Logged in as {u}")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# ===================== SIDEBAR =====================
st.sidebar.markdown("### Session")
st.sidebar.write(f"User: **{st.session_state.user}**")
if st.sidebar.button("Logout"):
    st.session_state.auth = False
    st.session_state.metrics = {}
    st.experimental_rerun()

# ===================== TABS =====================
tabs = st.tabs([
    "B0 Homogeneity", "Uniformity", "HCR",
    "Low Contrast", "SNR / Ghosting",
    "Slice Thickness", "Distortion",
    "CF / Gain", "PDF Report"
])

# ==================================================
# =============== B0 HOMOGENEITY ====================
# ==================================================
with tabs[0]:
    st.header("B0 Field Homogeneity (Dual-TE)")

    files = st.file_uploader(
        "Upload dual-TE DICOM stack",
        type=["dcm"],
        accept_multiple_files=True
    )

    if st.button("Compute B0"):
        if not files:
            st.warning("Please upload DICOM files.")
            st.stop()

        # ---- Load and group by slice ----
        slices = {}
        for f in files:
            d = pydicom.dcmread(f)
            loc = round(float(d.SliceLocation), 2)
            slices.setdefault(loc, []).append(d)

        ppm_maps = []
        max_ppm_slices = []

        for loc, imgs in slices.items():
            if len(imgs) != 2:
                continue

            d1, d2 = sorted(imgs, key=lambda x: x.EchoTime)
            phase1 = d1.pixel_array.astype(float)
            phase2 = d2.pixel_array.astype(float)

            delta_phase = np.angle(np.exp(1j * (phase2 - phase1)))
            delta_te = abs(d2.EchoTime - d1.EchoTime) / 1000
            b0_hz = delta_phase / (2 * np.pi * delta_te)
            ppm = b0_hz / d1.ImagingFrequency

            # ---- Phantom mask ----
            mag = np.abs(d1.pixel_array)
            thresh = threshold_otsu(mag)
            mask = mag > thresh
            mask = binary_fill_holes(mask)

            lbl = label(mask)
            region = max(regionprops(lbl), key=lambda r: r.area)
            rr, cc = region.coords.T
            cy, cx = region.centroid
            r = np.sqrt((rr-cy)**2 + (cc-cx)**2).max()
            inner = ((np.indices(mask.shape)[0]-cy)**2 +
                     (np.indices(mask.shape)[1]-cx)**2) <= (0.85*r)**2

            ppm_roi = ppm[inner]
            ppm_maps.append(ppm * inner)
            max_ppm_slices.append(np.max(np.abs(ppm_roi)))

        global_max = max(max_ppm_slices)
        status = "PASS" if global_max <= ACTION_LIMITS["B0_ppm"] else "FAIL"

        st.session_state.metrics["B0"] = {
            "Max_ppm": round(global_max, 3),
            "Status": status
        }

        st.success(f"Global max ppm: {global_max:.3f} → {status}")

        fig, ax = plt.subplots()
        im = ax.imshow(ppm_maps[0], cmap="seismic")
        plt.colorbar(im, label="ppm")
        st.pyplot(fig)

# ==================================================
# =============== MANUAL T1 / T2 TESTS ==============
# ==================================================
def dual_input(test, limit, label):
    st.subheader(test)
    t1 = st.number_input(f"{label} T1", key=f"{test}_t1")
    t2 = st.number_input(f"{label} T2", key=f"{test}_t2")
    if st.button(f"Submit {test}"):
        p1 = t1 >= limit
        p2 = t2 >= limit
        st.session_state.metrics[test] = {
            "T1": t1, "T2": t2,
            "Status": "PASS" if p1 and p2 else "FAIL"
        }
        st.success(st.session_state.metrics[test]["Status"])

with tabs[1]:
    dual_input("Uniformity", ACTION_LIMITS["Uniformity"], "Uniformity (%)")

with tabs[2]:
    dual_input("HCR", ACTION_LIMITS["HCR"], "Resolution (lp/mm)")

with tabs[3]:
    dual_input("LowContrast", ACTION_LIMITS["LowContrast"], "Objects visible")

with tabs[4]:
    dual_input("SNR", ACTION_LIMITS["SNR"], "SNR")

# ==================================================
# =============== SINGLE INPUT TESTS ================
# ==================================================
with tabs[5]:
    v = st.number_input("Slice thickness error (mm)")
    if st.button("Submit"):
        st.session_state.metrics["SliceThickness"] = {
            "Error_mm": v,
            "Status": "PASS" if abs(v) <= ACTION_LIMITS["SliceThickness"] else "FAIL"
        }

with tabs[6]:
    v = st.number_input("Max distortion (mm)")
    if st.button("Submit"):
        st.session_state.metrics["Distortion"] = {
            "Error_mm": v,
            "Status": "PASS" if v <= ACTION_LIMITS["Distortion"] else "FAIL"
        }

with tabs[7]:
    cf = st.number_input("Center Frequency (MHz)")
    gain = st.number_input("Gain")
    if st.button("Submit"):
        st.session_state.metrics["CF_Gain"] = {
            "CF": cf, "Gain": gain, "Status": "N/A"
        }

# ==================================================
# =================== PDF REPORT ===================
# ==================================================
with tabs[8]:
    st.header("Generate PDF Report")

    if st.button("Generate PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "ACR MRI QC Report", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Date: {datetime.now().date()}", ln=True)
        pdf.cell(0, 8, f"Performed by: {st.session_state.user}", ln=True)
        pdf.ln(5)

        for t in ALL_TESTS:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, t, ln=True)
            pdf.set_font("Arial", "", 11)
            if t in st.session_state.metrics:
                for k, v in st.session_state.metrics[t].items():
                    pdf.cell(0, 6, f"{k}: {v}", ln=True)
            else:
                pdf.cell(0, 6, "Not performed", ln=True)
            pdf.ln(2)

        path = os.path.join(REPORT_PATH, f"ACR_QC_{datetime.now().date()}.pdf")
        pdf.output(path)
        st.success("PDF generated")
        st.download_button("Download PDF", open(path, "rb"), file_name=os.path.basename(path))
