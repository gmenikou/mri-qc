# app.py
import streamlit as st
import os
import pandas as pd
import hashlib
from datetime import datetime
from fpdf import FPDF

# ======================= CONFIG =======================
DATA_PATH = "./ACR_QC_Data"
REPORTS_PATH = os.path.join(DATA_PATH, "Reports")
os.makedirs(REPORTS_PATH, exist_ok=True)

ACTION_LIMITS = {
    "B0_ppm": 0.5,
    "Uniformity_percent": 87,
    "HCR_lp": 1.0,
    "SliceThickness_mm": 1.0,
    "Distortion_mm": 1.0,
    "LowContrast_min": 3,
    "SNR_min": 40
}

ALL_TESTS = [
    "B0",
    "Uniformity",
    "HCR",
    "SliceThickness",
    "Distortion",
    "LowContrast",
    "SNR_Ghosting",
    "CF_Gain"
]

USERS = {
    "physicist": hashlib.sha256("acr123".encode()).hexdigest()
}

# ======================= HELPERS =======================
def check_password(user, pw):
    return user in USERS and hashlib.sha256(pw.encode()).hexdigest() == USERS[user]

def overall_status(pass_t1, pass_t2):
    return "PASS" if pass_t1 and pass_t2 else "FAIL"

def save_csv(test, data):
    folder = os.path.join(DATA_PATH, test)
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame([data])
    path = os.path.join(folder, f"{datetime.now().date()}_{test}.csv")
    df.to_csv(path, index=False)

def generate_pdf(metrics, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)

    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "ACR MRI QC Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, "Scanner: Philips 1.5T", ln=True)
    pdf.cell(0, 8, "Phantom: 30 cm Sphere", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.now().date()}", ln=True)

    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Summary", ln=True)

    for test in ALL_TESTS:
        pdf.set_font("Arial", "", 12)
        if test in metrics:
            pdf.cell(0, 8, f"{test}: {metrics[test]['Status']}", ln=True)
        else:
            pdf.cell(0, 8, f"{test}: NOT PERFORMED", ln=True)

    for test in ALL_TESTS:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, test, ln=True)
        pdf.set_font("Arial", "", 12)
        if test in metrics:
            for k, v in metrics[test].items():
                pdf.cell(0, 8, f"{k}: {v}", ln=True)
        else:
            pdf.cell(0, 8, "Test not performed", ln=True)

    path = os.path.join(REPORTS_PATH, filename)
    pdf.output(path)
    return path

# ======================= SESSION =======================
if "auth" not in st.session_state:
    st.session_state.auth = False
if "metrics" not in st.session_state:
    st.session_state.metrics = {}

# ======================= LOGIN ========================
if not st.session_state.auth:
    st.title("ACR MRI QC Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_password(u, p):
            st.session_state.auth = True
            st.success("Logged in")
        else:
            st.error("Invalid credentials")
    st.stop()

# ======================= TABS =========================
tabs = st.tabs([
    "B0 Homogeneity",
    "Uniformity",
    "HCR",
    "Slice Thickness",
    "Distortion",
    "Low Contrast",
    "SNR / Ghosting",
    "Trend",
    "PDF",
    "Settings",
    "CF / Gain"
])

metrics = st.session_state.metrics

# ======================= B0 ===========================
with tabs[0]:
    st.header("B0 Field Homogeneity (Dual-TE)")
    st.info("Automatic dual-TE phase-based B0 mapping will be implemented here.")
    st.warning("Currently placeholder – no computation yet.")

# ======================= UNIFORMITY ===================
with tabs[1]:
    st.header("Image Uniformity (T1 & T2)")
    t1 = st.number_input("Uniformity T1 (%)", 0.0, 100.0)
    t2 = st.number_input("Uniformity T2 (%)", 0.0, 100.0)
    if st.button("Save Uniformity"):
        p1 = t1 >= ACTION_LIMITS["Uniformity_percent"]
        p2 = t2 >= ACTION_LIMITS["Uniformity_percent"]
        metrics["Uniformity"] = {
            "T1_%": t1,
            "T2_%": t2,
            "Status": overall_status(p1, p2)
        }
        save_csv("Uniformity", metrics["Uniformity"])
        st.success(metrics["Uniformity"]["Status"])

# ======================= HCR ==========================
with tabs[2]:
    st.header("High Contrast Resolution (T1 & T2)")
    t1 = st.number_input("HCR T1 (lp/mm)", 0.0)
    t2 = st.number_input("HCR T2 (lp/mm)", 0.0)
    if st.button("Save HCR"):
        p1 = t1 >= ACTION_LIMITS["HCR_lp"]
        p2 = t2 >= ACTION_LIMITS["HCR_lp"]
        metrics["HCR"] = {
            "T1_lp": t1,
            "T2_lp": t2,
            "Status": overall_status(p1, p2)
        }
        save_csv("HCR", metrics["HCR"])
        st.success(metrics["HCR"]["Status"])

# ======================= SLICE ========================
with tabs[3]:
    st.header("Slice Thickness")
    v = st.number_input("Measured slice thickness (mm)", 0.0)
    if st.button("Save Slice Thickness"):
        metrics["SliceThickness"] = {
            "Measured_mm": v,
            "Status": "PASS" if abs(v - 5.0) <= ACTION_LIMITS["SliceThickness_mm"] else "FAIL"
        }
        save_csv("SliceThickness", metrics["SliceThickness"])
        st.success(metrics["SliceThickness"]["Status"])

# ======================= DISTORT ======================
with tabs[4]:
    st.header("Geometric Distortion")
    v = st.number_input("Max distortion (mm)", 0.0)
    if st.button("Save Distortion"):
        metrics["Distortion"] = {
            "Max_mm": v,
            "Status": "PASS" if v <= ACTION_LIMITS["Distortion_mm"] else "FAIL"
        }
        save_csv("Distortion", metrics["Distortion"])
        st.success(metrics["Distortion"]["Status"])

# ======================= LOW CONTR ====================
with tabs[5]:
    st.header("Low Contrast Detectability (T1 & T2)")
    t1 = st.number_input("Objects visible T1", 0)
    t2 = st.number_input("Objects visible T2", 0)
    if st.button("Save Low Contrast"):
        p1 = t1 >= ACTION_LIMITS["LowContrast_min"]
        p2 = t2 >= ACTION_LIMITS["LowContrast_min"]
        metrics["LowContrast"] = {
            "T1_objects": t1,
            "T2_objects": t2,
            "Status": overall_status(p1, p2)
        }
        save_csv("LowContrast", metrics["LowContrast"])
        st.success(metrics["LowContrast"]["Status"])

# ======================= SNR ==========================
with tabs[6]:
    st.header("SNR / Ghosting (T1 & T2)")
    t1 = st.number_input("SNR T1", 0.0)
    t2 = st.number_input("SNR T2", 0.0)
    if st.button("Save SNR"):
        p1 = t1 >= ACTION_LIMITS["SNR_min"]
        p2 = t2 >= ACTION_LIMITS["SNR_min"]
        metrics["SNR_Ghosting"] = {
            "T1_SNR": t1,
            "T2_SNR": t2,
            "Status": overall_status(p1, p2)
        }
        save_csv("SNR_Ghosting", metrics["SNR_Ghosting"])
        st.success(metrics["SNR_Ghosting"]["Status"])

# ======================= TREND ========================
with tabs[7]:
    st.header("Trend Analysis")
    st.info("Historical trend plots from CSVs can be added here.")

# ======================= PDF ==========================
with tabs[8]:
    st.header("Generate PDF Report")
    if st.button("Generate PDF"):
        name = f"{datetime.now().date()}_ACR_QC_Report.pdf"
        path = generate_pdf(metrics, name)
        st.success(f"Saved to {path}")
        st.download_button("Download PDF", open(path, "rb"), name)

# ======================= SETTINGS =====================
with tabs[9]:
    st.header("Settings")
    st.info("Manual measurements must follow ACR window/level guidance.")

# ======================= CF ===========================
with tabs[10]:
    st.header("Center Frequency / Gain")
    cf = st.number_input("Center Frequency (MHz)", 0.0)
    gain = st.number_input("Receiver Gain", 0.0)
    if st.button("Save CF/Gain"):
        metrics["CF_Gain"] = {"CF": cf, "Gain": gain, "Status": "N/A"}
        save_csv("CF_Gain", metrics["CF_Gain"])
        st.success("Saved")
