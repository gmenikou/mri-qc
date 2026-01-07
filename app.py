# app.py
import streamlit as st
import os, pandas as pd, hashlib, subprocess
from datetime import datetime
from fpdf import FPDF
import numpy as np
import pydicom
from skimage import filters, measure
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

# ------------------- Settings -------------------
DATA_PATH = "./ACR_QC_Data"
REPORTS_PATH = os.path.join(DATA_PATH, "Reports")
os.makedirs(REPORTS_PATH, exist_ok=True)

ALL_TESTS = [
    "B0", "Uniformity", "HCR", "SliceThickness", "Distortion",
    "LowContrast", "SNR_Ghosting", "CF_Gain"
]

ACTION_LIMITS = {
    "B0_ppm": 0.5,
    "Uniformity_percent": 87,
    "HCR_lp": 1.0,
    "SliceThickness_mm": 1.0,
    "Distortion_mm": 1.0,
    "LowContrast_min": 3,
    "SNR_min": 40
}

USERS = {
    "physicist1": hashlib.sha256("password1".encode()).hexdigest(),
    "physicist2": hashlib.sha256("password2".encode()).hexdigest()
}

USER_REPOS = {
    "physicist1": "./ACR_QC_Repo_Physicist1",
    "physicist2": "./ACR_QC_Repo_Physicist2"
}

# ------------------- Helper Functions -------------------
def check_password(username, password):
    return username in USERS and hashlib.sha256(password.encode()).hexdigest() == USERS[username]

def save_metrics_csv(test_name, df):
    folder = os.path.join(DATA_PATH, test_name)
    os.makedirs(folder, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(folder, f"{date_str}_{test_name}.csv")
    df.to_csv(path, index=False)
    return path

def generate_pdf(metrics_dict, filename, save_dir):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cover page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, "ACR MRI QC Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Scanner: Philips 1.5T", ln=True)
    pdf.cell(0, 10, f"Phantom: 30 cm Sphere", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(0, 10, "QC performed according to ACR guidelines.", ln=True)
    
    # Summary
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Summary of PASS/FAIL per Test", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    for test in ALL_TESTS:
        if test in metrics_dict:
            status = metrics_dict[test].get('Status', 'N/A')
            pdf.set_text_color(0,128,0) if status=="PASS" else pdf.set_text_color(255,0,0)
            pdf.cell(0, 6, f"{test}: {status}", ln=True)
        else:
            pdf.set_text_color(255,0,0)
            pdf.cell(0,6,f"{test}: Not performed", ln=True)
        pdf.set_text_color(0,0,0)
    
    # Detailed per-test pages
    for test in ALL_TESTS:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"{test} Metrics", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", '', 12)
        if test in metrics_dict:
            for key,val in metrics_dict[test].items():
                if key != "Status":
                    pdf.cell(0, 6, f"{key}: {val}", ln=True)
            pdf.ln(2)
            pdf.cell(0,6,f"Status: {metrics_dict[test]['Status']}", ln=True)
        else:
            pdf.set_text_color(255,0,0)
            pdf.multi_cell(0,6,"This test was not performed during this session.")
            pdf.set_text_color(0,0,0)
    
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    pdf.output(path)
    return path

# ------------------- B0 Helper Functions -------------------
def load_dicom_stack(files):
    slices = []
    for f in files:
        ds = pydicom.dcmread(f)
        slices.append((getattr(ds,'InstanceNumber',0), ds.pixel_array.astype(np.float32)))
    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices])

def get_phantom_mask(slice_img):
    slice_smooth = filters.gaussian(slice_img, sigma=1)
    thresh = filters.threshold_otsu(slice_smooth)
    binary = slice_smooth > thresh
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    if len(regions)==0:
        return np.zeros_like(slice_img, dtype=bool)
    largest_region = max(regions, key=lambda r: r.area)
    mask = labeled == largest_region.label
    return mask

def shrink_mask(mask, fraction=0.85):
    dist = distance_transform_edt(mask)
    if dist.max()==0:
        return mask
    inner_mask = dist >= (1-fraction)*dist.max()
    return inner_mask

def compute_field_map(stack_te1, stack_te2, mask_fraction=0.85, delta_TE=0.01):
    max_ppm_values = []
    field_maps = []
    for i in range(stack_te1.shape[0]):
        slice1 = stack_te1[i]
        slice2 = stack_te2[i]
        mask = get_phantom_mask(slice1)
        inner_mask = shrink_mask(mask, fraction=mask_fraction)
        delta_phase = np.angle(slice2 / (slice1 + 1e-12))
        ppm = delta_phase / (2*np.pi*delta_TE*1e6)
        field_maps.append(ppm*inner_mask)
        if inner_mask.sum()>0:
            max_ppm_values.append((ppm*inner_mask).max())
    return field_maps, max(max_ppm_values) if max_ppm_values else 0

def plot_field_map(ppm_map):
    fig, ax = plt.subplots()
    im = ax.imshow(ppm_map, cmap='RdBu', origin='lower')
    plt.colorbar(im, ax=ax, label='ppm')
    ax.axis('off')
    return fig

# ------------------- Session State Initialization -------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'metrics_store' not in st.session_state:
    st.session_state.metrics_store = {}
if 'user_repo_path' not in st.session_state:
    st.session_state.user_repo_path = "./ACR_QC_Repo"

metrics_store = st.session_state.metrics_store
user_repo_path = st.session_state.user_repo_path

# ------------------- Authentication -------------------
if not st.session_state.authenticated:
    st.title("ACR QC Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_clicked = st.button("Login")

    if login_clicked:
        if check_password(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_repo_path = USER_REPOS.get(username, "./ACR_QC_Repo")
            st.success(f"Welcome, {username}!")
        else:
            st.error("Incorrect username or password")

# ------------------- Main App -------------------
if st.session_state.authenticated:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    user_repo_path = st.session_state.user_repo_path
    os.makedirs(user_repo_path, exist_ok=True)

    tabs = st.tabs([
        "B0 Homogeneity","Uniformity","HCR","Slice Thickness",
        "Distortion","Low-Contrast","SNR/Ghosting","Trend","PDF","Settings","CF/Gain"
    ])

    # ------------------- B0 Homogeneity -------------------
    with tabs[0]:
        st.header("B0 Field Homogeneity")
        te1_files = st.file_uploader("Upload TE1 DICOM stack", type=["dcm"], accept_multiple_files=True)
        te2_files = st.file_uploader("Upload TE2 DICOM stack", type=["dcm"], accept_multiple_files=True)
        if st.button("Compute B0 Metrics"):
            if te1_files and te2_files:
                stack_te1 = load_dicom_stack(te1_files)
                stack_te2 = load_dicom_stack(te2_files)
                field_maps, max_ppm = compute_field_map(stack_te1, stack_te2)
                metrics_store['B0'] = {'Max_ppm': max_ppm, 'Status': "PASS" if max_ppm<=ACTION_LIMITS['B0_ppm'] else "FAIL"}
                st.session_state.metrics_store = metrics_store
                st.success(f"Max B0 shift: {max_ppm:.3f} ppm | Status: {metrics_store['B0']['Status']}")
                st.subheader("Field Maps per slice")
                for i, fmap in enumerate(field_maps):
                    if np.any(fmap):
                        st.text(f"Slice {i+1}")
                        fig = plot_field_map(fmap)
                        st.pyplot(fig)
            else:
                st.warning("Please upload both TE1 and TE2 DICOM stacks before computing.")

    # ------------------- Uniformity -------------------
    with tabs[1]:
        st.header("Image Uniformity")
        if 'Uniformity' not in metrics_store:
            st.info("Not filled yet")
        else:
            st.success(f"Current metrics: {metrics_store['Uniformity']}")
        t1_val = st.number_input("T1 (%)",0.0,100.0,key="uni_t1")
        t2_val = st.number_input("T2 (%)",0.0,100.0,key="uni_t2")
        if st.button("Submit Uniformity"):
            status = "PASS" if min(t1_val,t2_val) >= ACTION_LIMITS['Uniformity_percent'] else "FAIL"
            metrics_store['Uniformity'] = {'T1':t1_val,'T2':t2_val,'Status':status}
            st.session_state.metrics_store = metrics_store
            st.success(f"Uniformity recorded. Status: {status}")
            save_metrics_csv("Uniformity", pd.DataFrame([metrics_store['Uniformity']]))

    # ------------------- HCR -------------------
    with tabs[2]:
        st.header("High-Contrast Resolution")
        if 'HCR' not in metrics_store:
            st.info("Not filled yet")
        else:
            st.success(f"Current metrics: {metrics_store['HCR']}")
        hcr_val = st.number_input("Highest resolved line-pair (mm)",0.0,key="hcr")
        if st.button("Submit HCR"):
            status = "PASS" if hcr_val >= ACTION_LIMITS['HCR_lp'] else "FAIL"
            metrics_store['HCR'] = {'Resolved_lp':hcr_val,'Status':status}
            st.session_state.metrics_store = metrics_store
            st.success(f"HCR recorded. Status: {status}")
            save_metrics_csv("HCR", pd.DataFrame([metrics_store['HCR']]))

    # ------------------- Other tabs (SliceThickness, Distortion, LowContrast, SNR/Ghosting, CF/Gain, PDF, Settings, Trend) -------------------
    # Keep your previous implementations for these tabs.
