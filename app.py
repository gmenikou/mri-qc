# app.py
import streamlit as st
import os, pandas as pd, hashlib, subprocess
from datetime import datetime
from fpdf import FPDF
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage import filters, measure, morphology

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

# ------------------- B0 Functions -------------------
def load_dual_echo_dicom(files):
    datasets = [pydicom.dcmread(f) for f in files]
    datasets.sort(key=lambda x: int(x.InstanceNumber))
    te_values = np.array([float(ds.EchoTime) for ds in datasets])
    unique_te = np.unique(te_values)
    if len(unique_te) !=2:
        st.error("Dual-TE DICOM stack required (2 unique TEs).")
        return None,None,None
    te1, te2 = unique_te
    imgs_te1 = np.array([ds.pixel_array for ds in datasets if ds.EchoTime==te1])
    imgs_te2 = np.array([ds.pixel_array for ds in datasets if ds.EchoTime==te2])
    return imgs_te1, imgs_te2, (te1, te2)

def phantom_mask(img, fraction=0.85):
    thresh = filters.threshold_otsu(img)
    bw = img>thresh
    label = measure.label(bw)
    props = measure.regionprops(label)
    largest = max(props, key=lambda x:x.area)
    mask = (label==largest.label)
    radius = int(np.sqrt(largest.area/np.pi))
    erode_radius = int(radius*(1-fraction))
    from skimage.morphology import disk
    mask = morphology.erosion(mask,disk(erode_radius))
    return mask

def compute_b0_ppm(img1, img2, te1, te2):
    delta_te = (te2 - te1)/1000.0
    img1, img2 = img1.astype(np.complex64), img2.astype(np.complex64)
    phase_diff = np.angle(img2 / img1)
    b0_ppm = phase_diff / (2*np.pi*delta_te) / 42.577e6 * 1e6
    return b0_ppm

# ------------------- Session State -------------------
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
    if st.button("Login"):
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

    # ------------------- B0 -------------------
    with tabs[0]:
        st.header("B0 Field Homogeneity (30cm Sphere Phantom)")
        dicoms = st.file_uploader("Upload dual-TE DICOM stack (all slices)", type=["dcm"], accept_multiple_files=True)
        if st.button("Compute B0 Metrics"):
            if not dicoms:
                st.error("Please upload dual-TE DICOM stack before computing metrics.")
            else:
                imgs1, imgs2, (te1, te2) = load_dual_echo_dicom(dicoms)
                if imgs1 is not None:
                    max_ppm_slices = []
                    fig, axes = plt.subplots(1,len(imgs1), figsize=(3*len(imgs1),3))
                    if len(imgs1)==1:
                        axes=[axes]
                    for i,(im1,im2,ax) in enumerate(zip(imgs1,imgs2,axes)):
                        mask = phantom_mask(im1)
                        b0_map = compute_b0_ppm(im1, im2, te1, te2)
                        b0_masked = b0_map*mask
                        max_ppm_slices.append(np.max(np.abs(b0_masked)))
                        im = ax.imshow(b0_masked, cmap='jet')
                        ax.set_title(f"Slice {i+1}")
                        ax.axis('off')
                        fig.colorbar(im, ax=ax, fraction=0.046)
                    st.pyplot(fig)
                    max_ppm = np.max(max_ppm_slices)
                    metrics_store['B0'] = {'Max_ppm':max_ppm, 'Status':"PASS" if max_ppm <= ACTION_LIMITS['B0_ppm'] else "FAIL"}
                    st.success(f"Max B0: {max_ppm:.3f} ppm, Status: {metrics_store['B0']['Status']}")
                    save_metrics_csv("B0", pd.DataFrame([metrics_store['B0']]))

    # ------------------- Uniformity -------------------
    with tabs[1]:
        st.header("Image Uniformity (T1/T2)")
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
        hcr_val = st.number_input("Highest resolved line-pair (mm)",0.0,key="hcr")
        if st.button("Submit HCR"):
            status = "PASS" if hcr_val >= ACTION_LIMITS['HCR_lp'] else "FAIL"
            metrics_store['HCR'] = {'Resolved_lp':hcr_val,'Status':status}
            st.session_state.metrics_store = metrics_store
            st.success(f"HCR recorded. Status: {status}")
            save_metrics_csv("HCR", pd.DataFrame([metrics_store['HCR']]))

    # ------------------- Slice Thickness -------------------
    with tabs[3]:
        st.header("Slice Thickness")
        st_val = st.number_input("Measured slice thickness (mm)",0.0,key="slice_thick")
        if st.button("Submit Slice Thickness"):
            status = "PASS" if abs(st_val-5.0) <= ACTION_LIMITS['SliceThickness_mm'] else "FAIL"
            metrics_store['SliceThickness'] = {'Measured_mm':st_val,'Status':status}
            st.session_state.metrics_store = metrics_store
            st.success(f"Slice Thickness recorded. Status: {status}")
            save_metrics_csv("SliceThickness", pd.DataFrame([metrics_store['SliceThickness']]))

    # ------------------- Distortion -------------------
    with tabs[4]:
        st.header("Geometric Distortion")
        dist_val = st.number_input("Max distortion (mm)",0.0,key="distortion")
        if st.button("Submit Distortion"):
            status = "PASS" if dist_val <= ACTION_LIMITS['Distortion_mm'] else "FAIL"
            metrics_store['Distortion'] = {'MaxDistortion_mm':dist_val,'Status':status}
            st.session_state.metrics_store = metrics_store
            st.success(f"Distortion recorded. Status: {status}")
            save_metrics_csv("Distortion", pd.DataFrame([metrics_store['Distortion']]))

    # ------------------- Low Contrast -------------------
    with tabs[5]:
        st.header("Low-Contrast Objects")
        lc_val = st.number_input("Number of visible low-contrast objects",0,key="lowcontrast")
        if st.button("Submit Low Contrast"):
            status = "PASS" if lc_val >= ACTION_LIMITS['LowContrast_min'] else "FAIL"
            metrics_store['LowContrast'] = {'ObjectsVisible':lc_val,'Status':status}
            st.session_state.metrics_store = metrics_store
            st.success(f"Low-Contrast recorded. Status: {status}")
            save_metrics_csv("LowContrast", pd.DataFrame([metrics_store['LowContrast']]))

    # ------------------- SNR/Ghosting -------------------
    with tabs[6]:
        st.header("SNR/Ghosting")
        snr_val = st.number_input("SNR",0.0,key="snr")
        if st.button("Submit SNR/Ghosting"):
            status = "PASS" if snr_val >= ACTION_LIMITS['SNR_min'] else "FAIL"
            metrics_store['SNR_Ghosting'] = {'SNR':snr_val,'Status':status}
            st.session_state.metrics_store = metrics_store
            st.success(f"SNR/Ghosting recorded. Status: {status}")
            save_metrics_csv("SNR_Ghosting", pd.DataFrame([metrics_store['SNR_Ghosting']]))

    # ------------------- CF/Gain -------------------
    with tabs[10]:
        st.header("Center Frequency / Gain")
        cf_val = st.number_input("Center Frequency (MHz)",0.0,key="cf")
        gain_val = st.number_input("Receiver Gain",0.0,key="gain")
        if st.button("Submit CF/Gain"):
            metrics_store['CF_Gain'] = {'CF':cf_val,'Gain':gain_val,'Status':'N/A'}
            st.session_state.metrics_store = metrics_store
            st.success("CF/Gain recorded")
            save_metrics_csv("CF_Gain", pd.DataFrame([metrics_store['CF_Gain']]))

    # ------------------- Trend -------------------
    with tabs[7]:
        st.header("Trend Analysis")
        st.info("Trend plots from historical CSVs (optional)")

    # ------------------- PDF -------------------
    with tabs[8]:
        st.header("Generate Full PDF Report")
        save_option = st.radio("Save PDF:", ["Locally","User-defined Repo"], index=0)
        report_name = f"{datetime.now().strftime('%Y-%m-%d')}_ACR_QC_Report.pdf"
        if st.button("Generate PDF"):
            save_dir = REPORTS_PATH if save_option=="Locally" else os.path.join(user_repo_path,"Reports")
            pdf_path = generate_pdf(metrics_store, report_name, save_dir)
            st.success(f"PDF saved to: {pdf_path}")
            st.download_button("Download PDF", data=open(pdf_path,"rb").read(), file_name=report_name)
            if st.checkbox("Commit PDF to your GitHub repo"):
                try:
                    subprocess.run(["git","add",pdf_path], check=True)
                    subprocess.run(["git","commit","-m",f"{st.session_state.username} added QC report {report_name}"], check=True)
                    subprocess.run(["git","push"], check=True)
                    st.success("PDF committed and pushed to your repo!")
                except subprocess.CalledProcessError as e:
                    st.error(f"Git commit failed: {e}")

    # ------------------- Settings -------------------
    with tabs[9]:
        st.header("Settings / Help")
        st.text(f"Local data folder: {DATA_PATH}")
        st.text(f"Local reports folder: {REPORTS_PATH}")
        st.info("For tests requiring manual measurements, adjust W/WL and zoom on images per ACR guidelines.")
        new_repo = st.text_input("Optional: Change user-defined GitHub repo", value=user_repo_path)
        if new_repo != user_repo_path:
            st.session_state.user_repo_path = new_repo
            st.success(f"User repo path updated to: {new_repo}")
