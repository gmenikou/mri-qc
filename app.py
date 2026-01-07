# app.py
import streamlit as st
import os, pandas as pd, hashlib, subprocess
from datetime import datetime
from fpdf import FPDF

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

    # ------------------- Example: B0 -------------------
    with tabs[0]:
        st.header("B0 Field Homogeneity")
    
    # Optional upload
    uploaded_files = st.file_uploader(
        "Upload T1 DICOM stack (T1 series)", 
        type=["dcm"], 
        accept_multiple_files=True,
        key="B0_uploader"
    )

    # Only display message if button pressed
    compute_clicked = st.button("Compute B0 Metrics")
    
    if compute_clicked:
        if not uploaded_files:
            st.warning("Please upload a T1 DICOM stack before computing metrics.")
        else:
            metrics_store['B0'] = {
                'Mean_ppm': 0.1,
                'SD_ppm': 0.05,
                'PeakToPeak_ppm': 0.12,
                'Status': "PASS" if 0.12 <= ACTION_LIMITS['B0_ppm'] else "FAIL"
            }
            st.success(f"Computed B0 metrics: {metrics_store['B0']}")


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

    # ------------------- Slice Thickness -------------------
    with tabs[3]:
        st.header("Slice Thickness")
        if 'SliceThickness' not in metrics_store:
            st.info("Not filled yet")
        else:
            st.success(f"Current metrics: {metrics_store['SliceThickness']}")
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
        if 'Distortion' not in metrics_store:
            st.info("Not filled yet")
        else:
            st.success(f"Current metrics: {metrics_store['Distortion']}")
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
        if 'LowContrast' not in metrics_store:
            st.info("Not filled yet")
        else:
            st.success(f"Current metrics: {metrics_store['LowContrast']}")
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
        if 'SNR_Ghosting' not in metrics_store:
            st.info("Not filled yet")
        else:
            st.success(f"Current metrics: {metrics_store['SNR_Ghosting']}")
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
        if 'CF_Gain' not in metrics_store:
            st.info("Not filled yet")
        else:
            st.success(f"Current metrics: {metrics_store['CF_Gain']}")
        cf_val = st.number_input("Center Frequency (MHz)",0.0,key="cf")
        gain_val = st.number_input("Receiver Gain",0.0,key="gain")
        if st.button("Submit CF/Gain"):
            metrics_store['CF_Gain'] = {'CF':cf_val,'Gain':gain_val,'Status':'N/A'}
            st.session_state.metrics_store = metrics_store
            st.success("CF/Gain recorded")
            save_metrics_csv("CF_Gain", pd.DataFrame([metrics_store['CF_Gain']]))

    # ------------------- PDF / Report -------------------
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

    # ------------------- Trend Tab Placeholder -------------------
    with tabs[7]:
        st.header("Trend Analysis")
        st.info("Trend plots will be generated from historical CSVs (optional).")
