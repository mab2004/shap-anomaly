import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier # Assuming your anomaly_detector is a RandomForestClassifier
import matplotlib.pyplot as plt
import shap # Needed for dynamic plotting of SHAP waterfall for individual instances
import plotly.express as px # For interactive bar charts
import time # For simulation delays
from datetime import datetime # For timestamps
import os # For checking file existence

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Advanced Network Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide",  # Use wide layout for more space
    initial_sidebar_state="expanded", # Keep sidebar expanded by default
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# This is an advanced Network Intrusion Detection System Dashboard built with Streamlit and explainable AI (SHAP)."
    }
)

# --- Custom CSS for enhanced visual appeal ---
st.markdown("""
<style>
/* Overall app styling */
.stApp {
    background-color: #0d1117; /* Dark GitHub-like background */
    color: #c9d1d9; /* Light text color */
    font-family: 'Segoe UI', sans-serif; /* Modern font */
}

/* Sidebar styling */
.stSidebar {
    background-color: #161b22; /* Slightly lighter dark for sidebar */
    padding-top: 2rem;
}
.stSidebar .stRadio div[role="radiogroup"] label {
    margin-bottom: 0.5rem; /* Space between radio buttons */
}
.stSidebar .stRadio div[role="radiogroup"] label > div {
    padding: 10px 15px; /* Padding for radio button labels */
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
    color: #c9d1d9;
    font-size: 1.1em;
}
.stSidebar .stRadio div[role="radiogroup"] label:hover > div {
    background-color: #222b36; /* Hover effect */
    color: #58a6ff;
}
.stSidebar .stRadio div[role="radiogroup"] label[data-baseweb="radio"] div[aria-selected="true"] > div {
    background-color: #58a6ff; /* Selected radio button */
    color: white;
    font-weight: bold;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #58a6ff; /* Accent color for headings (a shade of blue) */
    font-weight: 600;
}

/* Buttons */
.stButton>button {
    background-color: #238636; /* GitHub green for primary action */
    color: white;
    border-radius: 6px;
    padding: 10px 20px;
    font-size: 1.1em;
    font-weight: bold;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    width: 100%; /* Make buttons fill their container */
}
.stButton>button:hover {
    background-color: #2ea043; /* Lighter green on hover */
    transform: translateY(-2px);
    box_shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
}
/* Ensure button text is always visible */
.stButton>button span {
    color: white !important;
}


/* Tabs styling (for main navigation) */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px; /* Space between tabs */
    margin-bottom: 20px; /* Space below tabs */
}
.stTabs [data-baseweb="tab-list"] button {
    background-color: #161b22; /* Darker background for inactive tabs */
    color: #c9d1d9; /* Light text for inactive tabs */
    border_radius: 8px; /* Rounded corners for tabs */
    padding: 12px 25px;
    font_size: 1.05em;
    font_weight: 500;
    border: none;
    transition: all 0.2s ease-in-out;
    flex-grow: 1; /* Make tabs expand to fill space */
    text_align: center;
}
.stTabs [data-baseweb="tab-list"] button:hover {
    background-color: #222b36; /* Slightly lighter on hover */
    color: #58a6ff; /* Accent color on hover */
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: #58a6ff; /* Accent color for active tab background */
    color: white; /* White text for active tab */
    font_weight: bold;
    box_shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Shadow for active tab */
}
.stTabs [data-baseweb="tab-list"] {
    border-bottom: none; /* Remove default border */
}

/* Expander styling */
.streamlit-expanderContent {
    background-color: #161b22;
    border-radius: 8px;
    padding: 15px;
    border: 1px solid #30363d;
}
.streamlit-expanderHeader {
    background-color: #161b22;
    border-radius: 8px;
    border: 1px solid #30363d;
    padding: 10px;
    color: #58a6ff;
    font-weight: bold;
}

/* File uploader styling */
.stFileUploader {
    background-color: #161b22;
    border: 1px dashed #58a6ff;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.stFileUploader > div > div > button {
    background-color: #58a6ff;
    color: white;
    border-radius: 5px;
}

/* Dataframe styling */
.stDataFrame {
    border: 1px solid #30363d;
    border-radius: 8px;
}
.dataframe th {
    background-color: #161b22 !important;
    color: #58a6ff !important;
    font-weight: bold !important;
}
.dataframe td {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 2.5em;
    color: #238636; /* Green for success metrics */
    font-weight: bold;
}
[data-testid="stMetricLabel"] {
    font_size: 1.1em;
    color: #c9d1d9;
}
[data-testid="stMetricValue"] + [data-testid="stMetricDelta"] { /* Style for the change indicator in metric */
    color: #e94560; /* Red for potential increase in attacks, or green for decrease */
}

/* SHAP plots styling */
.stPlotlyChart {
    border: 1px solid #30363d;
    border-radius: 8px;
    background-color: #161b22;
    padding: 10px; /* Add some padding around plots */
}

/* Info/Warning/Error boxes */
.stAlert {
    border-radius: 8px;
}

/* Markdown containers */
.stMarkdown {
    color: #c9d1d9;
}

/* Custom box styling for System Guide */
.custom-box {
    background-color: #161b22;
    border-left: 5px solid #58a6ff; /* Accent border */
    padding: 15px;
    border_radius: 8px;
    margin-bottom: 15px;
    box_shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.custom-box h4 {
    color: #58a6ff;
    margin_top: 0;
}
</style>
""", unsafe_allow_html=True)


# --- Define KDD Column Names (Crucial for consistent preprocessing) ---
# This list must match the columns in your KDDTrain+.TXT and KDDTest+.TXT files,
# excluding the last (difficulty) column, and including the class label at the end.
KDD_COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "class_label" # The final column is the class label
]
# Categorical features within the KDD dataset
KDD_CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']


# --- Helper Functions (with st.cache_data/st.cache_resource for performance) ---

@st.cache_resource
def load_artifacts():
    """Loads all pre-trained model artifacts."""
    artifacts = {}
    try:
        with open('anomaly_detector.pkl', 'rb') as f:
            artifacts['model'] = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            artifacts['scaler'] = pickle.load(f)
        with open('features.pkl', 'rb') as f:
            artifacts['feature_names'] = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            artifacts['label_encoder'] = pickle.load(f)
        
        # SHAP explainer might not always be present or needed for basic functionality
        if os.path.exists('shap_explainer.pkl'):
            with open('shap_explainer.pkl', 'rb') as f:
                artifacts['shap_explainer'] = pickle.load(f)
        else:
            artifacts['shap_explainer'] = None # Indicate that explainer is not available

        # Assume optimal threshold is also saved, or use a default
        if os.path.exists('optimal_threshold.pkl'):
            with open('optimal_threshold.pkl', 'rb') as f:
                artifacts['optimal_threshold'] = pickle.load(f)
        else:
            artifacts['optimal_threshold'] = 0.5 # Default threshold if not found

        # Load model metrics
        if os.path.exists('model_metrics.pkl'):
            with open('model_metrics.pkl', 'rb') as f:
                artifacts['metrics'] = pickle.load(f)
        else:
            artifacts['metrics'] = None # Indicate that metrics are not available
        
        # Load original_df for Network Traffic Composition (assuming it's KDDTest+.TXT for simplicity)
        # If your overall traffic composition should be based on something else, adjust this.
        # This is needed for the "Network Traffic Composition" bar chart.
        artifacts['original_df'] = load_kdd_dataset('dataset/KDDTest+.TXT') 
        if artifacts['original_df'].empty:
            st.sidebar.warning("Could not load KDDTest+.TXT for network traffic composition analysis.")

        st.sidebar.success("All model artifacts loaded successfully!")
        return artifacts
    except FileNotFoundError as e:
        st.sidebar.error(f"Missing essential artifact file: {e}. Please ensure all .pkl files are in the same directory as this script.")
        st.stop() # Stop execution if essential files are missing
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred loading artifacts: {e}")
        st.stop() # Stop execution for other loading errors

@st.cache_data
def load_kdd_dataset(file_path):
    """Loads and returns the KDD dataset with proper column names."""
    try:
        df = pd.read_csv(file_path, header=None)
        
        # NSL-KDD has 41 features + 1 class label + 1 difficulty score (total 43 columns)
        # KDD99 has 41 features + 1 class label (total 42 columns)
        # Drop the last column if it's the difficulty score.
        if df.shape[1] == 43: # Assuming 43 columns means 41 features + class + difficulty
            df = df.iloc[:, :-1] # Drop the last column (difficulty score)
        
        # Assign standard KDD names
        if df.shape[1] == len(KDD_COL_NAMES):
            df.columns = KDD_COL_NAMES
        else:
            st.error(f"Dataset {file_path} has {df.shape[1]} columns. Expected {len(KDD_COL_NAMES)} after dropping difficulty. Please check your dataset format.")
            return pd.DataFrame()

        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure 'KDDTrain+.TXT' and 'KDDTest+.TXT' are in a 'dataset' folder in the same directory as this script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading or processing dataset {file_path}: {e}")
        return pd.DataFrame()

@st.cache_data
def preprocess_data_for_prediction(df, _scaler_obj, _label_encoder_obj, _expected_features):
    """
    Preprocesses the dataset for prediction using loaded scaler, label encoder, and feature names.
    This function MUST replicate the preprocessing steps from your project.py
    """
    if df.empty:
        return pd.DataFrame(), pd.Series()

    # Make a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Separate features and raw target
    X_raw = df_copy.drop('class_label', axis=1)
    y_raw = df_copy['class_label']

    # --- Crucial: Standardize labels to 'normal' or 'attack' BEFORE encoding ---
    # This aligns with how the label_encoder was trained in project.py (binary classification)
    y_standardized = y_raw.apply(lambda x: 'normal' if x == 'normal' else 'attack')

    # --- Handle categorical features with one-hot encoding ---
    X_processed = pd.get_dummies(X_raw, columns=KDD_CATEGORICAL_FEATURES)

    # --- Align columns to the expected feature set from training ---
    # Add missing columns with 0
    missing_cols = set(_expected_features) - set(X_processed.columns)
    for c in missing_cols:
        X_processed[c] = 0
    
    # Drop columns not in the expected features (if any new dummy vars appeared in the uploaded data)
    extra_cols = set(X_processed.columns) - set(_expected_features)
    if extra_cols:
        X_processed = X_processed.drop(columns=list(extra_cols))
    
    # Reorder columns to match the training data's feature order
    X_processed = X_processed[_expected_features]

    # --- Apply scaling ---
    X_scaled_array = _scaler_obj.transform(X_processed)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=_expected_features, index=X_processed.index)

    # --- Encode target labels using the loaded encoder ---
    y_encoded = pd.Series(index=y_standardized.index, dtype=int) # Should be int now
    for idx, label in y_standardized.items():
        if label in _label_encoder_obj.classes_:
            y_encoded.loc[idx] = _label_encoder_obj.transform([label])[0]
        else:
            # This case should ideally not happen after standardization, but as a fallback
            st.error(f"Post-standardization: Unseen label '{label}' encountered. This indicates a deeper issue.")
            y_encoded.loc[idx] = -1 # Or some other indicator of error


    return X_scaled_df, y_encoded, y_raw # Also return y_raw for display

# --- IP Formatting Helper (Placeholder if needed, as it was commented out) ---
# Assuming these are just placeholders and not actual IP conversions from raw data
def format_ip(value):
    # This function needs actual logic to format IP addresses from the raw data if src_bytes/dst_bytes are not IPs
    # For now, it will just return the value as is, or a placeholder IP
    if pd.isna(value):
        return "N/A"
    # Basic hashing for consistent dummy IPs based on feature values
    # This is a placeholder and not actual IP conversion
    hash_val = hash(value)
    return f"192.168.{(hash_val % 255)}.{(hash_val * 7 % 255)}"


# --- Load Model Artifacts ---
artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
feature_names = artifacts['feature_names']
label_encoder = artifacts['label_encoder']
shap_explainer = artifacts.get('shap_explainer')
optimal_threshold_from_training = artifacts['optimal_threshold']
# Load metrics and original_df from artifacts
model_metrics = artifacts['metrics']
original_df_for_composition = artifacts['original_df']
# Determine 'normal' and 'attack' class mapping from label_encoder
NORMAL_CLASS_LABEL = 'normal' # The string label for normal traffic
ATTACK_CLASS_LABEL_STRING = 'attack' # The string label for attack traffic (after binary conversion in project.py)
NORMAL_ENCODED_VALUE = None
ATTACK_ENCODED_VALUE = 1

try:
    if NORMAL_CLASS_LABEL in label_encoder.classes_ and ATTACK_CLASS_LABEL_STRING in label_encoder.classes_:
        NORMAL_ENCODED_VALUE = label_encoder.transform([NORMAL_CLASS_LABEL])[0]
        ATTACK_ENCODED_VALUE = label_encoder.transform([ATTACK_CLASS_LABEL_STRING])[0]
    else:
        st.error(f"Label encoder does not contain both '{NORMAL_CLASS_LABEL}' and '{ATTACK_CLASS_LABEL_STRING}' classes.")
        st.stop()
except Exception as e:
    st.error(f"Error determining encoded values for normal/attack labels: {e}")
    st.stop()


# --- Sidebar Navigation and Model Information ---
st.sidebar.title("Configuration")

# Redesign Sidebar Model Information Section
st.sidebar.header("🛡️ Model Information")

if model_metrics:
    st.sidebar.markdown(f"**Classifier**: Random Forest Classifier")
    st.sidebar.markdown(f"**Trained on**: KDDCup'99 Dataset")
    
    st.sidebar.subheader("Performance Snapshot:")
    st.sidebar.markdown(f"- **Accuracy**: `{model_metrics['accuracy']:.2f}%`")
    st.sidebar.markdown(f"- **Attack Recall**: `{model_metrics['recall']:.2f}%`")
    fpr_value = model_metrics.get('false_alarm_rate', 'N/A')
    if isinstance(fpr_value, (int, float)):
        st.sidebar.markdown(f"- **False Alarm Rate (FPR)**: `{fpr_value:.2f}%`")
    else:
        st.sidebar.markdown(f"- **False Alarm Rate (FPR)**: `{fpr_value}`")
    st.sidebar.markdown(f"- **F1 Score**: `{model_metrics['f1']:.2f}`")
    st.sidebar.markdown(f"- **Optimal Threshold**: `{model_metrics['optimal_threshold']:.4f}`") # More precision for threshold
    
    st.sidebar.info("These metrics reflect the model's performance on the unseen test set.")
else:
    st.sidebar.info("Model performance metrics not loaded. Please ensure `model_metrics.pkl` is generated.")
st.sidebar.markdown("---")

# --- Simulation Settings in Sidebar ---
st.sidebar.subheader("Simulation Settings")

# Initialize session state for sliders if not already present
if 'prediction_threshold' not in st.session_state:
    st.session_state['prediction_threshold'] = optimal_threshold_from_training
if 'simulation_delay' not in st.session_state:
    st.session_state['simulation_delay'] = 0.5 # Default delay

st.session_state['prediction_threshold'] = st.sidebar.slider(
    "Attack Alert Probability Threshold",
    0.0, 1.0, st.session_state['prediction_threshold'], 0.01,
    key='threshold_slider',
    help=f"Set the minimum probability for a connection to be flagged as an 'Alert'. Optimal from training: {optimal_threshold_from_training:.4f}"
)

st.session_state['simulation_delay'] = st.sidebar.slider(
    "Simulation Delay (seconds per record)",
    0.0, 2.0, st.session_state['simulation_delay'], 0.1,
    key='delay_slider',
    help="Adjust the delay between processing each network record in the live simulation."
)


st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; margin-top: 20px;">
<p style="color: #9CA3AF;">Developed by Muhammad Ali Bukhari</p>
</div>
""", unsafe_allow_html=True)


# --- Main Dashboard Content ---
st.title("🛡️ CyberShield AI")
st.markdown("""
<div style="background: #161b22; padding: 20px; border-radius: 10px; margin-bottom: 30px; border: 1px solid #30363d;">
<h3 style="color: #58a6ff; margin-bottom: 10px;">Network Anomaly Detection System</h3>
<p style="color: #c9d1d9;"> Real-time cybersecurity monitoring powered by Explainable AI. Detect and interpret network threats with transparent machine learning. </p>
</div>
""", unsafe_allow_html=True)

# Main navigation tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard Overview",
    "🚨 Threat Analysis",
    "📈 Model Performance",
    "📖 System Guide"
])

with tab1:
    # Redesign System/Dashboard Overview
    st.header("📊 System Overview & Real-time Metrics")

    # Get metrics from loaded artifacts
    if model_metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="Overall Accuracy", value=f"{model_metrics['accuracy']:.2f}%")
            st.progress(int(model_metrics['accuracy']), text="Model Accuracy")

        with col2:
            st.metric(label="Attack Recall (Detection Rate)", value=f"{model_metrics['recall']:.2f}%")
            st.progress(int(model_metrics['recall']), text="Effectiveness at finding attacks")

        with col3:
            false_alarm_rate = model_metrics.get('false_alarm_rate', 0.0) 
            st.metric(label="False Alarm Rate (FPR)", value=f"{false_alarm_rate:.2f}%")
            # Invert progress for FPR as lower is better
            st.progress(int(100 - false_alarm_rate), text="Lower False Positives = Better") 

        with col4:
            st.metric(label="F1 Score", value=f"{model_metrics['f1']:.2f}")
            st.progress(int(model_metrics['f1']), text="Balance of Precision and Recall") # F1 is 0-1, convert to 0-100%

        st.markdown("---") # Separator
    else:
        st.info("Model performance metrics not available for Dashboard Overview. Please ensure `model_metrics.pkl` is generated and loaded.")

    # Recent Activity Snapshot
    st.subheader("Recent Activity Snapshot")
    st.write("A quick look at the latest network connections and their classification by the system.")

    # Initialize st.session_state.processed_attacks if it doesn't exist
    if 'processed_attacks' not in st.session_state:
        st.session_state.processed_attacks = []

    # Display the table, reversed to show most recent at the top
    # This table will now be populated by the Threat Analysis section's simulation loop
    if st.session_state.processed_attacks:
        recent_activity_df = pd.DataFrame(st.session_state.processed_attacks)
        st.dataframe(recent_activity_df.iloc[::-1], use_container_width=True) # Reverse to show most recent first
    else:
        st.info("No recent activity to display. Run a simulation in the 'Threat Analysis' tab.")

    # --- Network Traffic Composition Chart (Bar Chart) ---
    st.markdown("---")
    st.subheader("🌐 Network Traffic Composition")
    st.write("Distribution of network connections by their classified type from the loaded dataset.")

    if original_df_for_composition is not None and not original_df_for_composition.empty and label_encoder is not None:
        if 'class_label' in original_df_for_composition.columns:
            # Get the value counts of the 'class_label' column (which is still raw/string here)
            # Ensure consistency with how project.py processes labels if 'class_label' is not raw
            # For this chart, we want the *original* raw labels (e.g., 'normal', 'smurf', 'neptune')
            # So, it's crucial that original_df_for_composition has the raw labels.
            # If your original_df_for_composition has numerical labels, you need to inverse_transform them here.
            
            # Assuming 'class_label' column in original_df_for_composition contains string labels
            traffic_counts = original_df_for_composition['class_label'].value_counts().reset_index()
            traffic_counts.columns = ['Attack Type', 'Count']
            
            # To simplify for the bar chart, group all attack types into a single 'Attack' category
            # except 'normal'
            traffic_counts['Category'] = traffic_counts['Attack Type'].apply(
                lambda x: 'Normal' if x == 'normal' else 'Attack'
            )
            # Aggregate counts for 'Attack' category
            aggregated_traffic_counts = traffic_counts.groupby('Category')['Count'].sum().reset_index()

            # Create a bar chart using Plotly Express
            fig_bar = px.bar(
                aggregated_traffic_counts,
                x='Category',
                y='Count',
                title='Overall Network Traffic Composition (Aggregated)',
                labels={'Category': 'Traffic Category', 'Count': 'Number of Samples'},
                color='Category', # Color bars by category
                color_discrete_map={'Normal': '#238636', 'Attack': '#e94560'}, # Custom colors
                template="plotly_dark" # Use a dark theme for Plotly
            )
            
            fig_bar.update_layout(xaxis_title="Traffic Type", yaxis_title="Number of Samples",
                                  xaxis={'categoryorder':'total descending'}) # Order bars by count
            
            st.plotly_chart(fig_bar, use_container_width=True)

            with st.expander("Detailed Attack Type Breakdown"):
                # Bar chart for detailed attack types
                fig_detailed_bar = px.bar(
                    traffic_counts[traffic_counts['Attack Type'] != 'normal'], # Exclude 'normal' for detailed view
                    x='Attack Type',
                    y='Count',
                    title='Detailed Breakdown of Attack Types',
                    labels={'Attack Type': 'Specific Attack Type', 'Count': 'Number of Samples'},
                    color='Attack Type', # Color bars by attack type
                    template="plotly_dark"
                )
                fig_detailed_bar.update_layout(xaxis_title="Attack Type", yaxis_title="Number of Samples",
                                                xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_detailed_bar, use_container_width=True)


        else:
            st.warning("Loaded dataset for composition does not contain a 'class_label' column.")
    else:
        st.info("Original dataset or Label Encoder not loaded. Cannot display Network Traffic Composition.")

with tab2:
    # Threat Analysis
    st.header("🚨 Threat Analysis")
    st.markdown("Run a simulation on a sample of network data or upload your own to analyze potential threats.")

    analysis_mode = st.radio(
        "Select Analysis Mode:",
        ("⬆️ Upload New Data", "🔄 Live Simulation (from KDDTest+.TXT)"),
        key="analysis_mode_radio"
    )

    df_for_analysis = pd.DataFrame() # Initialize empty DataFrame

    if analysis_mode == "⬆️ Upload New Data":
        st.subheader("Upload Network Data for Analysis")
        st.write("Upload a .txt or .csv file containing network connection records (KDD format: 41 features + 1 class_label column).")
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv"], key="file_uploader")

        if uploaded_file is not None:
            try:
                df_for_analysis = pd.read_csv(uploaded_file, header=None)
                if df_for_analysis.shape[1] == 42 or df_for_analysis.shape[1] == 43:
                    if df_for_analysis.shape[1] == 43:
                        df_for_analysis = df_for_analysis.iloc[:, :-1] # Drop difficulty score
                    df_for_analysis.columns = KDD_COL_NAMES # Assign standard KDD names
                    st.success("File uploaded successfully! Click 'Analyze Data' below.")
                    st.dataframe(df_for_analysis.head(), use_container_width=True)
                else:
                    st.error(f"Uploaded file has {df_for_analysis.shape[1]} columns. Expected 42 or 43 columns (KDD format). Please check your file.")
                    df_for_analysis = pd.DataFrame() # Reset to empty if invalid
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}. Please ensure it's a valid KDD-formatted .txt or .csv.")
                df_for_analysis = pd.DataFrame() # Reset to empty on error

            if not df_for_analysis.empty:
                if st.button("Analyze Uploaded Data", key="analyze_uploaded_button"):
                    st.session_state['trigger_analysis'] = True
                    st.session_state['analysis_data'] = df_for_analysis
                else:
                    st.session_state['trigger_analysis'] = False # If button not clicked
            else:
                st.session_state['trigger_analysis'] = False # No valid df
        else:
            st.session_state['trigger_analysis'] = False

    elif analysis_mode == "🔄 Live Simulation (from KDDTest+.TXT)":
        st.subheader("Run Live Simulation")
        st.write("Click the button below to run a simulation on a randomly selected batch of unseen data from the `KDDTest+.TXT` dataset.")
        
        if st.button("Start Live Simulation", key="start_simulation_button"):
            st.session_state['trigger_analysis'] = True
            test_df_full = load_kdd_dataset('dataset/KDDTest+.TXT')
            if not test_df_full.empty:
                sample_size = min(10, len(test_df_full)) # Simulate a larger batch for analysis
                df_for_analysis = test_df_full.sample(n=sample_size, random_state=int(time.time())) # Different sample each time
                st.session_state['analysis_data'] = df_for_analysis
            else:
                st.session_state['trigger_analysis'] = False
                st.warning("KDDTest+.TXT could not be loaded for simulation.")
        else:
            st.session_state['trigger_analysis'] = False

    # Perform analysis if triggered
    if st.session_state.get('trigger_analysis', False) and 'analysis_data' in st.session_state and not st.session_state['analysis_data'].empty:
        df_to_analyze = st.session_state['analysis_data']
        st.info(f"Processing {len(df_to_analyze)} records for analysis...")

        # Place a placeholder for live updates
        status_text = st.empty()
        progress_bar = st.progress(0)

        # Initialize results list
        analysis_results_list = []
       # Ensure st.session_state.processed_attacks is always a list
        if 'processed_attacks' not in st.session_state:
            st.session_state.processed_attacks = []
            
        # Simulate processing record by record for the progress bar
        for i, (original_idx, row) in enumerate(df_to_analyze.iterrows()):
            single_record_df = pd.DataFrame([row], columns=KDD_COL_NAMES)
            X_processed_single, y_true_encoded_single, y_true_raw_single = preprocess_data_for_prediction(
                single_record_df.copy(), scaler, label_encoder, feature_names
            )

            if not X_processed_single.empty:
                prediction = model.predict(X_processed_single)[0]
                prediction_proba = model.predict_proba(X_processed_single)[:, ATTACK_ENCODED_VALUE][0]

                is_alert = prediction_proba >= st.session_state['prediction_threshold']
                
                # Decode true and predicted labels for display
                true_label_display = y_true_raw_single.iloc[0] # Original raw label
                # Determine predicted_label_display based on threshold for consistency with alert
                predicted_label_display_string = ATTACK_CLASS_LABEL_STRING if is_alert else NORMAL_CLASS_LABEL

                analysis_results_list.append({
                    'Original_Index': original_idx,
                    'Timestamp': datetime.now().strftime("%H:%M:%S"),
                    'Source IP': format_ip(row['src_bytes']), # Using dummy format_ip
                    'Destination IP': format_ip(row['dst_bytes']), # Using dummy format_ip
                    'Protocol': row['protocol_type'],
                    'True_Class': true_label_display, # Display original label
                    'Predicted_Class': predicted_label_display_string,
                    'Attack_Probability': prediction_proba,
                    'Alert': is_alert
                })

                # Add to session state for Recent Activity Snapshot
                st.session_state.processed_attacks.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Original Data Index": original_idx,
                    "True Class": true_label_display.capitalize(),
                    "Predicted Class": predicted_label_display_string.capitalize(),
                    "Attack Probability": f"{prediction_proba:.4f}",
                    "Alert": "✅" if is_alert else "❌"
                })

                # Keep only the last 20 samples for the snapshot
                if len(st.session_state.processed_attacks) > 20:
                    st.session_state.processed_attacks.pop(0)
            progress_bar.progress((i + 1) / len(df_to_analyze))
            status_text.text(f"Processing record {i+1} of {len(df_to_analyze)}...")
            time.sleep(st.session_state['simulation_delay']) # Apply simulation delay

        status_text.text("Analysis completed!")
        progress_bar.empty()

        results_df = pd.DataFrame(analysis_results_list)

        # Filter for high-confidence attacks based on threshold
        high_confidence_attacks = results_df[
            (results_df['Predicted_Class'] == ATTACK_CLASS_LABEL_STRING) &
            (results_df['Alert'] == True)
        ]

        if not high_confidence_attacks.empty:
            st.subheader(f"🚨 Detected High-Confidence Attacks ({len(high_confidence_attacks)} instances)")
            st.write("Below are the details of connections classified as 'Attack' along with their individual SHAP explanations.")
            
            for i, row_data in high_confidence_attacks.iterrows():
                original_df_idx = row_data['Original_Index']
                st.markdown(f"**--- Detected Attack Instance {i+1} (Original Data Index: {original_df_idx}) ---**")
                
                instance_raw_data_df = df_to_analyze.loc[[original_df_idx]]
                X_processed_instance, _, _ = preprocess_data_for_prediction(
                    instance_raw_data_df.copy(), scaler, label_encoder, feature_names
                )

                st.write(f"**Prediction:** **{row_data['Predicted_Class'].upper()}** (Probability of Attack: {row_data['Attack_Probability']:.4f})")
                st.write(f"**True Class (Original):** {row_data['True_Class']}") # Display original label

                with st.expander("View Raw Data for this Instance"):
                    st.dataframe(instance_raw_data_df, use_container_width=True)
                
                # --- Dynamic SHAP Explanation for the instance ---
                if shap_explainer is not None and not X_processed_instance.empty:
                    
                    # Flag to control plot generation for this instance
                    skip_waterfall_plot = False 

                    try:
                        shap_values_instance = shap_explainer.shap_values(X_processed_instance)
                        
                        # Determine SHAP values for the ATTACK_ENCODED_VALUE class
                        shap_values_for_target_class = None
                        
                        if isinstance(shap_values_instance, list):
                            # For multi-output models, shap_values_instance is a list of arrays (one per class)
                            # Each array has shape (num_samples, num_features)
                            # We expect shap_values_instance[ATTACK_ENCODED_VALUE] to be (1, num_features)
                            shap_values_for_target_class = shap_values_instance[ATTACK_ENCODED_VALUE]
                        else:
                            # Handle cases where shap_values_instance might directly be an array
                            if (isinstance(shap_values_instance, np.ndarray) and
                                shap_values_instance.ndim == 2 and
                                shap_values_instance.shape[0] == len(feature_names) and
                                shap_values_instance.shape[1] == 2): # This matches (num_features, num_classes)
                                
                                shap_values_for_target_class = shap_values_instance[:, ATTACK_ENCODED_VALUE].reshape(1, -1) # Reshape to (1, num_features)

                            elif (isinstance(shap_values_instance, np.ndarray) and
                                    shap_values_instance.ndim == 3 and
                                    shap_values_instance.shape[0] == 1 and
                                    shap_values_instance.shape[2] == 2): # (1, num_features, num_classes)
                                shap_values_for_target_class = shap_values_instance[0, :, ATTACK_ENCODED_VALUE].reshape(1, -1)
                            
                            elif (isinstance(shap_values_instance, np.ndarray) and
                                    shap_values_instance.ndim == 2 and
                                    shap_values_instance.shape[0] == 1 and
                                    shap_values_instance.shape[1] == len(feature_names)): # (1, num_features) for a single-output model
                                shap_values_for_target_class = shap_values_instance
                            else:
                                st.warning(f"SHAP explainer output format is unexpected (not a list and not a recognized array shape). Current shape: {getattr(shap_values_instance, 'shape', 'N/A')}. Cannot generate waterfall plot.")
                                skip_waterfall_plot = True # Set flag to skip plotting
                                

                        # Ensure shap_values_for_target_class is (1, num_features) for a single instance
                        if not skip_waterfall_plot and (shap_values_for_target_class is None or shap_values_for_target_class.shape[0] != 1):
                            st.warning(f"SHAP values extraction resulted in an unexpected shape ({getattr(shap_values_for_target_class, 'shape', 'N/A')}) after initial processing for a single instance. Expected (1, {len(feature_names)}).")
                            skip_waterfall_plot = True # Set flag to skip plotting


                        # Determine Base value for the ATTACK_ENCODED_VALUE class and ensure it's a scalar
                        base_value_raw = None
                        if not skip_waterfall_plot: # Only proceed if not already skipped
                            if isinstance(shap_explainer.expected_value, list):
                                # If it's a list (e.g., from KernelExplainer for multi-output models)
                                # Each element in the list is the base value for a specific class
                                if len(shap_explainer.expected_value) > ATTACK_ENCODED_VALUE:
                                    base_value_raw = shap_explainer.expected_value[ATTACK_ENCODED_VALUE]
                                else:
                                    st.warning(f"ATTACK_ENCODED_VALUE ({ATTACK_ENCODED_VALUE}) is out of bounds for shap_explainer.expected_value list of size {len(shap_explainer.expected_value)}.")
                                    skip_waterfall_plot = True
                            elif isinstance(shap_explainer.expected_value, np.ndarray):
                                # If it's a NumPy array (common for TreeExplainer on binary classification)
                                # It should be a 1D array where index corresponds to class
                                if shap_explainer.expected_value.ndim == 1 and shap_explainer.expected_value.size > ATTACK_ENCODED_VALUE:
                                    base_value_raw = shap_explainer.expected_value[ATTACK_ENCODED_VALUE]
                                elif shap_explainer.expected_value.size == 1: # Handle single output model or single expected value
                                    base_value_raw = shap_explainer.expected_value.item() # Convert 1-element array to scalar
                                else:
                                    st.warning(f"Unexpected shape or size for shap_explainer.expected_value NumPy array: {shap_explainer.expected_value.shape}. Cannot determine base value.")
                                    skip_waterfall_plot = True
                            else: # Fallback for scalar or other unexpected types directly
                                base_value_raw = shap_explainer.expected_value # Assume it's already a scalar or compatible

                        # Crucial: Ensure base_value_for_target_class is a scalar after extraction
                        if not skip_waterfall_plot:
                            if isinstance(base_value_raw, np.ndarray) and base_value_raw.size == 1:
                                base_value_for_target_class = base_value_raw.item() # Convert to scalar
                            elif isinstance(base_value_raw, (float, int)):
                                base_value_for_target_class = base_value_raw
                            else:
                                st.warning(f"Final conversion: Unexpected type or shape for SHAP base value: {type(base_value_raw)}, size: {getattr(base_value_raw, 'size', 'N/A')}. Cannot generate waterfall plot.")
                                skip_waterfall_plot = True

                        # Only generate the plot if no issues were encountered
                        if not skip_waterfall_plot:
                            # Prepare the 1D array for values and Series for data for the waterfall plot
                            shap_values_to_plot = shap_values_for_target_class[0] 
                            data_to_plot = X_processed_instance.iloc[0]

                            # Create the Explanation object
                            explanation_obj = shap.Explanation(values=shap_values_to_plot,
                                                            base_values=base_value_for_target_class,
                                                            data=data_to_plot,
                                                            feature_names=feature_names)
                            
                            # Correct SHAP plot function call: shap.plots.waterfall
                            plt.figure(figsize=(10, 6)) # Create a new figure to ensure proper rendering in Streamlit
                            shap.plots.waterfall(explanation_obj, show=False, max_display=15)
                            st.pyplot(plt.gcf()) # Get current figure and display
                            plt.clf() # Clear the plot to prevent overlapping plots on subsequent runs
                            st.success("Dynamic SHAP Waterfall Plot Generated Successfully!")

                    except IndexError:
                        st.warning("SHAP explanation could not be processed for this instance's waterfall plot due to incorrect indexing (IndexError). This might indicate the `ATTACK_ENCODED_VALUE` is out of bounds for `shap_values_instance` or `expected_value`.")
                    except Exception as shap_e:
                        st.warning(f"Could not generate dynamic SHAP explanation for this instance: {shap_e}. Please ensure your `shap_explainer.pkl` is correctly loaded and compatible with the data.")
                else:
                    st.info("`shap_explainer.pkl` not loaded or data empty. Cannot generate dynamic SHAP waterfall plots for individual instances.")
                st.markdown("---") # Separator for instances

        # Display the full results table
        display_cols_full = ['Timestamp', 'Source IP', 'Destination IP', 'Protocol', 'True_Class', 'Predicted_Class', 'Attack_Probability', 'Alert']
        st.dataframe(results_df[display_cols_full].style.apply(
            lambda x: ['background-color: #23863620' if x['Alert'] else '' for _ in x],
            axis=1
        ), use_container_width=True)


with tab3:
    # Model Performance
    st.header("📈 Model Performance")
    st.markdown("Detailed insights into the model's performance metrics and explainability.")

    st.subheader("Performance Curves")
    st.write("Visualizations demonstrating the model's detection capabilities.")
    # Check if performance_curves.png exists before displaying
    if os.path.exists("performance_curves.png"):
        st.image("performance_curves.png", caption="ROC Curve and Precision-Recall Curve", use_container_width=True)
    else:
        st.error("`performance_curves.png` not found. Please ensure the image file is in the same directory as this script.")

    with st.expander("Explanation of Performance Curves"):
        st.markdown("""
        #### Receiver Operating Characteristic (ROC) Curve
        The ROC curve illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. A model with perfect classification would have an ROC curve that passes through the top-left corner (100% TPR, 0% FPR), and the Area Under the Curve (AUC) would be 1.0. A value closer to 1.0 indicates a better model.

        #### Precision-Recall (PR) Curve
        The PR curve plots Precision (Positive Predictive Value) against Recall (True Positive Rate) at various threshold settings. This curve is particularly informative for imbalanced datasets, where the number of negative instances greatly outweighs the positive instances (e.g., normal traffic vs. attacks). A high Area Under the Precision-Recall Curve (AP) indicates a good model.
        """)

    st.markdown("---")
    st.subheader("Global Feature Importance (SHAP Insights)")
    st.write("These visualizations reveal which features are most influential for the model's overall predictions across the entire dataset, providing global interpretability.")
    
    col_shap_bar, col_shap_beeswarm = st.columns(2)
    with col_shap_bar:
        # Check if shap_feature_importance.png exists before displaying
        if os.path.exists("shap_feature_importance.png"):
            st.image("shap_feature_importance.png", caption="Top 20 Features Importance (SHAP Bar Plot)", use_container_width=True)
            with st.expander("Explanation of SHAP Bar Plot"):
                st.markdown("""
                **SHAP Bar Plot Explanation:** This plot displays the average absolute SHAP value for each feature. It provides a global ranking of feature importance, showing which features have the greatest overall impact on the model's output magnitude, regardless of the direction of impact. For instance, a long bar for 'duration' means 'duration' is generally important for predictions.
                """)
        else:
            st.error("`shap_feature_importance.png` not found. Please ensure the image file is in the same directory as this script.")
    with col_shap_beeswarm:
        # Check if shap_detailed_impact.png exists before displaying
        if os.path.exists("shap_detailed_impact.png"):
            st.image("shap_detailed_impact.png", caption="Feature Impact on Attack Predictions (SHAP Beeswarm Plot)", use_container_width=True)
            with st.expander("Explanation of SHAP Beeswarm Plot"):
                st.markdown("""
                **SHAP Beeswarm Plot Explanation:** This plot visualizes the distribution of SHAP values for each feature across the dataset. Each dot represents a single instance's SHAP value for that feature.
                * **Color (Red to Blue):** Represents the feature value for that instance (red for high, blue for low).
                * **Position on X-axis:** Indicates the impact of that feature on the model's output (higher SHAP value means higher impact towards attack prediction, lower towards normal).
                This plot helps understand not just feature importance, but also the direction of their impact and how it varies with feature values.
                """)
        else:
            st.error("`shap_detailed_impact.png` not found. Please ensure the image file is in the same directory as this script.")

    st.markdown("---")
    st.subheader("Individual Prediction Explanation (SHAP Waterfall Plot Example)")
    st.write("An example of how SHAP can explain a single network connection's classification.")
    # Check if shap_waterfall_example.png exists before displaying
    if os.path.exists("shap_waterfall_example.png"):
        st.image("shap_waterfall_example.png", caption="Example SHAP Waterfall Plot for a Detected Attack", use_container_width=True)
        with st.expander("Explanation of SHAP Waterfall Plot"):
            st.markdown("""
            **SHAP Waterfall Plot Explanation:** This plot explains a single prediction by showing how each feature contributes to pushing the model's output from the `base_value` (average prediction) to the final `f(x)` (current instance's prediction).
            * **Base Value (E[f(x)]):** The average model output (e.g., average log-odds of attack) over the training dataset.
            * **Each Bar:** Represents the SHAP value for a specific feature, showing its contribution. Positive values (red) increase the prediction towards the positive class (attack), while negative values (blue) decrease it (towards normal).
            * **f(x):** The final output of the model for the specific instance being explained.
            This plot provides a granular, interpretable view of why a particular connection was classified as an attack.
            """)
    else:
        st.error("`shap_waterfall_example.png` not found. Please ensure the image file is in the same directory as this script.")


with tab4:
    # System Guide
    st.header("📖 System Guide")
    st.markdown("This section provides a comprehensive overview of the CyberShield AI system's architecture, its components, and how they interact to provide robust network intrusion detection.")

    st.subheader("System Architecture Overview")
    st.write("The diagram below illustrates the end-to-end pipeline of the CyberShield AI system.")
    # Check if system_architecture.png exists before displaying
    if os.path.exists("system_architecture.png"):
        st.image("system_architecture.png", caption="CyberShield AI System Architecture Diagram", use_container_width=True)
    else:
        st.error("`system_architecture.png` not found. Please ensure the image file is in the same directory as this script.")

    # Collapsible Data Pipeline Section
    with st.expander("Data Pipeline Details"):
        st.markdown("""
        ### Data Pipeline: From Raw Traffic to Clean Features
        The system's journey begins with the meticulous preparation of network traffic data.
        * **Data Ingestion**: Raw network traffic records, primarily from the NSL-KDD dataset, are ingested. This forms the foundational input for the entire anomaly detection process.
        * **Preprocessing**: This critical phase transforms raw data into a machine-learning-ready format. It involves:
            * **Data Cleaning**: Removing irrelevant columns (e.g., `difficulty_level`).
            * **Categorical Encoding**: Converting nominal features like `protocol_type`, `service`, and `flag` into numerical representations using **One-Hot Encoding**. This expands the feature space with binary columns for each category.
            * **Label Standardization**: Consolidating all various attack labels into a single 'attack' class, making it a **binary classification problem** (Normal vs. Attack) using **`label_encoder.pkl`**.
            * **Feature Scaling**: Normalizing numerical features to a consistent range (0 to 1) using **`scaler.pkl` (MinMaxScaler)**. This prevents features with larger magnitudes from disproportionately influencing the model.
        """)

    # Collapsible Class Imbalance Handling Section
    with st.expander("Class Imbalance Handling Details"):
        st.markdown("""
        ### Class Imbalance Handling: Balancing the Scales
        Network anomaly datasets are inherently imbalanced, with 'normal' traffic vastly outnumbering 'attack' instances. To prevent the model from becoming biased towards the majority class:
        * **ADASYN (Adaptive Synthetic Sampling)**: This advanced oversampling technique is applied to the training data. Unlike simpler methods, ADASYN adaptively generates synthetic samples for the minority 'attack' class, focusing more on samples that are harder to learn (i.e., those closer to the decision boundary). This ensures the model learns robust patterns from the underrepresented attack types.
        """)

    # Collapsible Model Training & Evaluation Section
    with st.expander("Model Training & Evaluation Details"):
        st.markdown("""
        ### Model Training & Evaluation: Learning and Assessing Performance
        This phase involves building and validating the core detection engine.
        * **Model Selection**: A **RandomForestClassifier** (`anomaly_detector.pkl`) is chosen as the primary machine learning model. Its ensemble nature (a "forest" of decision trees) offers robustness, high accuracy, and resistance to overfitting, making it ideal for complex network patterns.
        * **Training**: The Random Forest model is trained on the balanced and preprocessed dataset. During this process, it learns to identify intricate correlations and patterns that differentiate normal behavior from malicious activities.
        * **Prediction Engine**: Once trained, the model serves as a high-speed prediction engine. It takes new, unseen network connections as input, preprocesses them using the same `scaler.pkl` and `label_encoder.pkl`, and rapidly classifies them as either 'Normal' or 'Attack'.
        * **Evaluation**: The model's performance is rigorously assessed on a separate, unseen test set. Key metrics calculated include **Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR)**, and **Average Precision (PR-AUC)**. Visualizations like the **ROC Curve** and **Precision-Recall Curve** (`performance_curves.png`) provide comprehensive insights into the model's effectiveness across various probability thresholds. An **optimal threshold** for classification is also determined to maximize the F1-score, balancing precision and recall.
        """)

    # Collapsible Explainability Layer Section
    with st.expander("Explainability Layer (SHAP) Details"):
        st.markdown("""
        ### Explainability Layer (SHAP): Unveiling Model Insights
        A crucial component of this system is its ability to explain *why* a prediction was made, fostering trust and enabling informed decision-making.
        * **SHAP Integration**: **SHAP (SHapley Additive exPlanations)** is integrated into the system using **`shap_explainer.pkl`**. SHAP provides a powerful framework for explaining the output of any machine learning model by computing "Shapley values," which represent the contribution of each feature to a prediction.
        * **Local Explanations**: For any individual network connection, SHAP generates a **Waterfall Plot** (`shap_waterfall_example.png`). This plot clearly illustrates how each specific feature (e.g., `src_bytes`, `duration`, `protocol_type_tcp`) pushes the prediction towards 'Normal' or 'Attack', relative to the average prediction.
        * **Global Explanations**: SHAP also provides overarching insights into feature importance across the entire dataset. This includes:
            * **Bar Plots** (`shap_feature_importance.png`): Summarizing the average impact of each feature.
            * **Beeswarm Plots** (`shap_detailed_impact.png`): Showing the distribution of SHAP values for each feature and how different feature values influence the prediction.
        """)

    # Collapsible Deployment & User Interface Section
    with st.expander("Deployment & User Interface Details"):
        st.markdown("""
        ### Deployment & User Interface: Empowering Analysts
        The system is deployed as an interactive web application, making it accessible and user-friendly.
        * **Streamlit Web App**: The entire system is encapsulated within a dynamic Streamlit application, allowing users to interact with the anomaly detection and explanation functionalities through a modern web interface.
        * **Visualizations**: The dashboard presents complex data, model insights, and performance metrics through various charts and tables, making complex information easily digestible for security analysts and non-experts alike.
        * **Interactive Simulation**: Users can upload their own network log files (`.csv`) or run simulations on a randomly sampled subset of the test data. This provides immediate predictions and detailed SHAP explanations, showcasing the system's capabilities with new, unseen traffic.
        * **Real-time Insights**: The dashboard is designed to provide near real-time anomaly detection, enabling network analysts to respond rapidly to emerging threats and take proactive measures based on explained alerts.
        """)

    # Collapsible Artifact Saving & Loading Section
    with st.expander("Artifact Saving & Loading Details"):
        st.markdown("""
        ### Artifact Saving & Loading: Ensuring Persistence and Reusability
        To ensure the system's reusability and efficiency, critical components are persistently stored.
        * **Pickle Serialization**: The trained machine learning model (`anomaly_detector.pkl`), preprocessing tools (`scaler.pkl`, `label_encoder.pkl`, `features.pkl`), and the SHAP explainer (`shap_explainer.pkl`) are serialized using Python's `pickle` module. This converts Python objects into a byte stream that can be saved to disk.
        * **Efficient Deployment**: These serialized artifacts can then be efficiently loaded into the Streamlit dashboard or any other environment, eliminating the need to retrain the model or re-fit the preprocessing steps every time the application is run. This ensures consistency and significantly speeds up deployment.
        """)

    # Collapsible Feedback Loop Section
    with st.expander("Feedback Loop Details"):
        st.markdown("""
        ### Feedback Loop: Continuous Improvement
        The CyberShield AI system incorporates a mechanism for continuous improvement, vital for adapting to evolving cyber threats.
        * **Analyst-driven Refinement**: Insights gained from analyzing detected anomalies and understanding their explanations (via SHAP) can be fed back into the model development process. This allows cybersecurity analysts to validate predictions and provide crucial feedback on false positives or missed detections.
        * **Iterative Retraining**: This feedback enables regular retraining and refinement of the model with new, labeled data. This iterative process ensures the system adapts to emerging attack vectors, maintains high detection accuracy over time, and remains robust against sophisticated new threats.
        """)

    st.markdown("---")
    st.subheader("Quick Guide: In a Nutshell")

    col_nutshell1, col_nutshell2 = st.columns(2)

    with col_nutshell1:
        st.markdown(
            """
            <div class="custom-box">
                <h4>User Guide</h4>
                <p>
                This dashboard allows you to:
                <ul>
                    <li>View system performance metrics at a glance.</li>
                    <li>Run live simulations or upload your own network data for threat analysis.</li>
                    <li>Explore detailed model performance metrics and interpret why attacks are detected using SHAP.</li>
                    <li>Understand the system's architecture and components.</li>
                </ul>
                Use the sidebar for navigation and simulation settings.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_nutshell2:
        st.markdown(
            """
            <div class="custom-box">
                <h4>Technical Details</h4>
                <p>
                The system employs a **Random Forest Classifier** trained on the **NSL-KDD dataset**.
                <ul>
                    <li>**Preprocessing:** MinMaxScaler for numerical features, One-Hot Encoding for categorical, and LabelEncoder for binary classification.</li>
                    <li>**Imbalance Handling:** ADASYN oversampling.</li>
                    <li>**Explainability:** SHAP values provide local and global interpretability.</li>
                    <li>**Deployment:** Streamlit for interactive web interface.</li>
                    <li>**Persistence:** Model and preprocessing artifacts serialized with `pickle`.</li>
                </ul>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
# Footer
st.markdown("""
<div class="footer">
    <p>CyberShield AI | Network Anomaly Detection System | © 2025</p>
</div>
""", unsafe_allow_html=True)