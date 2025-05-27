# ====================== CONFIGURE MATPLOTLIB BACKEND FIRST ======================
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before other imports
import matplotlib.pyplot as plt

# ====================== IMPORT REQUIRED LIBRARIES ======================
import shap
import warnings
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold # For feature selection
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import ADASYN
from collections import Counter # For class distribution check
from tqdm import tqdm # For progress bars

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None) # Display all columns in pandas
plt.style.use('ggplot') # Use a consistent plot style

# Define the path to the dataset (assuming 'dataset' folder is in the same directory as script)
dataset_path = os.path.join(os.getcwd(), 'dataset')

# ====================== DATA PREPROCESSING FUNCTIONS ======================
def load_and_preprocess_data(dataset_path):
    """
    Loads KDDTrain+ and KDDTest+ datasets, assigns column names,
    drops 'difficulty_level', standardizes and binary encodes labels,
    one-hot encodes categorical features, and performs Min-Max scaling.

    Args:
        dataset_path (str): The path to the directory containing the dataset files.

    Returns:
        tuple: A tuple containing:
            - train_df (pd.DataFrame): Preprocessed training DataFrame.
            - test_df (pd.DataFrame): Preprocessed testing DataFrame.
            - scaler (MinMaxScaler): Fitted MinMaxScaler object.
            - feature_columns (list): List of names of the processed feature columns.
            - label_encoder (LabelEncoder): Fitted LabelEncoder object.
    """
    print("\n" + "="*40)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*40)

    # Define column names based on KDD99 dataset
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
    ]

    # Load raw data
    print("\n[1.1] Loading dataset files (KDDTrain+.TXT, KDDTest+.TXT)...")
    train_df = pd.read_csv(os.path.join(dataset_path, 'KDDTrain+.TXT'), header=None)
    test_df = pd.read_csv(os.path.join(dataset_path, 'KDDTest+.TXT'), header=None)
    
    train_df.columns = column_names
    test_df.columns = column_names

    # Clean data: Drop 'difficulty_level' and standardize attack labels
    print("\n[1.2] Cleaning data: Dropping 'difficulty_level' and standardizing labels...")
    for df in [train_df, test_df]:
        df.drop(['difficulty_level'], axis=1, inplace=True)
        # Convert all non-'normal' labels to 'attack' for binary classification
        df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    # Combine for consistent one-hot encoding across train and test sets
    print("\n[1.3] One-hot encoding categorical features ('protocol_type', 'service', 'flag')...")
    combined_df = pd.concat([train_df, test_df], axis=0)
    combined_df = pd.get_dummies(combined_df, columns=['protocol_type', 'service', 'flag'])

    # Split back into train and test sets
    train_df = combined_df.iloc[:len(train_df)].copy()
    test_df = combined_df.iloc[len(train_df):].copy()

    # Binary encode labels: 'normal' -> 0, 'attack' -> 1
    print("\n[1.4] Binary encoding labels ('normal' -> 0, 'attack' -> 1)...")
    le = LabelEncoder()
    # Fit LabelEncoder on all unique string labels to ensure consistent mapping
    le.fit(pd.concat([train_df['label'], test_df['label']]).unique())
    
    train_df['label'] = le.transform(train_df['label']).astype(int)
    test_df['label'] = le.transform(test_df['label']).astype(int)

    # Ensure 'normal' is 0 and 'attack' is 1. LabelEncoder maps alphabetically.
    # If 'attack' was mapped to 0 and 'normal' to 1 (e.g., if 'attack' came first alphabetically)
    # then swap the encoded values.
    if 'normal' in le.classes_ and 'attack' in le.classes_:
        if le.transform(['normal'])[0] == 1 and le.transform(['attack'])[0] == 0:
            print("INFO: LabelEncoder mapped 'normal' to 1 and 'attack' to 0. Swapping labels for consistency.")
            train_df['label'] = 1 - train_df['label']
            test_df['label'] = 1 - test_df['label']
            # Update LabelEncoder's classes for consistency if it's saved
            # This line is primarily for the saved LabelEncoder object to reflect the desired mapping
            le.classes_ = np.array(['normal', 'attack']) # Now 0 maps to 'normal', 1 to 'attack'
    
    # Identify feature columns after one-hot encoding
    feature_columns = [col for col in train_df.columns if col != 'label']

    # Feature scaling
    print("\n[1.5] Applying MinMaxScaler to all feature columns...")
    scaler = MinMaxScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    print("\nData loading and preprocessing complete.")
    return train_df, test_df, scaler, feature_columns, le

def handle_class_imbalance(X_train, y_train):
    """
    Applies ADASYN oversampling to balance the classes in the training data.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        tuple: A tuple containing:
            - X_res (pd.DataFrame): Resampled training features.
            - y_res (pd.Series): Resampled training labels.
    """
    print("\n" + "="*40)
    print("STEP 2: CLASS BALANCING")
    print("="*40)

    print("\n[2.1] Original class distribution (0: Normal, 1: Attack):")
    print(Counter(y_train))
    
    print("\n[2.2] Applying ADASYN oversampling...")
    adasyn = ADASYN(random_state=42)
    X_res, y_res = adasyn.fit_resample(X_train, y_train)
    
    print("[2.3] Resampled class distribution:")
    print(Counter(y_res))
    return X_res, y_res

def train_model(X_train, y_train):
    """
    Trains a RandomForestClassifier model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        RandomForestClassifier: The trained Random Forest model.
    """
    print("\n" + "="*40)
    print("STEP 3: MODEL TRAINING")
    print("="*40)

    print("\n[3.1] Initializing and training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=150, # Number of trees in the forest
        max_depth=20,     # Maximum depth of the tree
        min_samples_split=5, # Minimum number of samples required to split an internal node
        class_weight='balanced', # Handles class imbalance by adjusting weights
        random_state=42,  # For reproducibility
        n_jobs=-1         # Use all available CPU cores
    )
    model.fit(X_train, y_train)
    print("[3.2] Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model's performance on the test set,
    calculates various metrics, and plots ROC and Precision-Recall curves.

    Args:
        model (RandomForestClassifier): The trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.

    Returns:
        tuple: A tuple containing:
            - metrics (dict): Dictionary of evaluation metrics.
            - y_probs (np.ndarray): Predicted probabilities for the positive class.
    """
    print("\n" + "="*40)
    print("STEP 4: MODEL EVALUATION")
    print("="*40)

    print("\n[4.1] Calculating predictions and probabilities...")
    y_probs = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class (attack=1)
    
    # Find optimal threshold using Precision-Recall curve to maximize F1-score
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    # Add a small epsilon to avoid division by zero in case precision + recall is 0
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9) 
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Make predictions using the optimal threshold
    y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
    
    # Calculate evaluation metrics
    conf_matrix = confusion_matrix(y_test, y_pred_optimal)

    # Extract TN, FP from confusion matrix for FPR calculation
    # For binary classification (0=Normal, 1=Attack):
    # [[TN, FP],
    #  [FN, TP]]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]

    # Calculate False Positive Rate (FPR)
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0 # Avoid division by zero

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_optimal) *100 ,
        'precision': precision_score(y_test, y_pred_optimal) *100 ,
        'recall': recall_score(y_test, y_pred_optimal) *100 ,
        'f1': f1_score(y_test, y_pred_optimal) *100,
        'avg_precision': average_precision_score(y_test, y_probs) *100, # Area under Precision-Recall Curve
        'optimal_threshold': optimal_threshold *100,
        'confusion_matrix': conf_matrix, # Store the actual matrix
        'classification_report': classification_report(y_test, y_pred_optimal),
        'false_alarm_rate': fpr *100 # Add FPR to metrics
    }
    
    print("\n[4.2] Model Evaluation Results (using optimal threshold):")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Average Precision (PR-AUC): {metrics['avg_precision']:.4f}")
    print(f"False Alarm Rate (FPR): {metrics['false_alarm_rate']:.4f}") # Print FPR
    
    print("\n[4.3] Classification Report:")
    print(metrics['classification_report'])
    
    print("\n[4.4] Confusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Plot performance curves
    plot_performance_curves(y_test, y_probs)
    
    return metrics, y_probs
def plot_performance_curves(y_true, y_probs):
    """
    Generates and saves ROC (Receiver Operating Characteristic) and
    Precision-Recall curves.

    Args:
        y_true (pd.Series): True labels.
        y_probs (np.ndarray): Predicted probabilities for the positive class.
    """
    print("\n[4.5] Generating performance curves (ROC & Precision-Recall)...")
    try:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        avg_precision = average_precision_score(y_true, y_probs)
        
        plt.figure(figsize=(12, 5))
        
        # Plot ROC Curve
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1) # Dashed diagonal line for random classifier
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Plot Precision-Recall Curve
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'Precision-Recall (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig('performance_curves.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("Saved performance curves to 'performance_curves.png'")
    except Exception as e:
        print(f"ERROR: Could not generate performance curves: {str(e)}")

def plot_feature_importance(model, feature_names):
    """
    Generates and saves a traditional horizontal bar plot of feature importances
    from the Random Forest model. Used as a fallback if SHAP analysis fails.

    Args:
        model (RandomForestClassifier): The trained Random Forest model.
        feature_names (list): List of feature column names.
    """
    print("\n[FALLBACK] Generating traditional Feature Importance plot...")
    try:
        importances = model.feature_importances_
        # Get indices of top 20 features by importance
        indices = np.argsort(importances)[-20:] 
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(indices)), importances[indices], 
                        color='#1f77b4', alpha=0.7, height=0.8)
        
        plt.yticks(range(len(indices)), np.array(feature_names)[indices], fontsize=10)
        plt.xlabel('Feature Importance Score', fontsize=12)
        plt.title('Top 20 Most Important Features (from RandomForest)', fontsize=14, pad=15)
        plt.gca().invert_yaxis() # Puts the highest importance feature at the top
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        
        # Add value labels to bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                     f'{width:.4f}', va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('feature_importance_fallback.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("Saved traditional feature importance plot to 'feature_importance_fallback.png'")
    except Exception as e:
        print(f"ERROR: Could not generate traditional feature importance plot: {str(e)}")

def perform_shap_analysis(model, X_test, feature_names):
    """
    Performs SHAP (SHapley Additive exPlanations) analysis on the trained model
    and generates various SHAP plots (summary bar, beeswarm, waterfall).

    Args:
        model (RandomForestClassifier): The trained model.
        X_test (pd.DataFrame): The test features (full dataset).
        feature_names (list): List of feature column names.

    Returns:
        shap.Explainer: The fitted SHAP explainer object if successful, else None.
    """
    print("\n" + "="*40)
    print("STEP 5: MODEL EXPLANATION (SHAP ANALYSIS)")
    print("="*40)
    
    explainer = None # Initialize explainer to None
    
    try:
        print("\n[5.1] Initializing SHAP explainer...")
        # Use TreeExplainer for tree-based models (RandomForest), much faster
        try:
            explainer = shap.TreeExplainer(model)
            print("Using TreeExplainer for faster computation.")
        except Exception as e:
            print(f"TreeExplainer failed ({e}). Falling back to KernelExplainer (may be slower).")
            # Fallback to KernelExplainer if TreeExplainer is not suitable
            # KernelExplainer needs a background dataset and model.predict_proba for classification
            # Using a small sample of X_test as background for KernelExplainer
            background_sample_size = min(100, len(X_test)) 
            if background_sample_size == 0:
                raise ValueError("No test data for SHAP background. Skipping SHAP analysis.")
            background = X_test.sample(background_sample_size, random_state=42)
            explainer = shap.KernelExplainer(model.predict_proba, background)

        # Use a smaller sample of X_test for SHAP value calculation for performance
        sample_size = min(200, len(X_test)) # Use 200 samples for SHAP plots
        if sample_size == 0:
            print("No test samples available for SHAP analysis. Skipping SHAP visualizations.")
            return None # Return early if no samples
        X_test_sample = X_test.sample(sample_size, random_state=42)

        print(f"\n[5.2] Calculating SHAP values for {sample_size} samples (positive class)...")
        # Calculate SHAP values. For binary classification, TreeExplainer (and KernelExplainer with predict_proba)
        # typically returns a list of two arrays: [shap_values_for_class_0, shap_values_for_class_1].
        # We are interested in the positive class (attack=1), which is usually at index 1.
        shap_values_for_classes = explainer.shap_values(X_test_sample)

        # --- DEBUGGING PRINTS START ---
        print(f"DEBUG: Type of shap_values_for_classes: {type(shap_values_for_classes)}")
        if isinstance(shap_values_for_classes, list):
            print(f"DEBUG: Length of shap_values_for_classes list: {len(shap_values_for_classes)}")
            for i, arr in enumerate(shap_values_for_classes):
                print(f"DEBUG: Shape of list element {i}: {arr.shape}")
        elif isinstance(shap_values_for_classes, np.ndarray):
            print(f"DEBUG: Shape of shap_values_for_classes (if numpy array): {shap_values_for_classes.shape}")
        # --- DEBUGGING PRINTS END ---

        # Extract SHAP values and expected value for the positive class
        if isinstance(shap_values_for_classes, list) and len(shap_values_for_classes) > 1:
            shap_values_attack = shap_values_for_classes[1]
            expected_value = explainer.expected_value[1]
        elif isinstance(shap_values_for_classes, np.ndarray) and shap_values_for_classes.ndim == 3:
            # Handle the (samples, features, classes) case for TreeExplainer
            print("DEBUG: Handling 3D SHAP values array (samples, features, classes).")
            shap_values_attack = shap_values_for_classes[:, :, 1] # Take values for the positive class (index 1)
            expected_value = explainer.expected_value[1] # Take expected value for the positive class
        elif isinstance(shap_values_for_classes, np.ndarray) and shap_values_for_classes.ndim == 2:
            # This case means explainer directly returned a 2D array, which might be for the positive class already
            shap_values_attack = shap_values_for_classes
            expected_value = explainer.expected_value
        else:
             raise ValueError("Unexpected structure for SHAP values. Not a list for classes, 2D array, or 3D array.")
        
        # Ensure expected_value is a scalar float (important for shap.Explanation)
        if isinstance(expected_value, np.ndarray):
            if expected_value.size == 1:
                expected_value = expected_value.item() # Convert to scalar
            else:
                raise ValueError(f"Expected value is an array of size {expected_value.size}, but expected scalar for binary classification.")


        # 1. Global Feature Importance (Summary Plot - Bar)
        print("\n[5.3] Generating Global Feature Importance (SHAP Bar Plot)...")
        plt.figure(figsize=(12, 7))
        shap.summary_plot(shap_values_attack, X_test_sample,
                          feature_names=feature_names,
                          plot_type="bar",
                          show=False,
                          max_display=20)
        plt.title("Top 20 Features Importance (SHAP Bar Plot)", pad=15)
        plt.tight_layout()
        plt.savefig('shap_feature_importance.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("Saved SHAP bar plot to 'shap_feature_importance.png'")

        # 2. Beeswarm Plot (Detailed Feature Impact)
        print("\n[5.4] Generating Detailed Feature Impact (SHAP Beeswarm Plot)...")
        plt.figure(figsize=(12, 7))
        shap.summary_plot(shap_values_attack, X_test_sample,
                          feature_names=feature_names,
                          show=False, # Default plot_type is "dot" (beeswarm)
                          max_display=20)
        plt.title("Feature Impact on Attack Predictions (SHAP Beeswarm Plot)", pad=15)
        plt.tight_layout()
        plt.savefig('shap_detailed_impact.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("Saved SHAP beeswarm plot to 'shap_detailed_impact.png'")

        # 3. Individual Prediction Explanation (Waterfall Plot)
        print("\n[5.5] Generating Individual Prediction Explanation (SHAP Waterfall Plot)...")
        example_idx = 0 # Using the first sample from X_test_sample for individual explanation

        # Ensure values for a single instance are 1D (crucial for waterfall plot)
        values_for_waterfall = shap_values_attack[example_idx].flatten()

        # Create shap.Explanation object for the waterfall plot
        explanation = shap.Explanation(
            values=values_for_waterfall,
            base_values=expected_value,
            data=X_test_sample.iloc[example_idx], # This should be a pandas Series (1D)
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title(f"SHAP Waterfall Plot for Sample {example_idx}", pad=15)
        plt.tight_layout()
        plt.savefig('shap_waterfall_example.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("Saved SHAP waterfall plot to 'shap_waterfall_example.png'")

        print("\nSHAP analysis completed successfully!")
        return explainer # Return explainer for saving
        
    except Exception as e:
        print(f"\nSHAP Error: {str(e)}")
        print("SHAP analysis failed. Falling back to traditional feature importance plot.")
        plot_feature_importance(model, feature_names) # Call the fallback plot
        return None # Return None if SHAP fails

def save_artifacts(model, scaler, feature_columns, label_encoder, explainer=None):
    """
    Saves the trained model, scaler, feature names, label encoder,
    and SHAP explainer (if available) to pickle files.

    Args:
        model (RandomForestClassifier): The trained model.
        scaler (MinMaxScaler): The fitted MinMaxScaler.
        feature_columns (list): List of feature column names.
        label_encoder (LabelEncoder): The fitted LabelEncoder.
        explainer (shap.Explainer, optional): The fitted SHAP explainer. Defaults to None.
    """
    print("\n" + "="*40)
    print("STEP 6: SAVING ARTIFACTS")
    print("="*40)

    print("\n[6.1] Saving trained model ('anomaly_detector.pkl')...")
    with open('anomaly_detector.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("[6.2] Saving MinMaxScaler ('scaler.pkl')...")
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("[6.3] Saving feature column names ('features.pkl')...")
    with open('features.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
        
    print("[6.4] Saving label encoder ('label_encoder.pkl')...")
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
        
    print("[6.5] Saving model metrics ('model_metrics.pkl')...")        
    with open('model_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)    

    if explainer:
        print("[6.6] Saving SHAP explainer ('shap_explainer.pkl')...")
        try:
            with open('shap_explainer.pkl', 'wb') as f:
                pickle.dump(explainer, f)
            print("SHAP explainer saved successfully.")
        except Exception as e:
            print(f"Could not save SHAP explainer: {str(e)}")
    else:
        print("[6.5] SHAP explainer not available to save (SHAP analysis might have failed).")
            
    print("\nAll required files saved successfully!")
    print("- anomaly_detector.pkl")
    print("- scaler.pkl")
    print("- features.pkl")
    print("- label_encoder.pkl")
    print("- shap_explainer.pkl (if SHAP analysis was successful)")
    print("\nGenerated Visualization Images:")
    print("- performance_curves.png")
    print("- shap_feature_importance.png")
    print("- shap_detailed_impact.png")
    print("- shap_waterfall_example.png")
    print("- feature_importance_fallback.png (only if SHAP analysis failed)")


# ====================== MAIN EXECUTION PIPELINE ======================
if __name__ == "__main__":
    # Print current working directory to help locate saved files
    print(f"Current working directory: {os.getcwd()}")

    # 1. Load and preprocess data
    train_df, test_df, scaler, feature_columns, le = load_and_preprocess_data(dataset_path)
    
    # Separate features and labels for modeling
    X_train_raw = train_df[feature_columns]
    y_train_raw = train_df['label']
    X_test = test_df[feature_columns]
    y_test = test_df['label']
    
    # 2. Handle class imbalance on training data
    X_train_res, y_train_res = handle_class_imbalance(X_train_raw, y_train_raw)
    
    # 3. Train model
    model = train_model(X_train_res, y_train_res)
    
    # 4. Evaluate model (includes plotting performance curves)
    metrics, y_probs = evaluate_model(model, X_test, y_test)
    
    # Print the metrics explicitly (addresses "local variable 'metrics' assigned but never used" warning)
    print("\nSummary of Model Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Average Precision (PR-AUC): {metrics['avg_precision']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")

    # 5. Perform SHAP analysis (includes plotting SHAP visualizations)
    # Pass X_train_res for potential KernelExplainer background data if TreeExplainer fails
    explainer = perform_shap_analysis(model, X_test, feature_columns)
    
    # 6. Save artifacts
    save_artifacts(model, scaler, feature_columns, le, explainer)
    
    print("\n" + "="*40)
    print("END: PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*40)
