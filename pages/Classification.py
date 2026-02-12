import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, matthews_corrcoef, cohen_kappa_score)

# Model imports
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Check if dataset exists
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

st.set_page_config(page_title="AutoML Classification", layout="wide")

# Sidebar
with st.sidebar: 
    st.image("https://static.javatpoint.com/tutorial/machine-learning/images/classification-algorithm-in-machine-learning.png")
    st.title("AutoML Classification")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This application helps you build and explore your classification models.")

# Helper Functions
def generate_profile_report(dataframe):
    """Generate comprehensive data profiling statistics"""
    profile = {}
    
    # Basic info
    profile['shape'] = dataframe.shape
    profile['duplicates'] = dataframe.duplicated().sum()
    profile['memory_usage'] = dataframe.memory_usage(deep=True).sum() / 1024**2  # MB
    
    # Column-wise analysis
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    
    profile['numeric_columns'] = len(numeric_cols)
    profile['categorical_columns'] = len(categorical_cols)
    profile['total_columns'] = len(dataframe.columns)
    
    return profile, numeric_cols, categorical_cols

def detect_data_types(dataframe):
    """Detect and categorize column types"""
    dtypes_info = {}
    for col in dataframe.columns:
        if dataframe[col].dtype in ['int64', 'float64']:
            dtypes_info[col] = 'numeric'
        else:
            dtypes_info[col] = 'categorical'
    return dtypes_info

def preprocess_data(dataframe, target_column, test_size=0.2, random_state=42):
    """Preprocess data similar to PyCaret setup"""
    
    # Separate features and target
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    
    # Encode target if categorical
    label_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle missing values
    # Numeric columns - impute with median
    if numeric_features:
        numeric_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
    else:
        numeric_imputer = None
    
    # Categorical columns - impute with mode and encode
    if categorical_features:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
        
        # One-hot encode categorical variables
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    else:
        categorical_imputer = None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for consistency
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    preprocessing_info = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'numeric_imputer': numeric_imputer,
        'categorical_imputer': categorical_imputer,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': X_train.columns.tolist()
    }
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessing_info

def get_model_list():
    """Return a dictionary of classification models"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K Neighbors Classifier': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Ridge Classifier': RidgeClassifier(random_state=42),
        'SVM - Linear': SVC(kernel='linear', random_state=42, probability=True),
    }
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds=10):
    """Evaluate a single model with cross-validation"""
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, 
                                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                                scoring='accuracy')
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2 else np.nan,
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_test, y_pred, average='weighted'),
        'Kappa': cohen_kappa_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
    }
    
    return metrics, model

def compare_all_models(X_train, X_test, y_train, y_test, cv_folds=10):
    """Compare all models and return results"""
    
    models = get_model_list()
    results = []
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_models = len(models)
    
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}... ({idx+1}/{total_models})")
        
        try:
            metrics, trained_model = evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds)
            metrics['Model'] = name
            results.append(metrics)
            trained_models[name] = trained_model
        except Exception as e:
            st.warning(f"Could not train {name}: {str(e)}")
        
        progress_bar.progress((idx + 1) / total_models)
    
    status_text.text("All models trained successfully!")
    progress_bar.empty()
    status_text.empty()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    col_order = ['Model', 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC', 'CV_Mean', 'CV_Std']
    results_df = results_df[col_order]
    
    # Sort by accuracy
    results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    return results_df, trained_models

def create_setup_dataframe(X_train, y_train, preprocessing_info):
    """Create a setup summary DataFrame similar to PyCaret"""
    
    setup_data = {
        'Description': [
            'Session id',
            'Target',
            'Target type',
            'Original data shape',
            'Transformed data shape',
            'Transformed train set shape',
            'Numeric features',
            'Categorical features',
            'Preprocessing',
            'Imputation type',
            'Numeric imputer',
            'Categorical imputer',
            'Normalize',
            'Normalize method',
            'Fold strategy',
            'Fold number'
        ],
        'Value': [
            '42',
            'Target Variable',
            'Binary/Multiclass',
            f"{X_train.shape[0] + X_train.shape[0]//4} x {len(preprocessing_info['numeric_features']) + len(preprocessing_info['categorical_features'])}",
            f"{X_train.shape[0] + X_train.shape[0]//4} x {X_train.shape[1]}",
            f"{X_train.shape[0]} x {X_train.shape[1]}",
            len(preprocessing_info['numeric_features']),
            len(preprocessing_info['categorical_features']),
            'True',
            'simple',
            'median',
            'mode',
            'True',
            'zscore',
            'StratifiedKFold',
            '10'
        ]
    }
    
    return pd.DataFrame(setup_data)

# Main App Logic
if choice == "Upload":
    st.title("📤 Upload Your Dataset")
    st.write("Upload a CSV file to begin your classification analysis.")
    
    file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        
        st.success(f"✅ Dataset uploaded successfully! Shape: {df.shape}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10))
        
        with col2:
            st.subheader("Dataset Info")
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
            st.write(f"**Duplicates:** {df.duplicated().sum()}")

elif choice == "Profiling":
    if 'df' not in locals():
        st.error("⚠️ Please upload a dataset first!")
    else:
        st.title("📊 Exploratory Data Analysis")
        
        profile, numeric_cols, categorical_cols = generate_profile_report(df)
        
        # Overview
        st.header("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", profile['shape'][0])
        with col2:
            st.metric("Total Columns", profile['total_columns'])
        with col3:
            st.metric("Numeric Features", profile['numeric_columns'])
        with col4:
            st.metric("Categorical Features", profile['categorical_columns'])
        
        # Missing values
        st.header("Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig = px.bar(x=missing_data.index, y=missing_data.values,
                        labels={'x': 'Columns', 'y': 'Missing Count'},
                        title='Missing Values by Column')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
        
        # Numeric features distribution
        if numeric_cols:
            st.header("Numeric Features Distribution")
            
            selected_numeric = st.multiselect(
                "Select numeric columns to visualize",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
            
            if selected_numeric:
                n_cols = min(2, len(selected_numeric))
                n_rows = (len(selected_numeric) + n_cols - 1) // n_cols
                
                fig = make_subplots(rows=n_rows, cols=n_cols,
                                   subplot_titles=selected_numeric)
                
                for idx, col in enumerate(selected_numeric):
                    row = idx // n_cols + 1
                    col_pos = idx % n_cols + 1
                    
                    fig.add_trace(
                        go.Histogram(x=df[col], name=col, showlegend=False),
                        row=row, col=col_pos
                    )
                
                fig.update_layout(height=300*n_rows, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Categorical features
        if categorical_cols:
            st.header("Categorical Features Distribution")
            
            selected_cat = st.selectbox("Select a categorical column", categorical_cols)
            
            if selected_cat:
                value_counts = df[selected_cat].value_counts().head(20)
                
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           labels={'x': selected_cat, 'y': 'Count'},
                           title=f'Distribution of {selected_cat}')
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix for numeric features
        if len(numeric_cols) > 1:
            st.header("Correlation Matrix")
            
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix,
                          labels=dict(color="Correlation"),
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          color_continuous_scale='RdBu_r',
                          aspect="auto")
            
            fig.update_layout(title="Feature Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.header("Statistical Summary")
        st.dataframe(df.describe())

elif choice == "Modelling":
    if 'df' not in locals():
        st.error("⚠️ Please upload a dataset first!")
    else:
        st.title("🤖 Model Training & Comparison")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            chosen_target = st.selectbox('🎯 Choose the Target Column', df.columns)
        
        with col2:
            cv_folds = st.slider('Cross-Validation Folds', 3, 15, 10)
        
        if st.button('🚀 Run Modelling', type='primary', use_container_width=True):
            
            with st.spinner('Preprocessing data...'):
                # Preprocess data
                X_train, X_test, y_train, y_test, preprocessing_info = preprocess_data(
                    df, chosen_target
                )
                
                # Create setup DataFrame
                setup_df = create_setup_dataframe(X_train, y_train, preprocessing_info)
                
                st.success("✅ Data preprocessing completed!")
                
                st.subheader("Setup Configuration")
                st.dataframe(setup_df, use_container_width=True)
            
            st.divider()
            
            with st.spinner('Training and comparing models... This may take a few minutes.'):
                # Compare models
                results_df, trained_models = compare_all_models(
                    X_train, X_test, y_train, y_test, cv_folds
                )
                
                st.success("✅ Model training completed!")
                
                st.subheader("Model Comparison Results")
                
                # Style the dataframe
                styled_df = results_df.style.background_gradient(
                    subset=['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC'],
                    cmap='RdYlGn'
                ).format({
                    'Accuracy': '{:.4f}',
                    'AUC': '{:.4f}',
                    'Recall': '{:.4f}',
                    'Precision': '{:.4f}',
                    'F1': '{:.4f}',
                    'Kappa': '{:.4f}',
                    'MCC': '{:.4f}',
                    'CV_Mean': '{:.4f}',
                    'CV_Std': '{:.4f}'
                })
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Get best model
                best_model_name = results_df.iloc[0]['Model']
                best_model = trained_models[best_model_name]
                
                st.success(f"🏆 Best Model: **{best_model_name}** with Accuracy: **{results_df.iloc[0]['Accuracy']:.4f}**")
                
                # Visualization
                st.subheader("Model Performance Visualization")
                
                # Accuracy comparison
                fig = px.bar(results_df, x='Model', y='Accuracy',
                           title='Model Accuracy Comparison',
                           color='Accuracy',
                           color_continuous_scale='Viridis')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Save best model with preprocessing info
                model_package = {
                    'model': best_model,
                    'preprocessing_info': preprocessing_info,
                    'model_name': best_model_name,
                    'metrics': results_df.iloc[0].to_dict()
                }
                
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump(model_package, f)
                
                st.info("💾 Best model saved successfully! Go to the Download tab to download it.")

elif choice == "Download":
    st.title("⬇️ Download Trained Model")
    
    if os.path.exists('best_model.pkl'):
        # Load and display model info
        with open('best_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        
        st.success("✅ Model is ready for download!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Information")
            st.write(f"**Model Type:** {model_package['model_name']}")
            st.write(f"**Accuracy:** {model_package['metrics']['Accuracy']:.4f}")
            st.write(f"**F1 Score:** {model_package['metrics']['F1']:.4f}")
        
        with col2:
            st.subheader("Download")
            with open('best_model.pkl', 'rb') as f:
                st.download_button(
                    label='📥 Download Model Package',
                    data=f,
                    file_name="best_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        st.info("ℹ️ The model package includes the trained model, preprocessing pipeline, and metadata.")
        
    else:
        st.warning("⚠️ No model found. Please train a model first in the Modelling section.")
