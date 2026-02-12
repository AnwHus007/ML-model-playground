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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                             mean_absolute_percentage_error)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Regression Model imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Check if dataset exists
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

st.set_page_config(page_title="AutoML Regression", layout="wide")

# Sidebar
with st.sidebar: 
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1200px-Linear_regression.svg.png")
    st.title("AutoML Regression")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This application helps you build and explore your regression models.")

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

def preprocess_data(dataframe, target_column, test_size=0.2, random_state=42):
    """Preprocess data for regression"""
    
    # Separate features and target
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle missing values & Encoding
    # Numeric columns - impute with median
    if numeric_features:
        numeric_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
    else:
        numeric_imputer = None
    
    # Categorical columns - impute with mode and One-Hot Encode
    if categorical_features:
        # Impute
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
        
        # One-hot encode (drop_first to avoid multicollinearity in linear models)
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    else:
        categorical_imputer = None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale numeric features (Important for Linear/SVM/KNN models)
    # We fit on train and transform on test to avoid leakage
    scaler = StandardScaler()
    
    # We only scale the columns that were originally numeric (or all if we want)
    # For simplicity in this auto-tool, we scale everything after encoding
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
        'feature_names': X_train.columns.tolist()
    }
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessing_info

def get_model_list():
    """Return a dictionary of regression models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'Elastic Net': ElasticNet(random_state=42),
        'Huber Regressor': HuberRegressor(),
        'K Neighbors Regressor': KNeighborsRegressor(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
        'XGBoost': XGBRegressor(random_state=42, eval_metric='rmse'),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
        'Bayesian Ridge': BayesianRidge(),
    }
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test, cv_folds=10):
    """Evaluate a single regression model"""
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Cross-validation scores (using R2)
    cv_scores = cross_val_score(model, X_train, y_train, 
                                cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
                                scoring='r2')
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # RMSLE (Root Mean Squared Logarithmic Error) - handle negative values safely
    try:
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred)))
    except:
        rmsle = np.nan
    
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'MAPE': mape,
        'RMSLE': rmsle,
        'CV_R2_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
    }
    
    return metrics, model

def compare_all_models(X_train, X_test, y_train, y_test, cv_folds=10):
    """Compare all regression models"""
    
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
    col_order = ['Model', 'R2', 'RMSE', 'MAE', 'MAPE', 'MSE', 'RMSLE', 'CV_R2_Mean', 'CV_Std']
    results_df = results_df[col_order]
    
    # Sort by R2 (Descending)
    results_df = results_df.sort_values('R2', ascending=False).reset_index(drop=True)
    
    return results_df, trained_models

def create_setup_dataframe(X_train, y_train, preprocessing_info):
    """Create a setup summary DataFrame"""
    
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
            'Regression',
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
            'KFold',
            '10'
        ]
    }
    
    return pd.DataFrame(setup_data)

# Main App Logic
if choice == "Upload":
    st.title("📤 Upload Your Dataset")
    st.write("Upload a CSV file to begin your regression analysis.")
    
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
        
        # Correlation matrix
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

elif choice == "Modelling":
    if 'df' not in locals():
        st.error("⚠️ Please upload a dataset first!")
    else:
        st.title("🤖 Regression Modelling")
        
        # Identify numeric columns for target selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("❌ No numeric columns found in the dataset! Regression requires a numeric target.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                chosen_target = st.selectbox('🎯 Choose the Target Column', numeric_cols)
            
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
                    
                    # Style the dataframe (Higher R2 is Green, Lower RMSE/MAE is Green)
                    styled_df = results_df.style.background_gradient(
                        subset=['R2'], cmap='RdYlGn'
                    ).background_gradient(
                        subset=['RMSE', 'MAE', 'MSE', 'MAPE'], cmap='RdYlGn_r'
                    ).format({
                        'R2': '{:.4f}',
                        'RMSE': '{:.4f}',
                        'MAE': '{:.4f}',
                        'MSE': '{:.4f}',
                        'MAPE': '{:.4f}',
                        'RMSLE': '{:.4f}',
                        'CV_R2_Mean': '{:.4f}',
                        'CV_Std': '{:.4f}'
                    })
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Get best model
                    best_model_name = results_df.iloc[0]['Model']
                    best_model = trained_models[best_model_name]
                    
                    st.success(f"🏆 Best Model: **{best_model_name}** with R2: **{results_df.iloc[0]['R2']:.4f}**")
                    
                    # Visualization
                    st.subheader("Model Performance Visualization")
                    
                    # 1. Comparison Plot
                    fig1 = px.bar(results_df, x='Model', y='R2',
                               title='Model R2 Score Comparison',
                               color='R2',
                               color_continuous_scale='Viridis')
                    fig1.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # 2. Prediction vs Actual Plot (Best Model)
                    st.subheader(f"Analysis for {best_model_name}")
                    y_pred = best_model.predict(X_test)
                    
                    col_p1, col_p2 = st.columns(2)
                    
                    with col_p1:
                        # Scatter Plot: Actual vs Predicted
                        fig_scatter = px.scatter(x=y_test, y=y_pred, 
                                               labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                               title='Actual vs. Predicted Values')
                        # Add perfect prediction line
                        min_val = min(min(y_test), min(y_pred))
                        max_val = max(max(y_test), max(y_pred))
                        fig_scatter.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                                            line=dict(color="Red", dash="dash"))
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                    with col_p2:
                        # Residual Plot
                        residuals = y_test - y_pred
                        fig_resid = px.scatter(x=y_pred, y=residuals,
                                             labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                             title='Residual Plot')
                        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_resid, use_container_width=True)

                    # Save best model
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
            st.write(f"**R2 Score:** {model_package['metrics']['R2']:.4f}")
            st.write(f"**RMSE:** {model_package['metrics']['RMSE']:.4f}")
        
        with col2:
            st.subheader("Download")
            with open('best_model.pkl', 'rb') as f:
                st.download_button(
                    label='📥 Download Model Package',
                    data=f,
                    file_name="best_regression_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        st.info("ℹ️ The model package includes the trained model, preprocessing pipeline, and metadata.")
        
    else:
        st.warning("⚠️ No model found. Please train a model first in the Modelling section.")
