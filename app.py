import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FraudWatch Africa",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the trained models and preprocessors"""
    try:
        model = joblib.load('isolation_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return model, scaler, label_encoders
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please ensure all model files are in the same directory as this app.")
        return None, None, None

def wrangle(df):
    """
    Clean the dataset
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()

    # Strip whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()

    # Clean transaction_type: remove extra whitespace between words
    if 'transaction_type' in df_clean.columns:
        df_clean['transaction_type'] = df_clean['transaction_type'].str.strip()  # Remove leading/trailing spaces
        df_clean['transaction_type'] = df_clean['transaction_type'].str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with single space

    # Convert datetime column to proper datetime dtype if it exists
    if 'datetime' in df_clean.columns:
        df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
        
        # Extract hour and derive time_of_day if missing
        df_clean['hour'] = df_clean['datetime'].dt.hour

        def map_time_of_day(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 21:
                return 'evening'
            else:
                return 'night'

        df_clean['time_of_day'] = df_clean['hour'].map(map_time_of_day)
    else:
        # If no datetime column, create a default time_of_day
        df_clean['time_of_day'] = np.random.choice(['morning', 'afternoon', 'evening', 'night'], len(df_clean))

    # Drop columns if they exist
    columns_to_drop = ["Unnamed: 0", "hour"]
    if 'time_of_day(morning, afternoon, evening, night)' in df_clean.columns:
        columns_to_drop.append('time_of_day(morning, afternoon, evening, night)')
    
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    if existing_columns_to_drop:
        df_clean.drop(columns=existing_columns_to_drop, inplace=True)

    return df_clean

def validate_required_columns(df):
    """Validate that the dataframe has all required columns"""
    required_columns = [
        'transaction_id', 'user_id', 'transaction_type', 'amount', 'location',
        'device_type', 'network_provider', 'user_type', 'is_foreign_number',
        'is_sim_recently_swapped', 'has_multiple_accounts'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns

def preprocess_data(df, scaler, label_encoders):
    """Preprocess the data for prediction"""
    df_processed = df.copy()
    
    # Handle categorical columns (matching the training features)
    categorical_cols = ['transaction_type', 'location', 'device_type', 'network_provider', 'user_type', 'time_of_day']
    
    for col in categorical_cols:
        if col in df_processed.columns and col in label_encoders:
            # Handle unseen categories
            le = label_encoders[col]
            df_processed[col] = df_processed[col].astype(str)
            df_processed[col] = df_processed[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Scale all features (matching the training approach)
    feature_cols = ['amount', 'transaction_type', 'location', 'device_type', 
                   'network_provider', 'user_type', 'time_of_day',
                   'is_foreign_number', 'is_sim_recently_swapped', 'has_multiple_accounts']
    
    if scaler and all(col in df_processed.columns for col in feature_cols):
        df_processed[feature_cols] = scaler.transform(df_processed[feature_cols])
    
    return df_processed

def predict_fraud(df, model, scaler, label_encoders):
    """Make fraud predictions"""
    try:
        # Preprocess the data
        df_processed = preprocess_data(df, scaler, label_encoders)
        
        # Select features used in training (matching your training code)
        feature_cols = ['amount', 'transaction_type', 'location', 'device_type', 
                       'network_provider', 'user_type', 'time_of_day',
                       'is_foreign_number', 'is_sim_recently_swapped', 'has_multiple_accounts']
        
        X = df_processed[feature_cols]
        
        # Make predictions
        anomaly_scores = model.decision_function(X)
        predictions = model.predict(X)
        is_anomalous = (predictions == -1)
        
        return anomaly_scores, is_anomalous
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def get_risk_level(score):
    """Determine risk level based on anomaly score"""
    if score < -0.08:
        return "High", "🔴"
    elif score < -0.05:
        return "Medium", "🟡"
    elif score < 0:
        return "Low", "🟠"
    else:
        return "Normal", "🟢"

def create_visualizations(df):
    """Create various visualizations for the data"""
    
    st.subheader("📊 Data Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Distributions", "🌍 Geographic", "📱 Device & Network", "⚠️ Risk Analysis", "⏰ Time Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction Amount Distribution (Fixed: removed 'bins' parameter)
            fig_amount = px.histogram(df, x='amount', nbins=50, title='Transaction Amount Distribution')
            fig_amount.add_vline(x=df['amount'].mean(), line_dash="dash", line_color="red", 
                               annotation_text=f"Mean: {df['amount'].mean():.2f}")
            st.plotly_chart(fig_amount, use_container_width=True)
        
        with col2:
            # Transaction Types Distribution
            transaction_counts = df['transaction_type'].value_counts()
            fig_types = px.pie(values=transaction_counts.values, names=transaction_counts.index,
                             title='Transaction Types Distribution')
            st.plotly_chart(fig_types, use_container_width=True)
    
    with tab2:
        # Geographic Distribution
        location_counts = df['location'].value_counts().head(10)
        fig_location = px.bar(x=location_counts.index, y=location_counts.values,
                            title='Top 10 Transaction Locations',
                            labels={'x': 'Location', 'y': 'Number of Transactions'})
        fig_location.update_xaxes(tickangle=45)
        st.plotly_chart(fig_location, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Device Type Distribution
            device_counts = df['device_type'].value_counts()
            fig_device = px.bar(x=device_counts.index, y=device_counts.values,
                              title='Device Type Distribution',
                              labels={'x': 'Device Type', 'y': 'Number of Transactions'})
            st.plotly_chart(fig_device, use_container_width=True)
        
        with col2:
            # Network Provider Analysis
            network_counts = df['network_provider'].value_counts()
            fig_network = px.bar(x=network_counts.index, y=network_counts.values,
                               title='Network Provider Distribution',
                               labels={'x': 'Network Provider', 'y': 'Number of Transactions'})
            st.plotly_chart(fig_network, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            # User Type Distribution
            user_type_counts = df['user_type'].value_counts()
            fig_user = px.pie(values=user_type_counts.values, names=user_type_counts.index,
                            title='User Type Distribution')
            st.plotly_chart(fig_user, use_container_width=True)
        
        with col2:
            # Risk Indicators Analysis
            risk_cols = ['is_foreign_number', 'is_sim_recently_swapped', 'has_multiple_accounts']
            risk_data = df[risk_cols].sum()
            fig_risk = px.bar(x=risk_data.index, y=risk_data.values,
                            title='Risk Indicators Count',
                            labels={'x': 'Risk Indicator', 'y': 'Count'},
                            color=risk_data.values,
                            color_continuous_scale='reds')
            fig_risk.update_xaxes(tickangle=45)
            st.plotly_chart(fig_risk, use_container_width=True)
    
    with tab5:
        if 'time_of_day' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Time of Day Distribution
                time_counts = df['time_of_day'].value_counts()
                fig_time = px.bar(x=time_counts.index, y=time_counts.values,
                                title='Transaction Count by Time of Day',
                                labels={'x': 'Time of Day', 'y': 'Number of Transactions'},
                                color=time_counts.values,
                                color_continuous_scale='blues')
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col2:
                # Amount by Transaction Type Box Plot
                fig_box = px.box(df, x='transaction_type', y='amount',
                               title='Amount Distribution by Transaction Type')
                fig_box.update_xaxes(tickangle=45)
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Time analysis requires 'time_of_day' column. Please ensure your data includes datetime information.")

def get_shared_data():
    """Get data from session state"""
    if 'cleaned_data' in st.session_state:
        return st.session_state['cleaned_data']
    elif 'fraud_results' in st.session_state:
        return st.session_state['fraud_results']
    else:
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ FraudWatch Africa</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-time Mobile Money Fraud Detection System</p>', unsafe_allow_html=True)
    
    # Load models
    model, scaler, label_encoders = load_models()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🔧 System Controls")
    
    # Show data status
    shared_data = get_shared_data()
    if shared_data is not None:
        st.sidebar.success(f"✅ Data loaded: {len(shared_data)} records")
    else:
        st.sidebar.info("📤 Upload data to begin analysis")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose Function",
        ["🏠 Dashboard", "📤 Data Upload & Analysis", "🔍 Single Transaction Check", "📊 Batch Analysis", "📈 Analytics"]
    )
    
    if page == "🏠 Dashboard":
        dashboard_page(model, scaler, label_encoders)
    elif page == "📤 Data Upload & Analysis":
        data_upload_page(model, scaler, label_encoders)
    elif page == "🔍 Single Transaction Check":
        single_transaction_page(model, scaler, label_encoders)
    elif page == "📊 Batch Analysis":
        batch_analysis_page(model, scaler, label_encoders)
    elif page == "📈 Analytics":
        analytics_page()

def data_upload_page(model, scaler, label_encoders):
    """New page for data upload, cleaning, and visualization"""
    st.header("📤 Data Upload & Analysis")
    
    st.markdown("""
    Upload your transaction data (CSV or Excel) to:
    - ✨ Clean and preprocess the data
    - 📊 Generate comprehensive visualizations
    - 🔍 Detect fraud patterns
    - 📥 Download cleaned results
    """)
    
    # File upload - now supports both CSV and Excel
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", 
        type=['csv', 'xlsx', 'xls'],
        help="Upload your transaction data file"
    )
    
    if uploaded_file is not None:
        try:
            # Load data based on file type
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Loaded {len(df_raw)} rows and {len(df_raw.columns)} columns.")
            
            # Show raw data preview
            with st.expander("👀 Preview Raw Data", expanded=False):
                st.dataframe(df_raw.head(10))
                st.write(f"**Data Shape:** {df_raw.shape}")
                st.write(f"**Columns:** {list(df_raw.columns)}")
            
            # Data cleaning
            st.subheader("🧹 Data Cleaning")
            
            if st.button("🔄 Clean Data", type="primary"):
                with st.spinner("Cleaning data..."):
                    # Clean the data
                    df_clean = wrangle(df_raw)
                    
                    # Validate required columns
                    missing_cols = validate_required_columns(df_clean)
                    
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {missing_cols}")
                        st.info("""
                        **Required columns:**
                        - transaction_id, user_id, transaction_type, amount, location
                        - device_type, network_provider, user_type
                        - is_foreign_number, is_sim_recently_swapped, has_multiple_accounts
                        """)
                    else:
                        st.success("✅ Data cleaning completed successfully!")
                        
                        # Store cleaned data in session state
                        st.session_state['cleaned_data'] = df_clean
                        
                        # Show cleaning summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Rows", len(df_raw))
                        with col2:
                            st.metric("Cleaned Rows", len(df_clean))
                        with col3:
                            rows_removed = len(df_raw) - len(df_clean)
                            st.metric("Rows Removed", rows_removed)
                        
                        # Show cleaned data preview
                        with st.expander("👀 Preview Cleaned Data", expanded=True):
                            st.dataframe(df_clean.head(10))
            
            # Analysis and visualization section
            if 'cleaned_data' in st.session_state:
                df_clean = st.session_state['cleaned_data']
                
                st.subheader("📊 Data Analysis & Visualization")
                
                # Create tabs for different analysis options
                analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📈 Visualizations", "🔍 Fraud Detection", "📥 Download Results"])
                
                with analysis_tab1:
                    create_visualizations(df_clean)
                
                with analysis_tab2:
                    st.subheader("🔍 Fraud Detection Analysis")
                    
                    if st.button("🚨 Run Fraud Detection", type="primary"):
                        with st.spinner("Detecting fraud patterns..."):
                            scores, anomalies = predict_fraud(df_clean, model, scaler, label_encoders)
                            
                            if scores is not None:
                                # Add results to dataframe
                                df_results = df_clean.copy()
                                df_results['anomaly_score'] = scores
                                df_results['is_anomalous'] = anomalies
                                df_results['risk_level'] = [get_risk_level(score)[0] for score in scores]
                                
                                # Store results
                                st.session_state['fraud_results'] = df_results
                                
                                # Summary metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Transactions", len(df_results))
                                with col2:
                                    fraud_count = anomalies.sum()
                                    st.metric("Flagged Transactions", fraud_count)
                                with col3:
                                    fraud_rate = fraud_count / len(df_results) * 100
                                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                                with col4:
                                    high_risk_count = (scores < -0.08).sum()
                                    st.metric("High Risk", high_risk_count)
                                
                                # Risk level distribution
                                risk_counts = df_results['risk_level'].value_counts()
                                fig_risk_dist = px.bar(
                                    x=risk_counts.index, 
                                    y=risk_counts.values,
                                    title='Risk Level Distribution',
                                    color=risk_counts.index,
                                    color_discrete_map={
                                        'High': '#ff4444',
                                        'Medium': '#ffaa00', 
                                        'Low': '#ff8800',
                                        'Normal': '#44ff44'
                                    }
                                )
                                st.plotly_chart(fig_risk_dist, use_container_width=True)
                                
                                # Show flagged transactions
                                fraud_data = df_results[df_results['is_anomalous']].copy()
                                if not fraud_data.empty:
                                    st.subheader("🚨 Flagged Transactions")
                                    fraud_data_display = fraud_data.sort_values('anomaly_score').head(20)
                                    st.dataframe(fraud_data_display[['transaction_id', 'amount', 'transaction_type', 
                                                                   'location', 'risk_level', 'anomaly_score']])
                
                with analysis_tab3:
                    st.subheader("📥 Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download cleaned data
                        if st.button("📄 Download Cleaned Data"):
                            csv_clean = df_clean.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv_clean,
                                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    with col2:
                        # Download fraud detection results
                        if 'fraud_results' in st.session_state:
                            if st.button("🚨 Download Fraud Results"):
                                df_results = st.session_state['fraud_results']
                                csv_results = df_results.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results CSV",
                                    data=csv_results,
                                    file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Please make sure your file format is correct and contains the required columns.")
    
    else:
        # Show expected format when no file is uploaded
        st.subheader("📋 Expected Data Format")
        
        st.markdown("""
        Your CSV/Excel file should contain the following columns:
        
        **Required Columns:**
        - `transaction_id`: Unique transaction identifier
        - `user_id`: User identifier
        - `transaction_type`: Type of transaction (e.g., Send Money, Withdraw Cash)
        - `amount`: Transaction amount
        - `location`: Transaction location
        - `device_type`: Device used (Android, iOS, Feature Phone)
        - `network_provider`: Network provider (Safaricom, Airtel, etc.)
        - `user_type`: User type (individual, agent)
        - `is_foreign_number`: Binary flag (0/1)
        - `is_sim_recently_swapped`: Binary flag (0/1)
        - `has_multiple_accounts`: Binary flag (0/1)
        
        **Optional Columns:**
        - `datetime`: Transaction timestamp
        - Any additional columns will be preserved
        """)
        
        # Show sample format
        sample_df = pd.DataFrame({
            'transaction_id': ['TX100000', 'TX100001', 'TX100002'],
            'user_id': ['user_8270', 'user_1860', 'user_6390'],
            'transaction_type': ['Withdraw Cash', 'Send Money', 'Deposit Cash'],
            'amount': [2646.35, 2844.69, 2384.46],
            'location': ['Nakuru', 'Garissa', 'Nyeri'],
            'device_type': ['Feature Phone', 'iOS', 'Feature Phone'],
            'network_provider': ['Telkom Kenya', 'Safaricom', 'Telkom Kenya'],
            'user_type': ['individual', 'agent', 'agent'],
            'is_foreign_number': [0, 0, 0],
            'is_sim_recently_swapped': [0, 0, 1],
            'has_multiple_accounts': [0, 0, 2],
            'datetime': ['2024-06-16 21:45:13', '2024-06-05 00:49:25', '2024-06-13 15:54:02']
        })
        
        st.dataframe(sample_df)

def dashboard_page(model, scaler, label_encoders):
    st.header("📊 Real-time Dashboard")
    
    # Check if we have shared data
    shared_data = get_shared_data()
    
    if shared_data is not None:
        # Use uploaded data
        st.info("📊 Using uploaded data for dashboard analysis")
        sample_data = shared_data.copy()
        
        # If fraud results exist, use them; otherwise run prediction
        if 'fraud_results' in st.session_state and 'anomaly_score' in sample_data.columns:
            scores = sample_data['anomaly_score'].values
            anomalies = sample_data['is_anomalous'].values if 'is_anomalous' in sample_data.columns else (scores < 0)
        else:
            scores, anomalies = predict_fraud(sample_data, model, scaler, label_encoders)
            if scores is not None:
                sample_data['anomaly_score'] = scores
                sample_data['is_anomalous'] = anomalies
    else:
        # Generate sample data for demonstration
        st.info("📊 Using sample data for dashboard demonstration. Upload your data in the 'Data Upload & Analysis' section.")
        if st.button("🔄 Refresh Sample Data"):
            sample_data = generate_sample_data(100)
            
            if sample_data is not None:
                scores, anomalies = predict_fraud(sample_data, model, scaler, label_encoders)
                
                if scores is not None:
                    sample_data['anomaly_score'] = scores
                    sample_data['is_anomalous'] = anomalies
                else:
                    return
            else:
                return
        else:
            return
    
    if scores is not None:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(sample_data))
        
        with col2:
            fraud_count = anomalies.sum()
            st.metric("Flagged Transactions", fraud_count, f"{fraud_count/len(sample_data)*100:.1f}%")
        
        with col3:
            avg_score = scores[anomalies].mean() if fraud_count > 0 else 0
            st.metric("Avg Anomaly Score", f"{avg_score:.4f}")
        
        with col4:
            high_risk = (scores < -0.08).sum()
            st.metric("High Risk Alerts", high_risk)
        
        # Recent alerts
        st.subheader("🚨 Recent Fraud Alerts")
        fraud_data = sample_data[sample_data['is_anomalous']].copy()
        
        if not fraud_data.empty:
            fraud_data = fraud_data.sort_values('anomaly_score').head(10)
            
            for idx, row in fraud_data.iterrows():
                risk_level, emoji = get_risk_level(row['anomaly_score'])
                
                with st.expander(f"{emoji} {risk_level} Risk - Transaction {row['transaction_id']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Amount:** {row['amount']:,.2f} KES")
                        st.write(f"**Type:** {row['transaction_type']}")
                        st.write(f"**Location:** {row['location']}")
                        st.write(f"**User Type:** {row['user_type']}")
                    
                    with col2:
                        st.write(f"**Anomaly Score:** {row['anomaly_score']:.4f}")
                        st.write(f"**Device:** {row['device_type']}")
                        st.write(f"**Network:** {row['network_provider']}")
                        
                        risk_flags = []
                        if row['is_foreign_number']: risk_flags.append("Foreign Number")
                        if row['is_sim_recently_swapped']: risk_flags.append("Recent SIM Swap")
                        if row['has_multiple_accounts']: risk_flags.append("Multiple Accounts")
                        
                        if risk_flags:
                            st.write(f"**Risk Flags:** {', '.join(risk_flags)}")
        else:
            st.success("✅ No fraud alerts in recent transactions!")

def single_transaction_page(model, scaler, label_encoders):
    st.header("🔍 Single Transaction Analysis")
    
    # Check if we have data to provide options
    shared_data = get_shared_data()
    
    # Get unique values from shared data for dropdown options
    if shared_data is not None:
        transaction_types = sorted(shared_data['transaction_type'].unique().tolist())
        locations = sorted(shared_data['location'].unique().tolist())
        device_types = sorted(shared_data['device_type'].unique().tolist())
        network_providers = sorted(shared_data['network_provider'].unique().tolist())
        user_types = sorted(shared_data['user_type'].unique().tolist())
    else:
        # Default options
        transaction_types = ["Withdraw Cash", "Send Money", "Deposit Cash", "Lipa na M-Pesa", "Buy Airtime", "Pay Bill"]
        locations = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"]
        device_types = ["Android", "iOS", "Feature Phone"]
        network_providers = ["Safaricom", "Airtel", "Telkom Kenya"]
        user_types = ["individual", "agent"]
    
    with st.form("transaction_form"):
        st.subheader("Enter Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_id = st.text_input("Transaction ID", value="TX_TEST_001")
            user_id = st.text_input("User ID", value="user_12345")
            amount = st.number_input("Amount (KES)", min_value=0.0, value=1000.0, step=100.0)
            transaction_type = st.selectbox("Transaction Type", transaction_types)
            location = st.selectbox("Location", locations)
        
        with col2:
            device_type = st.selectbox("Device Type", device_types)
            network_provider = st.selectbox("Network Provider", network_providers)
            user_type = st.selectbox("User Type", user_types)
            
            st.subheader("Risk Factors")
            is_foreign_number = st.checkbox("Foreign Number")
            is_sim_recently_swapped = st.checkbox("Recent SIM Swap")
            has_multiple_accounts = st.checkbox("Multiple Accounts")
        
        submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary")
        
        if submitted:
            # Create dataframe for prediction
            transaction_data = pd.DataFrame({
                'transaction_id': [transaction_id],
                'user_id': [user_id],
                'amount': [amount],
                'transaction_type': [transaction_type],
                'location': [location],
                'device_type': [device_type],
                'network_provider': [network_provider],
                'user_type': [user_type],
                'is_foreign_number': [int(is_foreign_number)],
                'is_sim_recently_swapped': [int(is_sim_recently_swapped)],
                'has_multiple_accounts': [int(has_multiple_accounts)],
                'time_of_day': ['morning']  # Default time_of_day
            })
            
            # Make prediction
            scores, anomalies = predict_fraud(transaction_data, model, scaler, label_encoders)
            
            if scores is not None:
                score = scores[0]
                is_fraud = anomalies[0]
                risk_level, emoji = get_risk_level(score)
                
                # Display results
                st.subheader("🎯 Analysis Results")
                
                if is_fraud:
                    st.error(f"{emoji} **FRAUD ALERT!** This transaction is flagged as anomalous.")
                else:
                    st.success(f"{emoji} **Normal Transaction** - No fraud detected.")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Anomaly Score", f"{score:.4f}")
                with col2:
                    st.metric("Risk Level", risk_level)
                with col3:
                    st.metric("Fraud Probability", f"{max(0, -score*10):.1f}%")
                
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = -score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Score"},
                    delta = {'reference': 5},
                    gauge = {
                        'axis': {'range': [None, 15]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 5], 'color': "lightgreen"},
                            {'range': [5, 8], 'color': "yellow"},
                            {'range': [8, 15], 'color': "red"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 10}}))
                
                st.plotly_chart(fig, use_container_width=True)

def batch_analysis_page(model, scaler, label_encoders):
    st.header("📊 Batch Transaction Analysis")
    
    # Check if we have shared data
    shared_data = get_shared_data()
    
    if shared_data is not None:
        st.info("✅ Using data from Data Upload & Analysis section")
        df = shared_data.copy()
        
        # Show data preview
        st.subheader("📋 Data Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            if 'anomaly_score' in df.columns:
                fraud_count = df['is_anomalous'].sum() if 'is_anomalous' in df.columns else 0
                st.metric("Already Analyzed", f"{fraud_count} flagged")
            else:
                st.metric("Status", "Ready to analyze")
        
        st.dataframe(df.head())
        
        if st.button("🔍 Analyze All Transactions", type="primary"):
            with st.spinner("Analyzing transactions..."):
                scores, anomalies = predict_fraud(df, model, scaler, label_encoders)
                
                if scores is not None:
                    df['anomaly_score'] = scores
                    df['is_anomalous'] = anomalies
                    df['risk_level'] = [get_risk_level(score)[0] for score in scores]
                    
                    # Update session state
                    st.session_state['fraud_results'] = df
                    
                    # Summary metrics
                    st.subheader("📊 Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Transactions", len(df))
                    with col2:
                        fraud_count = anomalies.sum()
                        st.metric("Flagged as Fraud", fraud_count)
                    with col3:
                        st.metric("Fraud Rate", f"{fraud_count/len(df)*100:.2f}%")
                    with col4:
                        avg_score = scores.mean()
                        st.metric("Avg Anomaly Score", f"{avg_score:.4f}")
                    
                    # Risk level distribution
                    risk_counts = df['risk_level'].value_counts()
                    fig_risk_dist = px.bar(
                        x=risk_counts.index, 
                        y=risk_counts.values,
                        title='Risk Level Distribution',
                        color=risk_counts.index,
                        color_discrete_map={
                            'High': '#ff4444',
                            'Medium': '#ffaa00', 
                            'Low': '#ff8800',
                            'Normal': '#44ff44'
                        }
                    )
                    st.plotly_chart(fig_risk_dist, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"fraud_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Show flagged transactions
                    fraud_df = df[df['is_anomalous']].copy()
                    if not fraud_df.empty:
                        st.subheader("🚨 Flagged Transactions")
                        st.dataframe(fraud_df.sort_values('anomaly_score'))
    else:
        st.info("📤 Please upload data in the 'Data Upload & Analysis' section first.")
        st.markdown("""
        ### How to use Batch Analysis:
        1. Go to **Data Upload & Analysis** section
        2. Upload your CSV or Excel file
        3. Clean the data
        4. Return to this section for batch analysis
        
        Alternatively, you can upload a new file here:
        """)
        
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                # Load data based on file type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                    
                st.success(f"✅ Loaded {len(df)} transactions successfully!")
                
                # Clean the data
                df_clean = wrangle(df)
                missing_cols = validate_required_columns(df_clean)
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    return
                
                # Show data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df_clean.head())
                
                if st.button("🔍 Analyze All Transactions", type="primary"):
                    with st.spinner("Analyzing transactions..."):
                        scores, anomalies = predict_fraud(df_clean, model, scaler, label_encoders)
                        
                        if scores is not None:
                            df_clean['anomaly_score'] = scores
                            df_clean['is_anomalous'] = anomalies
                            df_clean['risk_level'] = [get_risk_level(score)[0] for score in scores]
                            
                            # Summary metrics
                            st.subheader("📊 Analysis Summary")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Transactions", len(df_clean))
                            with col2:
                                fraud_count = anomalies.sum()
                                st.metric("Flagged as Fraud", fraud_count)
                            with col3:
                                st.metric("Fraud Rate", f"{fraud_count/len(df_clean)*100:.2f}%")
                            with col4:
                                avg_score = scores.mean()
                                st.metric("Avg Anomaly Score", f"{avg_score:.4f}")
                            
                            # Download results
                            csv = df_clean.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"fraud_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Show flagged transactions
                            fraud_df = df_clean[df_clean['is_anomalous']].copy()
                            if not fraud_df.empty:
                                st.subheader("🚨 Flagged Transactions")
                                st.dataframe(fraud_df.sort_values('anomaly_score'))
            
            except Exception as e:
                st.error(f"Error processing file: {e}")

def analytics_page():
    st.header("📈 System Analytics")
    
    # Check if we have fraud results data
    shared_data = get_shared_data()
    
    if shared_data is not None and 'anomaly_score' in shared_data.columns:
        st.success("📊 Displaying analytics from uploaded data")
        df = shared_data.copy()
        
        # Main metrics
        st.subheader("🎯 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_transactions = len(df)
        fraud_transactions = df['is_anomalous'].sum() if 'is_anomalous' in df.columns else 0
        fraud_rate = (fraud_transactions / total_transactions * 100) if total_transactions > 0 else 0
        avg_transaction_amount = df['amount'].mean()
        
        with col1:
            st.metric("Total Transactions", f"{total_transactions:,}")
        with col2:
            st.metric("Fraud Transactions", f"{fraud_transactions:,}")
        with col3:
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col4:
            st.metric("Avg Transaction", f"KES {avg_transaction_amount:,.2f}")
        
        # Create analytics tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Anomaly Analysis", "🌍 Geographic Insights", "📱 Device & Network Analysis", "💰 Transaction Patterns"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Anomaly Score Distribution
                fig_scores = px.histogram(
                    df, x='anomaly_score', 
                    nbins=50, 
                    title='Anomaly Score Distribution',
                    labels={'anomaly_score': 'Anomaly Score', 'count': 'Number of Transactions'}
                )
                fig_scores.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Normal Threshold")
                st.plotly_chart(fig_scores, use_container_width=True)
            
            with col2:
                # Risk Level Distribution
                if 'risk_level' in df.columns:
                    risk_counts = df['risk_level'].value_counts()
                    fig_risk = px.pie(
                        values=risk_counts.values, 
                        names=risk_counts.index,
                        title='Risk Level Distribution',
                        color=risk_counts.index,
                        color_discrete_map={
                            'High': '#ff4444',
                            'Medium': '#ffaa00', 
                            'Low': '#ff8800',
                            'Normal': '#44ff44'
                        }
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
        
        with tab2:
            # Geographic analysis
            if 'is_anomalous' in df.columns:
                location_fraud = df.groupby('location').agg({
                    'is_anomalous': ['count', 'sum'],
                    'amount': 'mean'
                }).round(2)
                location_fraud.columns = ['Total_Transactions', 'Fraud_Count', 'Avg_Amount']
                location_fraud['Fraud_Rate'] = (location_fraud['Fraud_Count'] / location_fraud['Total_Transactions'] * 100).round(2)
                location_fraud = location_fraud.sort_values('Fraud_Rate', ascending=False).head(15)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_location_fraud = px.bar(
                        x=location_fraud.index, 
                        y=location_fraud['Fraud_Rate'],
                        title='Fraud Rate by Location (%)',
                        labels={'x': 'Location', 'y': 'Fraud Rate (%)'}
                    )
                    fig_location_fraud.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_location_fraud, use_container_width=True)
                
                with col2:
                    fig_location_volume = px.bar(
                        x=location_fraud.index, 
                        y=location_fraud['Total_Transactions'],
                        title='Transaction Volume by Location',
                        labels={'x': 'Location', 'y': 'Number of Transactions'}
                    )
                    fig_location_volume.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_location_volume, use_container_width=True)
        
        with tab3:
            if 'is_anomalous' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Device type fraud analysis
                    device_fraud = df.groupby('device_type').agg({
                        'is_anomalous': ['count', 'sum']
                    })
                    device_fraud.columns = ['Total', 'Fraud']
                    device_fraud['Fraud_Rate'] = (device_fraud['Fraud'] / device_fraud['Total'] * 100).round(2)
                    
                    fig_device = px.bar(
                        x=device_fraud.index, 
                        y=device_fraud['Fraud_Rate'],
                        title='Fraud Rate by Device Type (%)',
                        color=device_fraud['Fraud_Rate'],
                        color_continuous_scale='reds'
                    )
                    st.plotly_chart(fig_device, use_container_width=True)
                
                with col2:
                    # Network provider fraud analysis
                    network_fraud = df.groupby('network_provider').agg({
                        'is_anomalous': ['count', 'sum']
                    })
                    network_fraud.columns = ['Total', 'Fraud']
                    network_fraud['Fraud_Rate'] = (network_fraud['Fraud'] / network_fraud['Total'] * 100).round(2)
                    
                    fig_network = px.bar(
                        x=network_fraud.index, 
                        y=network_fraud['Fraud_Rate'],
                        title='Fraud Rate by Network Provider (%)',
                        color=network_fraud['Fraud_Rate'],
                        color_continuous_scale='reds'
                    )
                    st.plotly_chart(fig_network, use_container_width=True)
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                # Transaction type analysis
                if 'is_anomalous' in df.columns:
                    txn_fraud = df.groupby('transaction_type').agg({
                        'is_anomalous': ['count', 'sum'],
                        'amount': 'mean'
                    }).round(2)
                    txn_fraud.columns = ['Total', 'Fraud', 'Avg_Amount']
                    txn_fraud['Fraud_Rate'] = (txn_fraud['Fraud'] / txn_fraud['Total'] * 100).round(2)
                    
                    fig_txn = px.scatter(
                        x=txn_fraud['Avg_Amount'], 
                        y=txn_fraud['Fraud_Rate'],
                        size=txn_fraud['Total'],
                        hover_name=txn_fraud.index,
                        title='Transaction Amount vs Fraud Rate by Type',
                        labels={'x': 'Average Amount (KES)', 'y': 'Fraud Rate (%)'}
                    )
                    st.plotly_chart(fig_txn, use_container_width=True)
            
            with col2:
                # Risk factors correlation
                risk_cols = ['is_foreign_number', 'is_sim_recently_swapped', 'has_multiple_accounts']
                if 'is_anomalous' in df.columns:
                    risk_analysis = []
                    for col in risk_cols:
                        if col in df.columns:
                            fraud_with_risk = df[df[col] == 1]['is_anomalous'].mean() * 100
                            fraud_without_risk = df[df[col] == 0]['is_anomalous'].mean() * 100
                            risk_analysis.append({
                                'Risk_Factor': col.replace('_', ' ').title(),
                                'With_Risk': fraud_with_risk,
                                'Without_Risk': fraud_without_risk
                            })
                    
                    if risk_analysis:
                        risk_df = pd.DataFrame(risk_analysis)
                        fig_risk_factors = px.bar(
                            risk_df, 
                            x='Risk_Factor', 
                            y=['With_Risk', 'Without_Risk'],
                            title='Fraud Rate by Risk Factors (%)',
                            barmode='group'
                        )
                        fig_risk_factors.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_risk_factors, use_container_width=True)
        
        # Summary insights
        st.subheader("🔍 Key Insights")
        
        insights = []
        
        if fraud_rate > 5:
            insights.append(f"⚠️ **High fraud rate detected**: {fraud_rate:.1f}% of transactions are flagged as anomalous")
        elif fraud_rate < 1:
            insights.append(f"✅ **Low fraud rate**: Only {fraud_rate:.1f}% of transactions are flagged")
        
        if 'location' in df.columns and 'is_anomalous' in df.columns:
            high_risk_locations = df.groupby('location')['is_anomalous'].mean().sort_values(ascending=False).head(3)
            if high_risk_locations.iloc[0] > 0.1:  # More than 10% fraud rate
                insights.append(f"🎯 **High-risk locations**: {', '.join(high_risk_locations.head(3).index)}")
        
        if len(insights) == 0:
            insights.append("📊 **Overall**: Transaction patterns appear normal with standard risk distribution")
        
        for insight in insights:
            st.markdown(insight)
    
    else:
        st.info("📊 No analyzed data available. Please upload and analyze data first in the 'Data Upload & Analysis' section.")
        st.markdown("""
        ### Available Analytics:
        Once you upload and analyze your data, you'll see:
        
        - **Anomaly Score Distribution**: Understanding the spread of risk scores
        - **Geographic Risk Analysis**: Fraud patterns across different locations  
        - **Device & Network Insights**: Risk analysis by device type and network provider
        - **Transaction Pattern Analysis**: Fraud correlation with transaction types and amounts
        - **Risk Factor Analysis**: Impact of various risk indicators
        - **Key Performance Insights**: Automated insights from your data
        
        ### How to Generate Analytics:
        1. Go to **Data Upload & Analysis**
        2. Upload your transaction data
        3. Run fraud detection analysis
        4. Return here to view comprehensive analytics
        """)

def generate_sample_data(n=100):
    """Generate sample transaction data for demonstration"""
    np.random.seed(42)
    
    transaction_types = ['Withdraw Cash', 'Send Money', 'Deposit Cash', 'Lipa na M-Pesa', 'Buy Airtime', 'Pay Bill']
    locations = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Meru', 'Thika']
    devices = ['Android', 'iOS', 'Feature Phone']
    networks = ['Safaricom', 'Airtel', 'Telkom Kenya']
    user_types = ['individual', 'agent']
    time_of_days = ['morning', 'afternoon', 'evening', 'night']
    
    data = {
        'transaction_id': [f'TX{100000+i}' for i in range(n)],
        'user_id': [f'user_{np.random.randint(1000, 9999)}' for _ in range(n)],
        'amount': np.random.lognormal(7, 1, n),
        'transaction_type': np.random.choice(transaction_types, n),
        'location': np.random.choice(locations, n),
        'device_type': np.random.choice(devices, n),
        'network_provider': np.random.choice(networks, n),
        'user_type': np.random.choice(user_types, n, p=[0.85, 0.15]),
        'time_of_day': np.random.choice(time_of_days, n),
        'is_foreign_number': np.random.choice([0, 1], n, p=[0.97, 0.03]),
        'is_sim_recently_swapped': np.random.choice([0, 1], n, p=[0.95, 0.05]),
        'has_multiple_accounts': np.random.choice([0, 1], n, p=[0.90, 0.10])
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main()