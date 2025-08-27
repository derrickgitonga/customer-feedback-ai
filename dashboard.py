import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import io

from inference import FeedbackAnalyzer
from models.theme_extractor import ThemeExtractor
import yaml

# Page configuration
st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize analyzer (cached for performance)
@st.cache_resource
def load_analyzer():
    return FeedbackAnalyzer()

@st.cache_resource
def load_theme_extractor():
    return ThemeExtractor()

analyzer = load_analyzer()
theme_extractor = load_theme_extractor()

# Main app
def main():
    st.title("📊 Customer Feedback Analyzer")
    st.markdown("""
    AI-powered analysis of customer feedback for sentiment classification, 
    theme extraction, and actionable insights.
    """)
    
    # Sidebar
    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=config['inference']['confidence_threshold'],
        help="Minimum confidence score for predictions"
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Single Feedback", "Batch Analysis", "Dashboard", "API Demo"
    ])
    
    with tab1:
        render_single_analysis()
    
    with tab2:
        render_batch_analysis()
    
    with tab3:
        render_dashboard()
    
    with tab4:
        render_api_demo()

def render_single_analysis():
    """Single feedback analysis interface"""
    st.header("Analyze Single Feedback")
    
    feedback_text = st.text_area(
        "Enter customer feedback:",
        height=150,
        placeholder="e.g., 'The app is amazing! Very user-friendly...'"
    )
    
    if st.button("Analyze", type="primary") and feedback_text:
        with st.spinner("Analyzing feedback..."):
            result = analyzer.analyze_single(feedback_text)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Category",
                    result['category'],
                    f"Confidence: {result['confidence']:.2%}"
                )
            
            with col2:
                st.metric("Themes", ", ".join(result['themes']) if result['themes'] else "None")
            
            with col3:
                urgency = "🔴 High" if result['category'] == 'urgent' else "🟢 Normal"
                st.metric("Urgency", urgency)
            
            # Confidence scores
            st.subheader("Confidence Scores")
            prob_df = pd.DataFrame.from_dict(
                result['probabilities'], 
                orient='index', 
                columns=['Probability']
            )
            prob_df = prob_df.sort_values('Probability', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=prob_df.index, y=prob_df['Probability'], ax=ax, palette='viridis')
            ax.set_ylabel('Probability')
            ax.set_xlabel('Category')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

def render_batch_analysis():
    """Batch feedback analysis interface"""
    st.header("Batch Feedback Analysis")
    
    upload_option = st.radio(
        "Choose input method:",
        ["Upload CSV File", "Paste Text List"]
    )
    
    if upload_option == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:", df.head())
            
            text_column = st.selectbox("Select text column", df.columns)
            
            if st.button("Analyze Batch", type="primary"):
                with st.spinner("Analyzing batch feedback..."):
                    results_df = analyzer.analyze_dataframe(df, text_column)
                    
                    st.success("Analysis completed!")
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "feedback_analysis.csv",
                        "text/csv"
                    )
    
    else:
        text_input = st.text_area(
            "Enter feedback texts (one per line):",
            height=200
        )
        
        if text_input and st.button("Analyze", type="primary"):
            texts = [line.strip() for line in text_input.split('\n') if line.strip()]
            with st.spinner(f"Analyzing {len(texts)} feedback entries..."):
                results = analyzer.analyze_batch(texts)
                results_df = pd.DataFrame(results)
                
                st.dataframe(results_df)
                
                # Summary statistics
                st.subheader("Batch Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Feedback", len(results))
                
                with col2:
                    urgent_count = sum(1 for r in results if r['category'] == 'urgent')
                    st.metric("Urgent Issues", urgent_count)
                
                with col3:
                    avg_confidence = sum(r['confidence'] for r in results) / len(results)
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")

def render_dashboard():
    """Interactive dashboard with visualizations"""
    st.header("Analytics Dashboard")
    
    # Sample data for demonstration
    sample_data = [
        "The app is amazing! Very user-friendly.",
        "I can't login to my account, need urgent help!",
        "Please add dark mode feature.",
        "The service is terrible and slow.",
        "Billing issue: charged twice.",
        "Love the new update! Great work team!",
        "System crashed and lost my data.",
        "Would be great to have export functionality."
    ]
    
    if st.button("Load Demo Data", type="secondary"):
        with st.spinner("Generating insights..."):
            results = analyzer.analyze_batch(sample_data)
            
            # Category distribution
            categories = [r['category'] for r in results]
            category_counts = pd.Series(categories).value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Category Distribution")
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                category_counts.plot.pie(autopct='%1.1f%%', ax=ax1)
                ax1.set_ylabel('')
                st.pyplot(fig1)
            
            with col2:
                st.subheader("Theme Analysis")
                all_themes = []
                for r in results:
                    all_themes.extend(r['themes'])
                
                if all_themes:
                    theme_counts = pd.Series(all_themes).value_counts()
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    theme_counts.plot.bar(ax=ax2, color='skyblue')
                    ax2.tick_params(axis='x', rotation=45)
                    st.pyplot(fig2)
                else:
                    st.info("No themes detected in sample data")
            
            # Confidence distribution
            st.subheader("Confidence Scores Distribution")
            confidences = [r['confidence'] for r in results]
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            sns.histplot(confidences, bins=10, ax=ax3, kde=True)
            ax3.set_xlabel('Confidence')
            ax3.set_ylabel('Count')
            st.pyplot(fig3)

def render_api_demo():
    """API demonstration interface"""
    st.header("API Integration Demo")
    
    st.markdown("""
    ### REST API Endpoint Example
    
    ```python
    import requests
    import json