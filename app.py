"""
Fake News Detection Web App
Built with Streamlit and PyTorch
"""

import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DEFINE THE SAME MODEL ARCHITECTURE
# ============================================
class FakeNewsDetector(nn.Module):
    """
    Neural Network Architecture (MUST match training architecture)
    """
    def __init__(self, input_size):
        super(FakeNewsDetector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

# ============================================
# CACHE MODEL LOADING (for performance)
# ============================================
@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    try:
        # Load vectorizer
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load model metadata
        with open('models/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Create model with correct input size
        input_size = metadata['input_size']
        model = FakeNewsDetector(input_size)
        
        # Load trained weights
        model.load_state_dict(torch.load('models/fake_news_model.pth', map_location='cpu'))
        model.eval()  # Set to evaluation mode
        
        return model, vectorizer, metadata
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you've trained the model first by running: python fake_news_final.py")
        return None, None, None

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_news(text, model, vectorizer):
    """Predict if news is real or fake"""
    # Convert text to features
    text_features = vectorizer.transform([text]).toarray()
    text_tensor = torch.FloatTensor(text_features)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(text_tensor)
        probability = prediction.item()
    
    # Return result
    is_fake = probability > 0.5
    confidence = probability if is_fake else 1 - probability
    
    return is_fake, confidence, probability

# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.title("📰 Fake News Detection System")
    st.markdown("### Powered by PyTorch Deep Learning")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, vectorizer, metadata = load_model_and_vectorizer()
    
    if model is None:
        st.stop()
    
    # Sidebar with model info
    with st.sidebar:
        st.header("🤖 Model Information")
        st.markdown(f"**Accuracy:** {metadata['test_accuracy']*100:.2f}%")
        st.markdown(f"**Training Data:** {metadata['num_articles']:,} articles")
        st.markdown(f"**FAKE Articles:** {metadata['num_fake']:,}")
        st.markdown(f"**REAL Articles:** {metadata['num_real']:,}")
        st.markdown(f"**Features:** {metadata['num_features']:,}")
        st.markdown(f"**Epochs:** {metadata['epochs']}")
        
        st.markdown("---")
        st.header("🏗️ Model Architecture")
        st.markdown("""""")

        st.markdown("---")
        st.header("📊 How It Works")
        st.info("""
1. Enter news article text
2. Model converts text to numbers
3. Neural network analyzes patterns
4. Returns probability of being FAKE or REAL
""")

        st.markdown("---")
        st.header("🎯 Tips")
        st.success("""
- Copy entire news article
- Include both title and content
- Longer articles give better results
""")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Enter News Article")

        # Text input area
        news_text = st.text_area(
            "Paste the full article here:",
            height=300,
            placeholder="Example: NASA scientists have discovered evidence of water on Mars...",
            help="Include as much text as possible for better accuracy"
        )

        # Example buttons
        st.markdown("**Try an example:**")
        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            if st.button("📰 Example REAL News"):
                news_text = """The Federal Reserve announced today that it will raise interest rates by 0.25% to combat inflation. 
        This decision comes after months of economic analysis and aims to stabilize the market. 
        The central bank's committee voted unanimously in favor of the increase, citing strong job growth 
        and consumer spending as key factors. This marks the third rate hike this year as part of 
        ongoing efforts to manage economic growth."""

        with col_ex2:
            if st.button("⚠️ Example FAKE News"):
                news_text = """SHOCKING: Scientists discover miracle cure for all diseases! This one weird trick that 
        doctors don't want you to know about can cure cancer, diabetes, and even aging! 
        Click here to learn the secret that Big Pharma is hiding from you! 
        Thousands of people have already been cured using this simple method!"""

        # Analyze button
        analyze_button = st.button("🔍 Analyze News", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Quick Stats")
        st.metric("Model Accuracy", f"{metadata['test_accuracy']*100:.1f}%")
        st.metric("Articles Trained", f"{metadata['num_articles']:,}")
        st.metric("Features Used", f"{metadata['num_features']:,}")

    # Analysis section
    if analyze_button and news_text:
        st.markdown("---")
        st.subheader("📊 Analysis Results")

        # Check text length
        if len(news_text) < 50:
            st.warning("⚠️ The article is quite short. For better accuracy, please paste more text (at least 50 characters).")

        # Make prediction
        with st.spinner("Analyzing article..."):
            is_fake, confidence, fake_prob = predict_news(news_text, model, vectorizer)

        # Display results in columns
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            if is_fake:
                st.error("### ⚠️ FAKE NEWS DETECTED")
                st.markdown(f"**Confidence:** {confidence:.1%}")
            else:
                st.success("### ✅ REAL NEWS")
                st.markdown(f"**Confidence:** {confidence:.1%}")

        with col_res2:
            st.markdown("### 📈 Probability Distribution")
            st.markdown(f"**Fake Probability:** {fake_prob:.1%}")
            st.markdown(f"**Real Probability:** {1-fake_prob:.1%}")

        with col_res3:
            st.markdown("### 🎯 Decision Threshold")
            st.markdown("**Threshold:** 50%")
            st.markdown(f"**Result:** {'FAKE' if is_fake else 'REAL'}")

        # Progress bar
        st.markdown("### 📊 Confidence Meter")
        if is_fake:
            st.progress(fake_prob)
            st.caption(f"Fake News Probability: {fake_prob:.1%}")
        else:
            st.progress(1-fake_prob)
            st.caption(f"Real News Probability: {1-fake_prob:.1%}")

        # Detailed explanation
        with st.expander("🔬 Detailed Analysis"):
            st.markdown("""
    **How the model made this prediction:**
    
    1. **Text Processing:** Your article was converted into numerical features using TF-IDF
    2. **Pattern Recognition:** The neural network analyzed these features
    3. **Classification:** Based on patterns learned from 44,898 articles
    
    **Key factors that influenced this decision:**
    - Word choice and frequency
    - Phrase patterns
    - Article structure
    - Presence of sensational language
    """)
            
            # Show text stats
            st.markdown("**Article Statistics:**")
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.metric("Character Count", len(news_text))
                st.metric("Word Count", len(news_text.split()))
            with col_stats2:
                st.metric("Unique Words", len(set(news_text.lower().split())))
                st.metric("Avg Word Length", f"{sum(len(word) for word in news_text.split()) / max(len(news_text.split()), 1):.1f}")

        # Disclaimer
        st.markdown("---")
        st.caption("⚠️ **Disclaimer:** This is an AI-based prediction tool. Results should be used as a reference, not absolute truth. Always verify news from multiple reliable sources.")

    elif analyze_button and not news_text:
        st.warning("⚠️ Please enter some news text to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("""
<div style='text-align: center'>
<p>Built with ❤️ using PyTorch and Streamlit</p>
<p>📊 Trained on 44,898 news articles (23,481 FAKE + 21,417 REAL)</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# RUN THE APP
# ============================================
if __name__ == "__main__":
    main()
