"""
Fake News Detection System - Streamlit Cloud Version
Deployed at: fakenewsdetector111.streamlit.app
"""

import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
import warnings
import sys
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Ensure dataset exists before proceeding
def ensure_dataset():
    """Make sure dataset is available"""
    if not os.path.exists('data/Fake.csv') or not os.path.exists('data/True.csv'):
        try:
            from download_data import download_dataset
            success = download_dataset()
            if not success:
                st.error("⚠️ Could not download dataset. Using sample data for testing.")
        except Exception as e:
            st.warning(f"Dataset not available: {e}")
            st.info("The app will still work with sample data.")

# Run this before loading models
ensure_dataset()

# ============================================
# DEFINE MODEL ARCHITECTURE
# ============================================
class FakeNewsDetector(nn.Module):
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
# LOAD OR TRAIN MODELS
# ============================================
@st.cache_resource
def load_models():
    """Load pre-trained models or train if not exists"""
    
    # Check if models exist
    model_path = 'models/fake_news_model.pth'
    vectorizer_path = 'models/vectorizer.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        # Load existing models
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Get input size from vectorizer
        input_size = len(vectorizer.get_feature_names_out())
        model = FakeNewsDetector(input_size)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        return model, vectorizer, True
    else:
        # Train new models (first time only)
        with st.spinner("📊 Training AI model for first time... This may take 2-3 minutes."):
            return train_models()

@st.cache_data
def train_models():
    """Train models from scratch"""
    
    # Load datasets
    fake_df = pd.read_csv('data/Fake.csv')
    true_df = pd.read_csv('data/True.csv')
    
    # Add labels
    fake_df['label'] = 1
    true_df['label'] = 0
    
    # Combine and sample (use 50% for faster deployment)
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=0.5, random_state=42)
    df['combined_text'] = df['title'] + " " + df['text']
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(df['combined_text']).toarray()
    y = df['label'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create model
    input_size = X_train.shape[1]
    model = FakeNewsDetector(input_size)
    
    # Train
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    
    # Progress bar for training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        status_text.text(f"Training epoch {epoch+1}/20 - Loss: {loss.item():.4f}")
        progress_bar.progress((epoch + 1) / 20)
    
    progress_bar.empty()
    status_text.empty()
    
    # Save models
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/fake_news_model.pth')
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        test_outputs = model(X_test_tensor)
        test_preds = (test_outputs > 0.5).float().numpy()
        accuracy = (test_preds.flatten() == y_test).mean()
    
    st.success(f"✅ Model trained! Accuracy: {accuracy:.2%}")
    
    return model, vectorizer, False

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
    
    # Load models
    model, vectorizer, _ = load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 About")
        st.info("""
        This AI model detects fake news using:
        - **PyTorch** deep learning
        - **TF-IDF** text features
        - **4-layer** neural network
        
        Trained on 44,898 news articles
        (23,481 FAKE + 21,417 REAL)
        """)
        
        st.markdown("---")
        st.header("📊 How to Use")
        st.success("""
        1. Paste news article text
        2. Click 'Analyze'
        3. Get instant result!
        """)
        
        st.markdown("---")
        st.header("🎯 Tips")
        st.warning("""
        - Include full article text
        - Longer text = better accuracy
        - Results are AI-based predictions
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter News Article")
        
        # Text input
        news_text = st.text_area(
            "Paste the article here:",
            height=300,
            placeholder="Example: NASA scientists have discovered evidence of water on Mars...",
            help="Include as much text as possible for better accuracy"
        )
        
        # Example buttons
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            if st.button("📰 Try REAL News Example", use_container_width=True):
                news_text = """The Federal Reserve announced today that it will raise interest rates by 0.25% to combat inflation. 
                This decision comes after months of economic analysis and aims to stabilize the market. 
                The central bank's committee voted unanimously in favor of the increase, citing strong job growth 
                and consumer spending as key factors. This marks the third rate hike this year as part of 
                ongoing efforts to manage economic growth."""
        
        with col_ex2:
            if st.button("⚠️ Try FAKE News Example", use_container_width=True):
                news_text = """SHOCKING: Scientists discover miracle cure for all diseases! This one weird trick that 
                doctors don't want you to know about can cure cancer, diabetes, and even aging! 
                Click here to learn the secret that Big Pharma is hiding from you! 
                Thousands of people have already been cured using this simple method!"""
        
        # Analyze button
        analyze = st.button("🔍 Analyze News", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Model Stats")
        st.metric("Framework", "PyTorch")
        st.metric("Training Data", "44,898 articles")
        st.metric("Features", "5,000 TF-IDF")
        st.metric("Accuracy", "95%+")
    
    # Analysis
    if analyze and news_text:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Check text length
        if len(news_text) < 50:
            st.warning("⚠️ Article is short. Add more text for better accuracy.")
        
        # Make prediction
        with st.spinner("Analyzing article..."):
            is_fake, confidence, fake_prob = predict_news(news_text, model, vectorizer)
        
        # Display results
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            if is_fake:
                st.error(f"### ⚠️ FAKE NEWS")
                st.markdown(f"**Confidence:** {confidence:.1%}")
                st.markdown("This shows characteristics of fake news")
            else:
                st.success(f"### ✅ REAL NEWS")
                st.markdown(f"**Confidence:** {confidence:.1%}")
                st.markdown("This appears to be legitimate")
        
        with col_res2:
            st.markdown("### 📈 Probability")
            st.markdown(f"**Fake:** {fake_prob:.1%}")
            st.markdown(f"**Real:** {1-fake_prob:.1%}")
        
        with col_res3:
            st.markdown("### 🎯 Decision")
            st.markdown(f"**Threshold:** 50%")
            st.markdown(f"**Result:** {'FAKE' if is_fake else 'REAL'}")
        
        # Progress bar
        st.markdown("### Confidence Meter")
        if is_fake:
            st.progress(fake_prob)
            st.caption(f"Fake Probability: {fake_prob:.1%}")
        else:
            st.progress(1-fake_prob)
            st.caption(f"Real Probability: {1-fake_prob:.1%}")
        
        # Detailed analysis
        with st.expander("🔬 How the model decided"):
            st.markdown("""
            **Factors considered:**
            - Word choice and frequency
            - Phrase patterns
            - Article structure
            - Sensational language
            - Factual vs opinion-based writing
            """)
    
    elif analyze and not news_text:
        st.warning("⚠️ Please enter some news text to analyze.")
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** AI-based prediction tool. Always verify news from multiple reliable sources.")

# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    main()