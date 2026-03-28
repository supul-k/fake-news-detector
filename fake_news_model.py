"""
FAKE NEWS DETECTION SYSTEM
Complete implementation with PyTorch
Dataset: 44,898 news articles (23,481 FAKE + 21,417 REAL)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("="*80)
print("🚀 FAKE NEWS DETECTION SYSTEM - PyTorch Implementation")
print("="*80)
print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================
# STEP 1: LOAD AND COMBINE DATASETS
# ============================================
print("\n📂 STEP 1: Loading datasets...")
print("-"*50)

# Load both CSV files
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

print(f"✅ FAKE news loaded: {len(fake_df):,} articles")
print(f"✅ REAL news loaded: {len(true_df):,} articles")

# Add labels (1 = FAKE, 0 = REAL)
fake_df['label'] = 1
true_df['label'] = 0

# Combine and shuffle
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n📊 Combined dataset: {len(df):,} total articles")
print(f"   - FAKE articles: {sum(df['label']==1):,} ({sum(df['label']==1)/len(df)*100:.1f}%)")
print(f"   - REAL articles: {sum(df['label']==0):,} ({sum(df['label']==0)/len(df)*100:.1f}%)")

# Show sample data
print(f"\n📝 Sample data (first 2 rows):")
print(df[['title', 'subject', 'label']].head(2))

# ============================================
# STEP 2: DATA PREPROCESSING
# ============================================
print("\n🔤 STEP 2: Preprocessing text data...")
print("-"*50)

# Combine title and text for richer features
df['combined_text'] = df['title'] + " " + df['text']

print(f"✅ Created combined text (title + body)")
print(f"📝 Sample combined text:")
print(f"   {df['combined_text'].iloc[0][:200]}...")

# Check text lengths
text_lengths = df['combined_text'].str.len()
print(f"\n📊 Text statistics:")
print(f"   Average length: {text_lengths.mean():.0f} characters")
print(f"   Min length: {text_lengths.min():,} characters")
print(f"   Max length: {text_lengths.max():,} characters")

# ============================================
# STEP 3: CONVERT TEXT TO NUMERICAL FEATURES (TF-IDF)
# ============================================
print("\n🔢 STEP 3: Converting text to numerical features...")
print("-"*50)
print("Using TF-IDF Vectorizer (Term Frequency - Inverse Document Frequency)")
print("This converts text to numbers based on word importance")

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,           # Use top 5000 most important words
    stop_words='english',        # Remove common words (the, a, is, etc.)
    ngram_range=(1, 2),          # Use single words AND pairs of words
    max_df=0.7,                  # Ignore words that appear in >70% of docs
    min_df=2                     # Ignore words that appear in <2 docs
)

print("⏳ Converting text to features (this may take 1-2 minutes)...")
start_time = time.time()

# Convert text to feature matrix
X = vectorizer.fit_transform(df['combined_text']).toarray()
y = df['label'].values

elapsed_time = time.time() - start_time
print(f"✅ Conversion completed in {elapsed_time:.1f} seconds")
print(f"📊 Feature matrix shape: {X.shape}")
print(f"   - {X.shape[0]:,} articles")
print(f"   - {X.shape[1]:,} features (unique words/phrases)")

# ============================================
# STEP 4: SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================
print("\n✂️ STEP 4: Splitting data into train/test sets...")
print("-"*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% for testing
    random_state=42,         # Ensures same split every time
    stratify=y               # Maintains class balance
)

print(f"✅ Training set: {len(X_train):,} articles ({len(X_train)/len(X)*100:.1f}%)")
print(f"   - FAKE: {sum(y_train==1):,}")
print(f"   - REAL: {sum(y_train==0):,}")
print(f"\n✅ Test set: {len(X_test):,} articles ({len(X_test)/len(X)*100:.1f}%)")
print(f"   - FAKE: {sum(y_test==1):,}")
print(f"   - REAL: {sum(y_test==0):,}")

# ============================================
# STEP 5: CONVERT TO PYTORCH TENSORS
# ============================================
print("\n🔥 STEP 5: Converting to PyTorch tensors...")
print("-"*50)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

print(f"✅ Training tensor shape: {X_train_tensor.shape}")
print(f"✅ Test tensor shape: {X_test_tensor.shape}")
print(f"✅ Labels tensor shape: {y_train_tensor.shape}")

# ============================================
# STEP 6: DEFINE NEURAL NETWORK ARCHITECTURE
# ============================================
print("\n🧠 STEP 6: Creating neural network model...")
print("-"*50)

class FakeNewsDetector(nn.Module):
    """
    Neural Network Architecture:
    Input: 5000 features (TF-IDF)
    ↓
    Layer 1: Linear(5000 → 256) + ReLU + Dropout(30%)
    ↓
    Layer 2: Linear(256 → 128) + ReLU + Dropout(30%)
    ↓
    Layer 3: Linear(128 → 64) + ReLU
    ↓
    Output: Linear(64 → 1) + Sigmoid
    """
    def __init__(self, input_size):
        super(FakeNewsDetector, self).__init__()
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second hidden layer
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Third hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # Output layer
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model
input_size = X_train.shape[1]
model = FakeNewsDetector(input_size)

print(f"✅ Model created successfully!")
print(f"📊 Architecture details:")
print(f"   - Input layer: {input_size:,} neurons")
print(f"   - Hidden layer 1: 256 neurons")
print(f"   - Hidden layer 2: 128 neurons")
print(f"   - Hidden layer 3: 64 neurons")
print(f"   - Output layer: 1 neuron")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 Model parameters:")
print(f"   - Total: {total_params:,}")
print(f"   - Trainable: {trainable_params:,}")

# ============================================
# STEP 7: SETUP LOSS FUNCTION AND OPTIMIZER
# ============================================
print("\n⚙️ STEP 7: Configuring training setup...")
print("-"*50)

# Binary Cross Entropy Loss (for binary classification)
criterion = nn.BCELoss()

# Adam Optimizer (adapts learning rate automatically)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"✅ Loss function: Binary Cross Entropy")
print(f"✅ Optimizer: Adam (learning rate = 0.001)")
print(f"✅ Batch size: Full batch (all {len(X_train):,} articles per epoch)")

# ============================================
# STEP 8: TRAIN THE MODEL
# ============================================
print("\n🏋️ STEP 8: Training the model...")
print("-"*50)

epochs = 30
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

print(f"Training for {epochs} epochs...")
print("\nEpoch | Train Loss | Test Loss | Train Acc | Test Acc")
print("-"*60)

for epoch in range(epochs):
    # ========== TRAINING PHASE ==========
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # Calculate training accuracy
    train_preds = (outputs > 0.5).float().numpy()
    train_acc = accuracy_score(y_train, train_preds)
    train_accuracies.append(train_acc)
    
    # ========== TESTING PHASE ==========
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())
        
        # Calculate test accuracy
        test_preds = (test_outputs > 0.5).float().numpy()
        test_acc = accuracy_score(y_test, test_preds)
        test_accuracies.append(test_acc)
    
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"{epoch+1:3d}   | {loss.item():.4f}     | {test_loss.item():.4f}    | {train_acc:.4f}    | {test_acc:.4f}")

print("\n✅ Training completed!")

# ============================================
# STEP 9: FINAL EVALUATION
# ============================================
print("\n📊 STEP 9: Final model evaluation...")
print("-"*50)

model.eval()
with torch.no_grad():
    final_train_preds = (model(X_train_tensor) > 0.5).float().numpy()
    final_test_preds = (model(X_test_tensor) > 0.5).float().numpy()

final_train_acc = accuracy_score(y_train, final_train_preds)
final_test_acc = accuracy_score(y_test, final_test_preds)

print(f"🎯 FINAL RESULTS:")
print(f"   ✅ Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"   ✅ Test Accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")

print(f"\n📋 Detailed Classification Report (Test Set):")
print(classification_report(y_test, final_test_preds, target_names=['REAL', 'FAKE']))

# ============================================
# STEP 10: CONFUSION MATRIX
# ============================================
print("\n📊 STEP 10: Creating confusion matrix...")
print("-"*50)

cm = confusion_matrix(y_test, final_test_preds)
print("Confusion Matrix:")
print("                 Predicted")
print("                 REAL  FAKE")
print(f"Actual REAL     {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"       FAKE     {cm[1,0]:5d}  {cm[1,1]:5d}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['REAL', 'FAKE'], 
            yticklabels=['REAL', 'FAKE'])
plt.title('Confusion Matrix - Fake News Detection')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
print("✅ Confusion matrix saved as 'confusion_matrix.png'")

# ============================================
# STEP 11: PLOT TRAINING PROGRESS
# ============================================
print("\n📈 STEP 11: Creating training visualizations...")
print("-"*50)

# Plot 1: Loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
ax1.plot(test_losses, label='Test Loss', color='red', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Model Loss During Training')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy curves
ax2.plot(train_accuracies, label='Train Accuracy', color='blue', linewidth=2)
ax2.plot(test_accuracies, label='Test Accuracy', color='red', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Model Accuracy During Training')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=100, bbox_inches='tight')
print("✅ Training curves saved as 'training_curves.png'")

# ============================================
# STEP 12: SAVE MODEL AND VECTORIZER
# ============================================
print("\n💾 STEP 12: Saving model and vectorizer...")
print("-"*50)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save PyTorch model
torch.save(model.state_dict(), 'models/fake_news_model.pth')
print("✅ Model saved to 'models/fake_news_model.pth'")

# Save vectorizer
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Vectorizer saved to 'models/vectorizer.pkl'")

# Save model metadata
metadata = {
    'input_size': input_size,
    'train_accuracy': final_train_acc,
    'test_accuracy': final_test_acc,
    'num_articles': len(df),
    'num_features': X.shape[1],
    'num_fake': int(sum(y==1)),
    'num_real': int(sum(y==0)),
    'epochs': epochs
}

with open('models/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✅ Metadata saved to 'models/metadata.pkl'")

# ============================================
# STEP 13: TEST WITH EXAMPLES
# ============================================
print("\n🔍 STEP 13: Testing with sample articles...")
print("-"*50)

# Get a few examples from the test set
test_indices = [0, 100, 500, 1000, 2000]  # Sample different articles

print("Sample predictions on test articles:")
print("-"*80)

for idx in test_indices[:5]:  # Test first 5 samples
    if idx < len(X_test):
        # Get the article text
        article_text = df.iloc[idx]['combined_text'][:200] + "..."
        actual = "FAKE" if y_test[idx] == 1 else "REAL"
        predicted = "FAKE" if final_test_preds[idx] == 1 else "REAL"
        confidence = model(X_test_tensor[idx:idx+1]).item()
        
        print(f"\n📰 Article {idx+1}:")
        print(f"   Text: {article_text}")
        print(f"   Actual: {actual}")
        print(f"   Predicted: {predicted}")
        print(f"   Confidence: {confidence:.2%} FAKE, {1-confidence:.2%} REAL")
        print(f"   ✓ {'Correct' if actual == predicted else '✗ Incorrect'}")

# ============================================
# STEP 14: SAVE RESULTS SUMMARY
# ============================================
print("\n📝 STEP 14: Saving results summary...")
print("-"*50)

with open('models/training_summary.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("FAKE NEWS DETECTION - TRAINING SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Articles: {len(df):,}\n")
    f.write(f"FAKE Articles: {sum(y==1):,}\n")
    f.write(f"REAL Articles: {sum(y==0):,}\n\n")
    f.write(f"Model Architecture:\n")
    f.write(f"  - Input Features: {input_size:,}\n")
    f.write(f"  - Hidden Layer 1: 256 neurons\n")
    f.write(f"  - Hidden Layer 2: 128 neurons\n")
    f.write(f"  - Hidden Layer 3: 64 neurons\n")
    f.write(f"  - Output: 1 neuron (Sigmoid)\n")
    f.write(f"  - Total Parameters: {total_params:,}\n\n")
    f.write(f"Training Results:\n")
    f.write(f"  - Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)\n")
    f.write(f"  - Test Accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)\n\n")
    f.write(f"Confusion Matrix:\n")
    f.write(f"                 Predicted\n")
    f.write(f"                 REAL  FAKE\n")
    f.write(f"Actual REAL     {cm[0,0]:5d}  {cm[0,1]:5d}\n")
    f.write(f"       FAKE     {cm[1,0]:5d}  {cm[1,1]:5d}\n")

print("✅ Training summary saved to 'models/training_summary.txt'")

# ============================================
# FINISH
# ============================================
print("\n" + "="*80)
print("🎉 FAKE NEWS DETECTION SYSTEM - COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🏆 Final Test Accuracy: {final_test_acc*100:.2f}%")
print("="*80)
print("\n📁 Files created:")
print("   - models/fake_news_model.pth (PyTorch model)")
print("   - models/vectorizer.pkl (Text vectorizer)")
print("   - models/metadata.pkl (Model information)")
print("   - models/training_summary.txt (Results summary)")
print("   - confusion_matrix.png (Visualization)")
print("   - training_curves.png (Training progress)")
print("\n🚀 Next step: Run 'streamlit run app.py' to use the model!")
print("="*80)