# download_data.py
import kagglehub
import pandas as pd
import os
import shutil
import sys

# Disable kagglehub telemetry (optional)
os.environ['KAGGLEHUB_DISABLE_TELEMETRY'] = '1'

# Set cache directory
os.environ['KAGGLEHUB_CACHE'] = '/tmp/kagglehub'  # For cloud deployment

def download_dataset():
    """Download fake news dataset from Kaggle using kagglehub"""
    
    os.makedirs('data', exist_ok=True)
    
    # Check if files already exist
    if os.path.exists('data/Fake.csv') and os.path.exists('data/True.csv'):
        print("✅ Dataset already exists!")
        return True
    
    print("📥 Downloading dataset from Kaggle...")
    
    try:
        # Download dataset
        path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
        print(f"✅ Downloaded to: {path}")
        
        # Find and copy CSV files
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        # Copy files to data folder
        for csv_file in csv_files:
            if 'Fake' in csv_file or 'fake' in csv_file:
                shutil.copy(csv_file, 'data/Fake.csv')
                print(f"✅ Copied Fake.csv")
            elif 'True' in csv_file or 'true' in csv_file:
                shutil.copy(csv_file, 'data/True.csv')
                print(f"✅ Copied True.csv")
        
        # Verify files
        if os.path.exists('data/Fake.csv') and os.path.exists('data/True.csv'):
            fake_df = pd.read_csv('data/Fake.csv')
            true_df = pd.read_csv('data/True.csv')
            print(f"✅ Dataset loaded: {len(fake_df)} fake, {len(true_df)} real articles")
            return True
        else:
            print("⚠️ CSV files not found, trying alternative...")
            return download_alternative()
            
    except Exception as e:
        print(f"❌ Kaggle download error: {e}")
        print("🔄 Trying alternative method...")
        return download_alternative()

def download_alternative():
    """Alternative download method if Kaggle fails"""
    
    print("📥 Downloading from alternative source...")
    
    # Alternative URLs (using GitHub raw files)
    fake_url = "https://raw.githubusercontent.com/laxmimerit/Fake-News-Detection/master/Fake.csv"
    true_url = "https://raw.githubusercontent.com/laxmimerit/Fake-News-Detection/master/True.csv"
    
    try:
        fake_df = pd.read_csv(fake_url)
        true_df = pd.read_csv(true_url)
        
        fake_df.to_csv('data/Fake.csv', index=False)
        true_df.to_csv('data/True.csv', index=False)
        
        print(f"✅ Downloaded: {len(fake_df)} fake, {len(true_df)} real")
        return True
        
    except Exception as e:
        print(f"❌ Alternative download failed: {e}")
        return create_sample_dataset()

def create_sample_dataset():
    """Create sample dataset as last resort"""
    
    print("📝 Creating sample dataset for testing...")
    
    # Sample data
    sample_fake = {
        'title': [
            "Miracle cure discovered! Doctors hate this!",
            "You won't believe what celebrity did!",
            "Government hiding shocking secret!"
        ],
        'text': [
            "This miracle cure can heal all diseases. Big Pharma doesn't want you to know!",
            "Famous celebrity caught in scandal. Click here to see the shocking photos!",
            "The government is hiding alien evidence from the public. This is huge!"
        ],
        'subject': ['Fake', 'Fake', 'Fake'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03']
    }
    
    sample_true = {
        'title': [
            "Scientists discover new exoplanet",
            "Government announces climate policy",
            "Local school wins national award"
        ],
        'text': [
            "Astronomers have discovered a new planet in the habitable zone of a nearby star.",
            "The government announced new environmental regulations to combat climate change.",
            "Local high school wins national science competition with innovative project."
        ],
        'subject': ['Real', 'Real', 'Real'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03']
    }
    
    fake_df = pd.DataFrame(sample_fake)
    true_df = pd.DataFrame(sample_true)
    
    fake_df.to_csv('data/Fake.csv', index=False)
    true_df.to_csv('data/True.csv', index=False)
    
    print(f"✅ Sample dataset created: {len(fake_df)} fake, {len(true_df)} real")
    return True

if __name__ == "__main__":
    download_dataset()