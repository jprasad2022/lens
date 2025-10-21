"""
Download MovieLens data from Google Cloud Storage
"""
import os
import time
from pathlib import Path
from google.cloud import storage
from google.api_core import exceptions

def download_movielens_data():
    """Download MovieLens data files from GCS if not present locally"""
    
    # Use local path for development, /app/data for production
    if os.path.exists("backend/data"):
        data_dir = Path("backend/data")  # Local development
    else:
        data_dir = Path("/app/data")  # Docker/Cloud Run
    
    data_dir.mkdir(exist_ok=True)
    
    bucket_name = "lens-data-940371601491"
    files_to_download = ["movies.dat", "ratings.dat", "users.dat"]
    
    # Check if files already exist
    all_exist = all((data_dir / f).exists() for f in files_to_download)
    if all_exist:
        print("✅ Data files already present")
        return
    
    print(f"📥 Downloading MovieLens data from gs://{bucket_name}/...")
    
    try:
        # Skip download in local development if GOOGLE_APPLICATION_CREDENTIALS not set
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') and os.path.exists("backend/data"):
            print("📁 Running locally with existing data files")
            return
            
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        for filename in files_to_download:
            local_path = data_dir / filename
            if local_path.exists():
                print(f"  ✓ {filename} already exists")
                continue
                
            print(f"  ⬇️  Downloading {filename}...", end="", flush=True)
            start_time = time.time()
            
            blob = bucket.blob(filename)
            blob.download_to_filename(str(local_path))
            
            download_time = time.time() - start_time
            file_size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f" Done! ({file_size_mb:.1f}MB in {download_time:.1f}s)")
            
        print("✅ All data files downloaded successfully")
        
    except exceptions.NotFound:
        print(f"❌ Bucket {bucket_name} not found")
    except exceptions.Forbidden:
        print(f"❌ Access denied to bucket {bucket_name}")
        print("   Make sure the service account has Storage Object Viewer permission")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print("   Continuing with empty data...")

if __name__ == "__main__":
    download_movielens_data()