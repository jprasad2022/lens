"""
Download model files from Google Cloud Storage
"""
import os
import time
from pathlib import Path
from google.cloud import storage
from google.api_core import exceptions


def download_model_registry():
    """Download model registry files from GCS if not present locally"""

    # Use local path for development, /app/model_registry for production
    if os.path.exists("backend/model_registry"):
        model_dir = Path("backend/model_registry")  # Local development
    else:
        model_dir = Path("/app/model_registry")  # Docker/Cloud Run

    model_dir.mkdir(exist_ok=True)

    bucket_name = "lens-data-940371601491"
    prefix = "model_registry/"

    # Check if models already exist
    model_folders = ["als", "collaborative", "popularity"]
    all_exist = all((model_dir / folder).exists() for folder in model_folders)
    if all_exist:
        print("✅ Model files already present")
        return

    print(f"📥 Downloading model registry from gs://{bucket_name}/{prefix}...")

    try:
        # Skip download in local development if GOOGLE_APPLICATION_CREDENTIALS not set
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') and os.path.exists("backend/model_registry"):
            print("📁 Running locally with existing model files")
            return

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Download all files with the model_registry/ prefix
        blobs = bucket.list_blobs(prefix=prefix)
        downloaded_count = 0

        for blob in blobs:
            # Skip if it's just the folder marker
            if blob.name.endswith('/'):
                continue

            # Create relative path (remove prefix)
            relative_path = blob.name[len(prefix):]
            local_path = model_dir / relative_path

            # Create parent directories if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if local_path.exists():
                print(f"  ✓ {relative_path} already exists")
                continue

            print(f"  ⬇️  Downloading {relative_path}...", end="", flush=True)
            start_time = time.time()

            blob.download_to_filename(str(local_path))

            download_time = time.time() - start_time
            file_size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f" Done! ({file_size_mb:.1f}MB in {download_time:.1f}s)")
            downloaded_count += 1

        if downloaded_count > 0:
            print(f"✅ Downloaded {downloaded_count} model files successfully")
        else:
            print("✅ All model files already present")

    except exceptions.NotFound:
        print(f"❌ Bucket {bucket_name} not found")
    except exceptions.Forbidden:
        print(f"❌ Access denied to bucket {bucket_name}")
        print("   Make sure the service account has Storage Object Viewer permission")
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        print("   Continuing without models...")


if __name__ == "__main__":
    download_model_registry()