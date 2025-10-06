#!/usr/bin/env python3
"""
Download MovieLens dataset for LENS recommendation system
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

def download_movielens(data_dir="./data"):
    """Download and extract MovieLens 1M dataset"""
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    ml_path = data_path / "ml-1m"
    
    # Check if already downloaded
    if ml_path.exists() and any(ml_path.iterdir()):
        print(f"MovieLens dataset already exists at {ml_path}")
        return True
    
    # Download URL
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = data_path / "ml-1m.zip"
    
    print(f"Downloading MovieLens 1M dataset from {url}...")
    
    try:
        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"Progress: {percent:.1f}%", end='\r')
        
        urllib.request.urlretrieve(url, zip_path, download_progress)
        print("\nDownload complete!")
        
        # Extract
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        # Clean up
        zip_path.unlink()
        print(f"Dataset extracted to {ml_path}")
        
        # Verify files
        expected_files = ['movies.dat', 'ratings.dat', 'users.dat', 'README']
        actual_files = list(ml_path.glob('*.dat')) + list(ml_path.glob('README'))
        
        if len(actual_files) >= 3:
            print("\nDataset contents:")
            for file in ml_path.iterdir():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name}: {size_mb:.1f} MB")
            return True
        else:
            print("Warning: Dataset may be incomplete")
            return False
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False

def create_sample_data(data_dir="./data"):
    """Create small sample dataset for testing"""
    
    sample_path = Path(data_dir) / "sample"
    sample_path.mkdir(exist_ok=True)
    
    # Sample movies
    movies_data = """1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Grumpier Old Men (1995)::Comedy|Romance
4::Waiting to Exhale (1995)::Comedy|Drama
5::Father of the Bride Part II (1995)::Comedy
6::Heat (1995)::Action|Crime|Thriller
7::Sabrina (1995)::Comedy|Romance
8::Tom and Huck (1995)::Adventure|Children's
9::Sudden Death (1995)::Action
10::GoldenEye (1995)::Action|Adventure|Thriller"""
    
    # Sample users
    users_data = """1::F::1::10::48067
2::M::56::16::70072
3::M::25::15::55117
4::M::45::7::02460
5::M::25::20::55455"""
    
    # Sample ratings
    ratings_data = """1::1::5::978300760
1::2::3::978302109
1::3::4::978301968
2::1::4::978300275
2::2::5::978298413
3::1::4::978299941
3::4::3::978299942
4::5::5::978299000
5::6::4::978298900
5::7::3::978299100"""
    
    # Write files
    (sample_path / "movies.dat").write_text(movies_data)
    (sample_path / "users.dat").write_text(users_data)
    (sample_path / "ratings.dat").write_text(ratings_data)
    
    print(f"\nSample dataset created at {sample_path}")
    print("Files created:")
    for file in sample_path.glob("*.dat"):
        print(f"  - {file.name}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MovieLens dataset")
    parser.add_argument("--data-dir", default="./data", help="Data directory path")
    parser.add_argument("--sample", action="store_true", help="Create sample dataset only")
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_data(args.data_dir)
    else:
        success = download_movielens(args.data_dir)
        if success:
            print("\n✅ Dataset ready for use!")
        else:
            print("\n❌ Failed to download dataset")
            print("You can create a sample dataset with: python download_movielens.py --sample")

if __name__ == "__main__":
    main()