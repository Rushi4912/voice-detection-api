#!/usr/bin/env python3
"""
Model Download Script

Downloads trained ML models from GitHub Releases on container startup.
This allows keeping models out of Git while ensuring they're available at runtime.
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path

# Configuration - Update these before deployment
GITHUB_REPO = "YOUR_GITHUB_USERNAME/voice-detection-api"  # UPDATE THIS
RELEASE_TAG = "v1.0.0-models"  # UPDATE THIS

# Model files to download
MODELS = {
    "resnet_asvspoof.pt": {
        "size_mb": 43,
        "sha256": None  # Optional: Add checksum for verification
    },
    "acoustic_classifier.pkl": {
        "size_mb": 0.5,
        "sha256": None
    },
    "acoustic_scaler.pkl": {
        "size_mb": 0.01,
        "sha256": None
    },
    "ensemble_weights.json": {
        "size_mb": 0.001,
        "sha256": None
    }
}

MODELS_DIR = Path(__file__).parent / "models"


def get_download_url(filename: str) -> str:
    """Get the GitHub Releases download URL for a model file"""
    return f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{filename}"


def download_file(url: str, dest_path: Path, show_progress: bool = True) -> bool:
    """Download a file from URL to destination path"""
    try:
        print(f"⬇️  Downloading: {dest_path.name}")
        print(f"   From: {url}")
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            if show_progress and total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r   Progress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\n✅ Downloaded: {dest_path.name}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {dest_path.name}: {e}")
        return False


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify file checksum"""
    if not expected_sha256:
        return True
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    actual = sha256_hash.hexdigest()
    if actual != expected_sha256:
        print(f"❌ Checksum mismatch for {file_path.name}")
        print(f"   Expected: {expected_sha256}")
        print(f"   Actual:   {actual}")
        return False
    
    return True


def download_models() -> bool:
    """Download all required models"""
    print("\n" + "="*60)
    print("🚀 Voice Detection API - Model Downloader")
    print("="*60 + "\n")
    
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    all_success = True
    
    for filename, config in MODELS.items():
        dest_path = MODELS_DIR / filename
        
        # Skip if already exists and valid
        if dest_path.exists():
            print(f"✅ Already exists: {filename}")
            continue
        
        # Download
        url = get_download_url(filename)
        success = download_file(url, dest_path)
        
        if success and config.get("sha256"):
            success = verify_checksum(dest_path, config["sha256"])
        
        if not success:
            all_success = False
    
    print("\n" + "="*60)
    if all_success:
        print("✅ All models downloaded successfully!")
    else:
        print("⚠️  Some models failed to download")
    print("="*60 + "\n")
    
    return all_success


def check_models_exist() -> bool:
    """Check if all required models exist"""
    for filename in MODELS.keys():
        if not (MODELS_DIR / filename).exists():
            return False
    return True


if __name__ == "__main__":
    # Check if models already exist
    if check_models_exist():
        print("✅ All models already present, skipping download")
        sys.exit(0)
    
    # Download models
    success = download_models()
    sys.exit(0 if success else 1)
