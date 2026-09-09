"""
Dataset Setup Script
Downloads the Brain Tumor MRI Dataset from Kaggle automatically.
Run this before training: python setup_dataset.py
"""

import os
import sys
import json
import zipfile

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


DATASET_ID   = "masoudnickparvar/brain-tumor-mri-dataset"
DATASET_DIR  = "dataset"
KAGGLE_JSON  = os.path.expanduser("~/.kaggle/kaggle.json")


def check_kaggle_creds():
    """Check if Kaggle API credentials are set up."""
    if os.path.exists(KAGGLE_JSON):
        print("✅ Kaggle credentials found at ~/.kaggle/kaggle.json")
        return True

    # Check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("✅ Kaggle credentials found in environment variables")
        return True

    print("\n" + "="*60)
    print("  ❌  Kaggle API credentials not found!")
    print("="*60)
    print("""
To download the dataset automatically, you need a Kaggle API key.

Option 1 — Kaggle API Key (Recommended):
  1. Go to https://www.kaggle.com/settings/account
  2. Click 'Create New Token' → downloads kaggle.json
  3. Place kaggle.json in: ~/.kaggle/kaggle.json
  4. Run this script again: python setup_dataset.py

Option 2 — Manual Download:
  1. Go to: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
  2. Download the ZIP file
  3. Extract it so you have:
       dataset/
         Training/
           glioma/        (images...)
           meningioma/    (images...)
           notumor/       (images...)
           pituitary/     (images...)
         Testing/
           glioma/
           meningioma/
           notumor/
           pituitary/
  4. Then run: python model/train.py
""")
    return False


def download_dataset():
    """Download dataset using Kaggle API."""
    try:
        import kaggle
    except ImportError:
        print("[ERROR] kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"\n📥 Downloading dataset '{DATASET_ID}'...")
    print("   This may take a few minutes (~300 MB)...")

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        DATASET_ID,
        path=DATASET_DIR,
        unzip=True,
        quiet=False,
    )
    print(f"\n✅ Dataset downloaded to '{DATASET_DIR}/'")


def verify_dataset():
    """Verify dataset structure."""
    expected = {
        "Training": ["glioma", "meningioma", "notumor", "pituitary"],
        "Testing":  ["glioma", "meningioma", "notumor", "pituitary"],
    }
    total = 0
    print("\n📋 Verifying dataset structure:")
    ok = True

    for split, classes in expected.items():
        split_dir = os.path.join(DATASET_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"   ❌ Missing: {split_dir}")
            ok = False
            continue
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"   ❌ Missing class dir: {cls_dir}")
                ok = False
                continue
            imgs = [f for f in os.listdir(cls_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f"   ✅ {split}/{cls}: {len(imgs)} images")
            total += len(imgs)

    if ok:
        print(f"\n🎉 Dataset ready! Total images: {total}")
    else:
        print("\n⚠️  Dataset structure incomplete. Please check the paths above.")

    return ok


import shutil

def download_via_kagglehub():
    """Try downloading dataset via kagglehub without requiring credentials."""
    try:
        import kagglehub
        print("\n📥 Downloading dataset via kagglehub...")
        cache_path = kagglehub.dataset_download(DATASET_ID)
        print(f"✅ Downloaded to cache: {cache_path}")
        
        # Check structure in cache_path
        src_training = os.path.join(cache_path, "Training")
        src_testing  = os.path.join(cache_path, "Testing")
        if not os.path.isdir(src_training):
            # sometimes nested inside a subfolder
            subdirs = [os.path.join(cache_path, d) for d in os.listdir(cache_path) if os.path.isdir(os.path.join(cache_path, d))]
            for sd in subdirs:
                if os.path.isdir(os.path.join(sd, "Training")):
                    src_training = os.path.join(sd, "Training")
                    src_testing  = os.path.join(sd, "Testing")
                    break
        
        if os.path.isdir(src_training) and os.path.isdir(src_testing):
            print("📦 Copying dataset files into project 'dataset/' folder...")
            os.makedirs(DATASET_DIR, exist_ok=True)
            dst_training = os.path.join(DATASET_DIR, "Training")
            dst_testing  = os.path.join(DATASET_DIR, "Testing")
            
            if not os.path.exists(dst_training):
                shutil.copytree(src_training, dst_training)
            if not os.path.exists(dst_testing):
                shutil.copytree(src_testing, dst_testing)
            return True
    except Exception as e:
        print(f"⚠️  kagglehub download failed or incomplete: {e}")
    return False


def get_dataset_stats():
    """Returns detailed image count statistics per split and per class."""
    expected = {
        "Training": ["glioma", "meningioma", "notumor", "pituitary"],
        "Testing":  ["glioma", "meningioma", "notumor", "pituitary"],
    }
    stats = {"total": 0, "splits": {}}
    for split, classes in expected.items():
        stats["splits"][split] = {"total": 0, "classes": {}}
        split_dir = os.path.join(DATASET_DIR, split)
        if os.path.isdir(split_dir):
            for cls in classes:
                cls_dir = os.path.join(split_dir, cls)
                if os.path.isdir(cls_dir):
                    count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    stats["splits"][split]["classes"][cls] = count
                    stats["splits"][split]["total"] += count
                    stats["total"] += count
                else:
                    stats["splits"][split]["classes"][cls] = 0
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Brain Tumor Dataset Setup & Stats")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--verify", action="store_true", help="Verify dataset structure only")
    parser.add_argument("--src", type=str, help="Path to local dataset directory (e.g., 'C:\\Users\\Nivas\\Desktop\\dataset mri')")
    args = parser.parse_args()

    print("=" * 60)
    print("  🧠  Brain Tumor Detection — Dataset Setup & Tools")
    print("=" * 60)

    if args.src:
        if os.path.exists(args.src):
            print(f"📦 Copying dataset from custom path: {args.src}")
            if os.path.exists(DATASET_DIR):
                shutil.rmtree(DATASET_DIR)
            shutil.copytree(args.src, DATASET_DIR)
            verify_dataset()
            sys.exit(0)
        else:
            print(f"❌ Source directory '{args.src}' not found.")
            sys.exit(1)

    if args.stats or args.verify:
        verify_dataset()
        sys.exit(0)

    # Check if already downloaded and ready
    if os.path.isdir(os.path.join(DATASET_DIR, "Training")):
        print("✅ Dataset directory found. Verifying...")
        if verify_dataset():
            sys.exit(0)

    # Check for local desktop dataset folders
    possible_local_paths = [
        r"C:\Users\Nivas\Desktop\dataset mri",
        r"C:\Users\Nivas\Desktop\datasetmri",
        os.path.expanduser("~/Desktop/dataset mri"),
        os.path.expanduser("~/Desktop/datasetmri"),
    ]
    for local_path in possible_local_paths:
        if os.path.isdir(os.path.join(local_path, "Training")):
            print(f"📦 Found local dataset at: {local_path}")
            print("   Copying into project 'dataset/' folder...")
            if os.path.exists(DATASET_DIR):
                shutil.rmtree(DATASET_DIR)
            shutil.copytree(local_path, DATASET_DIR)
            if verify_dataset():
                print("\n🚀 Dataset ready! Run: python model/train.py")
                sys.exit(0)

    # Try kagglehub first
    if download_via_kagglehub():
        if verify_dataset():
            print("\n🚀 Dataset ready! Run: python model/train.py")
            sys.exit(0)

    if not check_kaggle_creds():
        sys.exit(1)

    download_dataset()
    verify_dataset()

    print("\n🚀 Next step: Train the model")
    print("   python model/train.py")


