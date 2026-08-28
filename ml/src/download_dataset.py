import os
import zipfile
import sys
import requests
from ml.src.dataset_utils import TARGET_TOMATO_CLASSES

# Direct PlantVillage Tomato subset / master archive link
PLANTVILLAGE_ZIP_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

def download_and_extract_plantvillage():
    """
    Downloads and extracts the official PlantVillage dataset into ml/data/raw/.
    Gracefully falls back to detailed instructions if download fails or is constrained.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DATA_DIR, "plantvillage_master.zip")

    print("=== PlantVillage Dataset Downloader ===")
    print(f"Target raw directory: {RAW_DATA_DIR}")

    # Check if raw data already extracted
    extracted_dir = os.path.join(RAW_DATA_DIR, "PlantVillage-Dataset-master")
    if os.path.exists(extracted_dir):
        print(f"[OK] PlantVillage raw dataset already exists at {extracted_dir}")
        return True

    print(f"Attempting download from official repository: {PLANTVILLAGE_ZIP_URL}...")
    try:
        # Use requests stream with timeout and retry
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(PLANTVILLAGE_ZIP_URL, stream=True, timeout=60, headers=headers)
        response.raise_for_status()

        total_length = response.headers.get('content-length')
        total_size = int(total_length) if total_length else 0
        downloaded = 0

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        sys.stdout.write(f"\rDownloading PlantVillage zip: {percent}% ({downloaded / (1024*1024):.1f} MB)")
                    else:
                        sys.stdout.write(f"\rDownloading PlantVillage zip: ({downloaded / (1024*1024):.1f} MB)")
                    sys.stdout.flush()

        print("\n[OK] Download completed successfully.")

        print("Extracting zip archive into raw data directory...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print("[OK] Extraction completed successfully.")

        # Cleanup zip
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return True

    except Exception as e:
        print(f"\n[ERROR] Automatic download could not be completed automatically: {e}")
        print("\n--- MANUAL DATASET PLACEMENT INSTRUCTIONS ---")
        print("1. Download the PlantVillage dataset archive from official source:")
        print("   https://github.com/spMohanty/PlantVillage-Dataset")
        print("2. Extract the archive contents into:")
        print(f"   {RAW_DATA_DIR}/PlantVillage-Dataset-master/")
        print("3. Ensure the Tomato class folders are located at:")
        print(f"   {RAW_DATA_DIR}/PlantVillage-Dataset-master/raw/color/Tomato___*")
        print("--------------------------------------------------")
        return False

if __name__ == "__main__":
    download_and_extract_plantvillage()
