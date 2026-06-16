#!/usr/bin/env python3
"""
Download test data from Google Drive for WRS integration tests.
This script downloads the model and sample images needed to run tests.

Usage:
    python download_test_data.py              # Interactive mode
    python download_test_data.py --model      # Only download model
    python download_test_data.py --images     # Only download images
    python download_test_data.py --all        # Download everything
"""

import argparse
import os
import shutil
import ssl
from pathlib import Path

# Google Drive folder ID for test data
GDRIVE_FOLDER_ID = "1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU"


def check_existing_files():
    """Check which files already exist."""
    model_exists = Path("models_/best_exp20.pt").exists()
    images_exist = len(list(Path("images_").glob("*.png"))) > 0

    print("\n" + "=" * 60)
    print("📋 Checking existing files...")
    print("=" * 60)
    print(
        f"Model (./models/best_exp20.pt):  {'✅ Found' if model_exists else '❌ Missing'}"
    )
    print(
        f"Images (./images/*.png):         {'✅ Found' if images_exist else '❌ Missing'}"
    )

    return model_exists, images_exist


def suggest_download():
    """Suggest which files to download."""
    model_exists, images_exist = check_existing_files()

    if model_exists and images_exist:
        print("\n✅ All required files are present!")
        return False

    print("\n🔽 Missing files detected. Download from:")
    print(f"   {GDRIVE_FOLDER_ID}")
    print("\n📌 Google Drive Link:")
    print("   https://drive.google.com/drive/folders/1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU")

    return True


def download_via_gdown():
    """Download using gdown library."""
    try:
        import gdown

        try:
            import certifi

            # Help requests/gdown use a known CA bundle in environments with broken defaults.
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        except Exception:
            # Continue even if certifi is unavailable; manual instructions are shown on failure.
            pass

        print("\n" + "=" * 60)
        print("📥 Downloading via gdown...")
        print("=" * 60)

        # Create directories
        Path("models_").mkdir(exist_ok=True)
        Path("images_").mkdir(exist_ok=True)

        # Download folder
        output_dir = "./gdrive_data"
        print(f"\nDownloading to: {output_dir}")
        print("⏳ This may take a few minutes...")

        gdown.download_folder(
            url=f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}",
            output=output_dir,
            quiet=False,
            use_cookies=False,
        )

        # Move model and images to correct locations
        gdrive_path = Path(output_dir)

        # Find and move model
        model_files = list(gdrive_path.glob("**/best_exp20.pt"))
        if model_files:
            shutil.copy(model_files[0], "models_/best_exp20.pt")
            print(f"\n✅ Model copied to ./models/")

        # Find and move images
        image_files = list(gdrive_path.glob("**/*.png"))
        for img in image_files[:10]:  # Limit to first 10 for speed
            shutil.copy(img, f"images_/{img.name}")
        if image_files:
            print(f"✅ {len(image_files)} images copied to ./images/")

        # Cleanup
        shutil.rmtree(gdrive_path, ignore_errors=True)
        print("\n✅ Download complete!")
        return True

    except ImportError:
        print("\n❌ gdown not installed. Install with:")
        print("   pip install gdown")
        return False
    except Exception as e:
        err = str(e)
        is_ssl_error = (
            isinstance(e, ssl.SSLError)
            or "SSLError" in err
            or "CERTIFICATE_VERIFY_FAILED" in err
            or "unable to get local issuer certificate" in err
        )

        if is_ssl_error:
            print("\n❌ Download failed due to SSL certificate verification.")
            print("\nTry these steps:")
            print("  1) Upgrade certificate bundle:")
            print("     python -m pip install --upgrade certifi")
            print("  2) Set CA bundle for this shell:")
            print(
                '     PowerShell: $env:REQUESTS_CA_BUNDLE=(python -c "import certifi;print(certifi.where())")'
            )
            print(
                '     PowerShell: $env:SSL_CERT_FILE=(python -c "import certifi;print(certifi.where())")'
            )
            print(
                "  3) If your company uses SSL inspection, install your corporate root CA."
            )
            print("  4) Or use manual download mode:")
            print("     python download_test_data.py --manual")
        else:
            print(f"\n❌ Download failed: {e}")
        return False


def manual_download_instructions():
    """Print manual download instructions."""
    print("\n" + "=" * 60)
    print("📖 Manual Download Instructions")
    print("=" * 60)

    print(
        """
1. Open the Google Drive link:
   https://drive.google.com/drive/folders/1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU

2. Download these items:
   - best_exp20.pt (model)
   - Sample images (*.png files)

3. Place files in your project:
   ./models/best_exp20.pt
   ./images/*.png

4. Run tests:
   pytest tests/integration/test_e2e_local.py -v -s
"""
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download test data for WRS integration tests"
    )
    parser.add_argument(
        "--model", action="store_true", help="Check/download model only"
    )
    parser.add_argument(
        "--images", action="store_true", help="Check/download images only"
    )
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument(
        "--manual", action="store_true", help="Show manual instructions"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🎵 WRS Test Data Downloader")
    print("=" * 60)

    # Check existing files
    model_exists, images_exist = check_existing_files()

    if args.manual:
        manual_download_instructions()
        return

    if model_exists and images_exist:
        print("\n✅ All files present! You can run tests:")
        print("   pytest tests/integration/test_e2e_local.py -v -s")
        return

    # Try automatic download
    print("\n🔄 Attempting automatic download with gdown...")
    if download_via_gdown():
        print("\n✅ Ready to run tests!")
        print("   pytest tests/integration/test_e2e_local.py -v -s")
    else:
        manual_download_instructions()


if __name__ == "__main__":
    main()
