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
import logging
import os
import shutil
import ssl
from pathlib import Path

# Google Drive folder ID for test data
GDRIVE_FOLDER_ID = "1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU"
SEPARATOR = "============================================================"
logger = logging.getLogger(__name__)


def check_existing_files():
    """Check which files already exist."""
    model_exists = Path("models/best_exp20.pt").exists()
    images_root = Path("images")
    images_exist = len(list(images_root.glob("**/*.png"))) > 0

    gt_exists = len(list(images_root.glob("**/gt/*.json"))) > 0

    logger.info("%s", SEPARATOR)
    logger.info("📋 Checking existing files...")
    logger.info("%s", SEPARATOR)
    logger.info(
        f"Model (./models/best_exp20.pt):  {'✅ Found' if model_exists else '❌ Missing'}"
    )
    logger.info(
        f"Images (./images/**/*.png):      {'✅ Found' if images_exist else '❌ Missing'}"
    )
    logger.info(
        f"GT JSON (./images/**/gt/*.json):  {'✅ Found' if gt_exists else '❌ Missing'}"
    )

    return model_exists, images_exist and gt_exists


def suggest_download():
    """Suggest which files to download."""
    model_exists, images_exist = check_existing_files()

    if model_exists and images_exist:
        logger.info("✅ All required files are present!")
        return False

    logger.info("🔽 Missing files detected. Download from:")
    logger.info("   %s", GDRIVE_FOLDER_ID)
    logger.info("📌 Google Drive Link:")
    logger.info(
        "   https://drive.google.com/drive/folders/1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU"
    )

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

        logger.info("%s", SEPARATOR)
        logger.info("📥 Downloading via gdown...")
        logger.info("%s", SEPARATOR)

        # Create canonical project directories
        Path("models").mkdir(exist_ok=True)
        Path("images").mkdir(exist_ok=True)

        # Download folder
        output_dir = "./gdrive_data"
        logger.info("Downloading to: %s", output_dir)
        logger.info("⏳ This may take a few minutes...")

        gdown.download_folder(
            url=f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}",
            output=output_dir,
            quiet=False,
            use_cookies=False,
        )

        # Move model and images to correct locations
        gdrive_path = Path(output_dir)

        # Find and move model
        model_file = next(gdrive_path.glob("**/best_exp20.pt"), None)
        if model_file is not None:
            shutil.copy(model_file, Path("models") / "best_exp20.pt")
            logger.info("✅ Model copied to ./models/")

        # Preserve the downloaded images tree when available (keeps test/gt structure).
        image_root_candidates = [
            path for path in gdrive_path.glob("**/images") if path.is_dir()
        ]
        if image_root_candidates:
            image_root = next(iter(image_root_candidates))
            shutil.copytree(image_root, Path("images"), dirs_exist_ok=True)
            copied_pngs = list(Path("images").glob("**/*.png"))
            copied_jsons = list(Path("images").glob("**/gt/*.json"))
            logger.info("✅ %s images copied to ./images/", len(copied_pngs))
            logger.info("✅ %s GT JSON files copied to ./images/", len(copied_jsons))
        else:
            # Fall back to preserving any discovered test/gt subtree instead of flattening.
            gt_dir_candidates = [
                path for path in gdrive_path.glob("**/gt") if path.is_dir()
            ]
            if gt_dir_candidates:
                gt_dir = next(iter(gt_dir_candidates))
                test_dir = gt_dir.parent
                if test_dir.is_dir():
                    shutil.copytree(
                        test_dir, Path("images") / test_dir.name, dirs_exist_ok=True
                    )
                    copied_pngs = list(
                        (Path("images") / test_dir.name).glob("**/*.png")
                    )
                    copied_jsons = list(
                        (Path("images") / test_dir.name).glob("**/gt/*.json")
                    )
                    logger.info(
                        "✅ Preserved %s images and %s GT JSON files under ./images/%s/",
                        len(copied_pngs),
                        len(copied_jsons),
                        test_dir.name,
                    )
            else:
                image_files = list(gdrive_path.glob("**/*.png"))
                for img in image_files[:10]:
                    shutil.copy(img, Path("images") / img.name)
                if image_files:
                    logger.info(
                        "✅ %s images copied to ./images/", len(image_files[:10])
                    )

        # Cleanup
        shutil.rmtree(gdrive_path, ignore_errors=True)
        logger.info("✅ Download complete!")
        return True

    except ImportError:
        logger.error("❌ gdown not installed. Install with:")
        logger.error("   pip install gdown")
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
            logger.error("❌ Download failed due to SSL certificate verification.")
            logger.error("Try these steps:")
            logger.error("  1) Upgrade certificate bundle:")
            logger.error("     python -m pip install --upgrade certifi")
            logger.error("  2) Set CA bundle for this shell:")
            logger.error(
                '     PowerShell: $env:REQUESTS_CA_BUNDLE=(python -c "import certifi;print(certifi.where())")'
            )
            logger.error(
                '     PowerShell: $env:SSL_CERT_FILE=(python -c "import certifi;print(certifi.where())")'
            )
            logger.error(
                "  3) If your company uses SSL inspection, install your corporate root CA."
            )
            logger.error("  4) Or use manual download mode:")
            logger.error("     python download_test_data.py --manual")
        else:
            logger.error("❌ Download failed: %s", e)
        return False


def manual_download_instructions():
    """Print manual download instructions."""
    logger.info("%s", SEPARATOR)
    logger.info("📖 Manual Download Instructions")
    logger.info("%s", SEPARATOR)

    logger.info(
        """
1. Open the Google Drive link:
   https://drive.google.com/drive/folders/1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU

2. Download these items:
   - best_exp20.pt (model)
   - Sample images (*.png files)
   - Ground-truth JSON files under images/test/gt/

3. Place files in your project:
   ./models/best_exp20.pt
   ./images/test/*.png
   ./images/test/gt/*.json

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

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("%s", SEPARATOR)
    logger.info("🎵 WRS Test Data Downloader")
    logger.info("%s", SEPARATOR)

    # Check existing files
    model_exists, images_exist = check_existing_files()

    if args.manual:
        manual_download_instructions()
        return

    if model_exists and images_exist:
        logger.info("✅ All files present! You can run tests:")
        logger.info("   pytest tests/integration/test_e2e_local.py -v -s")
        return

    # Try automatic download
    logger.info("🔄 Attempting automatic download with gdown...")
    if download_via_gdown():
        logger.info("✅ Ready to run tests!")
        logger.info("   pytest tests/integration/test_e2e_local.py -v -s")
    else:
        manual_download_instructions()


if __name__ == "__main__":
    main()
