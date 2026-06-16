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

# Google Drive folder ID/URL for test data
GDRIVE_FOLDER_ID = "1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU"
GDRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU?usp=sharing"
)
SEPARATOR = "============================================================"
logger = logging.getLogger(__name__)


def _normalize_existing_files_status(status):
    """Accept both legacy 2-tuples and current 3-tuples from file checks."""
    if len(status) == 2:
        model_exists, images_exist = status
        # Legacy callers only reported model/image presence; treat that as sufficient.
        return model_exists, images_exist, images_exist

    if len(status) == 3:
        model_exists, images_exist, gt_exists = status
        return model_exists, images_exist, gt_exists

    raise ValueError(
        "check_existing_files() must return (model_exists, images_exist) or "
        "(model_exists, images_exist, gt_exists)"
    )


def _find_downloaded_dir(download_root: Path, dir_name: str) -> Path | None:
    """Find a downloaded directory anywhere under the temporary gdown output."""
    direct = download_root / dir_name
    if direct.is_dir():
        return direct

    return next(
        (path for path in download_root.glob(f"**/{dir_name}") if path.is_dir()), None
    )


def check_existing_files():
    """Check which files already exist."""
    model_exists = Path("models/best_exp20.pt").exists()
    images_root = Path("images")
    # Accept both legacy and nested image layouts.
    images_exist = (
        len(list(images_root.glob("*.png"))) > 0
        or len(list(images_root.glob("test/*.png"))) > 0
    )

    gt_exists = len(list(images_root.glob("test/gt/*.json"))) > 0

    logger.info("%s", SEPARATOR)
    logger.info("📋 Checking existing files...")
    logger.info("%s", SEPARATOR)
    logger.info(
        f"Model (./models/best_exp20.pt):  {'✅ Found' if model_exists else '❌ Missing'}"
    )
    logger.info(
        "Images (./images/*.png or ./images/test/*.png):      %s",
        "✅ Found" if images_exist else "❌ Missing",
    )
    logger.info(
        f"GT JSON (./images/test/gt/*.json):  {'✅ Found' if gt_exists else '❌ Missing'}"
    )

    # Keep backward-compatible semantics for callers/tests: images_exist means PNG assets are present.
    return model_exists, images_exist, gt_exists


def suggest_download():
    """Suggest which files to download."""
    model_exists, images_exist, gt_exists = _normalize_existing_files_status(
        check_existing_files()
    )

    if model_exists and images_exist and gt_exists:
        logger.info("✅ All required files are present!")
        return False

    logger.info("🔽 Missing files detected. Download from:")
    logger.info("   %s", GDRIVE_FOLDER_ID)
    logger.info("📌 Google Drive Link:")
    logger.info("   %s", GDRIVE_FOLDER_URL)

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
        output_dir = Path("./gdrive_data")
        shutil.rmtree(output_dir, ignore_errors=True)
        logger.info("Downloading to: %s", output_dir)
        logger.info("⏳ This may take a few minutes...")

        download_kwargs = {
            "url": GDRIVE_FOLDER_URL,
            "output": str(output_dir),
            "quiet": False,
            "use_cookies": False,
        }
        download_error = None
        try:
            try:
                gdown.download_folder(**download_kwargs, remaining_ok=True)
            except TypeError:
                # Compatibility with older gdown versions and unit-test fakes.
                gdown.download_folder(**download_kwargs)
        except Exception as exc:
            download_error = exc
            logger.warning(
                "⚠️ gdown reported a folder download error; checking whether the required files were still downloaded: %s",
                exc,
            )

        # Move model and images to correct locations
        gdrive_path = output_dir
        if not gdrive_path.exists():
            logger.error("❌ Download folder was not created: %s", gdrive_path)
            return False

        models_dir = _find_downloaded_dir(gdrive_path, "models")
        images_dir = _find_downloaded_dir(gdrive_path, "images")

        if models_dir is not None:
            shutil.copytree(models_dir, Path("models"), dirs_exist_ok=True)
        else:
            model_file = next(gdrive_path.glob("**/best_exp20.pt"), None)
            if model_file is not None:
                shutil.copy(model_file, Path("models") / "best_exp20.pt")

        if images_dir is not None:
            shutil.copytree(images_dir, Path("images"), dirs_exist_ok=True)

        copied_pngs = list(Path("images").glob("**/*.png"))
        copied_jsons = list(Path("images").glob("**/gt/*.json"))
        model_present = (Path("models") / "best_exp20.pt").exists()

        if model_present:
            logger.info("✅ Model copied to ./models/")
        else:
            logger.error("❌ best_exp20.pt was not found in downloaded data")

        logger.info("✅ %s images copied to ./images/", len(copied_pngs))
        logger.info("✅ %s GT JSON files copied to ./images/", len(copied_jsons))

        if not model_present or not copied_pngs:
            if download_error is not None:
                logger.error("❌ Download failed: %s", download_error)
            logger.error(
                "❌ Downloaded data is incomplete; required model or images are missing"
            )
            shutil.rmtree(gdrive_path, ignore_errors=True)
            return False

        if download_error is not None:
            logger.warning(
                "⚠️ Download completed with recoverable errors; the required files were obtained successfully."
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
   https://drive.google.com/drive/folders/1Ncz8UTeSilGqF_aU1uVjpPWdHMSErZqU?usp=sharing

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
    model_exists, images_exist, _gt_exists = _normalize_existing_files_status(
        check_existing_files()
    )

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
