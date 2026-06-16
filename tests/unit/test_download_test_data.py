"""Unit tests for download_test_data.py."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _reload_module():
    """Reload module to isolate monkeypatched imports per test."""
    if "download_test_data" in sys.modules:
        del sys.modules["download_test_data"]
    return importlib.import_module("download_test_data")


class TestCheckExistingFiles:
    """Tests for existing-files checks."""

    def test_check_existing_files_found(self, tmp_path, monkeypatch):
        module = _reload_module()
        monkeypatch.chdir(tmp_path)

        models_dir = tmp_path / "models"
        images_dir = tmp_path / "images"
        gt_dir = images_dir / "test" / "gt"
        models_dir.mkdir()
        images_dir.mkdir()
        gt_dir.mkdir(parents=True)
        (models_dir / "best_exp20.pt").write_text("model")
        (images_dir / "sample.png").write_text("png")
        (gt_dir / "sample.json").write_text("json")

        model_exists, images_exist, gt_exists = module.check_existing_files()

        assert model_exists is True
        assert images_exist is True
        assert gt_exists is True

    def test_check_existing_files_missing(self, tmp_path, monkeypatch):
        module = _reload_module()
        monkeypatch.chdir(tmp_path)
        (tmp_path / "models").mkdir()
        (tmp_path / "images").mkdir()

        model_exists, images_exist, gt_exists = module.check_existing_files()

        assert model_exists is False
        assert images_exist is False
        assert gt_exists is False


class TestSuggestDownload:
    """Tests for suggestion behavior."""

    def test_suggest_download_false_when_all_present(self, monkeypatch):
        module = _reload_module()
        monkeypatch.setattr(module, "check_existing_files", lambda: (True, True, True))

        assert module.suggest_download() is False

    def test_suggest_download_true_when_missing(self, monkeypatch):
        module = _reload_module()
        monkeypatch.setattr(
            module, "check_existing_files", lambda: (True, False, False)
        )

        assert module.suggest_download() is True

    def test_suggest_download_accepts_legacy_two_value_status(self, monkeypatch):
        module = _reload_module()
        monkeypatch.setattr(module, "check_existing_files", lambda: (True, True))

        assert module.suggest_download() is False


class TestDownloadViaGdown:
    """Tests for gdown download flow."""

    def test_download_via_gdown_success(self, tmp_path, monkeypatch):
        module = _reload_module()
        monkeypatch.chdir(tmp_path)

        def fake_download_folder(url, output, quiet, use_cookies):
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            nested = output_path / "downloaded"
            images_test = nested / "images" / "test"
            images_gt = images_test / "gt"
            images_gt.mkdir(parents=True, exist_ok=True)
            (nested / "best_exp20.pt").write_text("model")
            (images_test / "img1.png").write_text("png1")
            (images_test / "img2.png").write_text("png2")
            (images_gt / "img1.json").write_text("{}")

        fake_gdown = types.SimpleNamespace(download_folder=fake_download_folder)
        fake_certifi = types.SimpleNamespace(where=lambda: "C:/tmp/cert.pem")
        monkeypatch.setitem(sys.modules, "gdown", fake_gdown)
        monkeypatch.setitem(sys.modules, "certifi", fake_certifi)

        ok = module.download_via_gdown()

        assert ok is True
        assert (tmp_path / "models" / "best_exp20.pt").exists()
        assert (tmp_path / "images" / "test" / "img1.png").exists()
        assert (tmp_path / "images" / "test" / "gt" / "img1.json").exists()

    def test_download_via_gdown_uses_original_folder_url(self, tmp_path, monkeypatch):
        module = _reload_module()
        monkeypatch.chdir(tmp_path)

        calls = {}

        def fake_download_folder(url, output, quiet, use_cookies, remaining_ok=False):
            calls["url"] = url
            output_path = Path(output)
            nested = output_path / "downloaded"
            (nested / "models").mkdir(parents=True, exist_ok=True)
            (nested / "images" / "test" / "gt").mkdir(parents=True, exist_ok=True)
            (nested / "models" / "best_exp20.pt").write_text("model")
            (nested / "images" / "test" / "img1.png").write_text("png1")

        fake_gdown = types.SimpleNamespace(download_folder=fake_download_folder)
        fake_certifi = types.SimpleNamespace(where=lambda: "C:/tmp/cert.pem")
        monkeypatch.setitem(sys.modules, "gdown", fake_gdown)
        monkeypatch.setitem(sys.modules, "certifi", fake_certifi)

        ok = module.download_via_gdown()

        assert ok is True
        assert calls["url"] == module.GDRIVE_FOLDER_URL

    def test_download_via_gdown_import_error(self, monkeypatch):
        module = _reload_module()
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "gdown":
                raise ImportError("missing gdown")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)

        ok = module.download_via_gdown()

        assert ok is False

    def test_download_via_gdown_ssl_error(self, tmp_path, monkeypatch):
        module = _reload_module()
        monkeypatch.chdir(tmp_path)

        def fake_download_folder(url, output, quiet, use_cookies):
            raise Exception("CERTIFICATE_VERIFY_FAILED")

        fake_gdown = types.SimpleNamespace(download_folder=fake_download_folder)
        monkeypatch.setitem(sys.modules, "gdown", fake_gdown)

        ok = module.download_via_gdown()

        assert ok is False


class TestMain:
    """Tests for main CLI routing."""

    def test_main_manual_route(self, monkeypatch):
        module = _reload_module()
        called = {"manual": False}

        monkeypatch.setattr(sys, "argv", ["download_test_data.py", "--manual"])
        monkeypatch.setattr(
            module, "check_existing_files", lambda: (False, False, False)
        )
        monkeypatch.setattr(
            module,
            "manual_download_instructions",
            lambda: called.__setitem__("manual", True),
        )

        module.main()

        assert called["manual"] is True

    def test_main_accepts_legacy_two_value_status(self, monkeypatch):
        module = _reload_module()
        monkeypatch.setattr(sys, "argv", ["download_test_data.py"])
        monkeypatch.setattr(module, "check_existing_files", lambda: (True, True))
        monkeypatch.setattr(module, "download_via_gdown", lambda: False)

        module.main()

    def test_main_download_failure_falls_back_manual(self, monkeypatch):
        module = _reload_module()
        called = {"manual": False}

        monkeypatch.setattr(sys, "argv", ["download_test_data.py"])
        monkeypatch.setattr(
            module, "check_existing_files", lambda: (False, False, False)
        )
        monkeypatch.setattr(module, "download_via_gdown", lambda: False)
        monkeypatch.setattr(
            module,
            "manual_download_instructions",
            lambda: called.__setitem__("manual", True),
        )

        module.main()

        assert called["manual"] is True
