import json
from pathlib import Path

import pandas as pd

from scripts import WRSapplication, WRSresults


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


def test_application_to_results_pipeline_generates_consistent_metrics(
    tmp_path: Path, monkeypatch
):
    image_path = tmp_path / "images" / "sample.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_text("img", encoding="utf-8")

    app_output = tmp_path / "app_out"
    monkeypatch.setattr(
        WRSapplication, "files_list_creator", lambda *args, **kwargs: [str(image_path)]
    )
    monkeypatch.setattr(
        WRSapplication, "test_model", lambda *args, **kwargs: [_Result([1])]
    )
    monkeypatch.setattr(WRSapplication, "paint_results", lambda *args, **kwargs: None)

    def _fake_save_jsons(results, save_path, merge_iou):
        payload = [
            {
                "class": "w",
                "confidence": 0.95,
                "bbox": {"xmin": 10, "ymin": 100, "xmax": 20, "ymax": 200},
            }
        ]
        out = Path(save_path) / "sample.json"
        out.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(WRSapplication, "save_jsons", _fake_save_jsons)
    monkeypatch.setattr(
        "sys.argv",
        [
            "WRSapplication.py",
            "--data_folder",
            str(image_path.parent),
            "--output_results",
            str(app_output),
        ],
    )
    WRSapplication.run()

    results_output = tmp_path / "results_out"
    monkeypatch.setattr(
        WRSresults, "load_config", lambda *_: (0.01, 100.0, 1000.0, 448)
    )
    monkeypatch.setattr(WRSresults, "plot_WRSresults", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "WRSresults.py",
            "--data_folder",
            str(app_output),
            "--output_results",
            str(results_output),
            "--output_name",
            "pipeline",
        ],
    )
    WRSresults.run()

    out_csv = results_output / "pipeline.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv, sep=";", comment="#")
    assert len(df) == 1
    assert df.loc[0, "Tdur"] == 0.1
    assert df.loc[0, "Fdur"] == 10000.0
