import csv
from pathlib import Path

from scripts.run_hybstex import FINAL_VARIANT, _write_filtered_csv, build_commands, parse_args


def test_build_commands_selects_final_submission_model(tmp_path):
    args = parse_args(
        [
            "--datasets", "NYCTaxi",
            "--seeds", "1",
            "--device", "cpu",
            "--data-dir", str(tmp_path / "data"),
            "--output-dir", str(tmp_path / "results"),
            "--dry-run",
        ]
    )
    base, final, output_root, final_output = build_commands(args)

    assert "A_base_mae" in base
    assert FINAL_VARIANT in final
    assert "train_all_node_flow_p90_valid_v2" in base
    assert "file_verified" in final
    assert output_root == (tmp_path / "results").resolve()
    assert final_output == output_root / "final"


def test_write_filtered_csv_keeps_only_final_variant(tmp_path):
    source = tmp_path / "source.csv"
    destination = tmp_path / "filtered.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "variant", "mae"))
        writer.writeheader()
        writer.writerows(
            [
                {"dataset": "NYCTaxi", "variant": "A_base_mae", "mae": "11"},
                {"dataset": "NYCTaxi", "variant": FINAL_VARIANT, "mae": "10"},
            ]
        )

    assert _write_filtered_csv(source, destination) == 1
    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"dataset": "NYCTaxi", "variant": FINAL_VARIANT, "mae": "10"}]
