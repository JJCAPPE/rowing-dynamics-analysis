#!/usr/bin/env python3
"""Clean RP3 stroke exports by expanding curve_data into force-vs-length columns.

Usage:
    python expand_rp3_curve_data.py workouts/202602091949-rp3-row.csv
    python expand_rp3_curve_data.py input.csv -o output.csv --max-stroke-length-cm 170
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

DROP_COLUMNS = {
    "avg_power",
    "energy_sum",
    "pulse",
    "work_per_pulse",
    "ref",
    "workout_interval_id",
    "id",
    "k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop unused RP3 columns and expand curve_data into one force column "
            "per stroke-length step."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Input RP3 CSV file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path (default: <input-stem>-clean.csv)",
    )
    parser.add_argument(
        "--max-stroke-length-cm",
        type=float,
        default=170.0,
        help="Maximum stroke length to allocate columns for (default: 170)",
    )
    parser.add_argument(
        "--step-cm",
        type=float,
        default=2.2,
        help="Distance step represented by each curve_data value (default: 2.2)",
    )
    parser.add_argument(
        "--drop-curve-data",
        action="store_true",
        help="Remove original curve_data column after expansion",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help=(
            "If curve_data has more points than configured max columns, truncate "
            "extra points instead of failing"
        ),
    )
    return parser.parse_args()


def output_path(input_csv: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    if input_csv.parent.name == "dirty":
        return input_csv.parent.parent / "clean" / f"{input_csv.stem}-clean.csv"
    return input_csv.parent / "clean" / f"{input_csv.stem}-clean.csv"


def build_force_columns(max_stroke_length_cm: float, step_cm: float) -> list[str]:
    if max_stroke_length_cm <= 0:
        raise ValueError("max_stroke_length_cm must be > 0")
    if step_cm <= 0:
        raise ValueError("step_cm must be > 0")

    max_steps = math.ceil(max_stroke_length_cm / step_cm)
    columns = []
    for step in range(1, max_steps + 1):
        distance_cm = step * step_cm
        columns.append(f"force_at_{distance_cm:.1f}cm")
    return columns


def parse_curve_data(raw: str) -> list[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []

    return [value.strip() for value in raw.split(",") if value.strip()]


def base_columns(fieldnames: Iterable[str], drop_curve_data: bool) -> list[str]:
    columns: list[str] = []
    for name in fieldnames:
        if name in DROP_COLUMNS:
            continue
        if drop_curve_data and name == "curve_data":
            continue
        columns.append(name)
    return columns


def process_file(
    input_csv: Path,
    output_csv: Path,
    max_stroke_length_cm: float,
    step_cm: float,
    drop_curve_data: bool,
    truncate: bool,
) -> tuple[int, int]:
    force_columns = build_force_columns(max_stroke_length_cm, step_cm)

    with input_csv.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        if "curve_data" not in reader.fieldnames:
            raise ValueError("Input CSV is missing 'curve_data' column")

        keep_columns = base_columns(reader.fieldnames, drop_curve_data)
        out_columns = keep_columns + force_columns

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=out_columns)
            writer.writeheader()

            row_count = 0
            truncated_rows = 0
            max_points = len(force_columns)

            for row in reader:
                row_count += 1
                curve_points = parse_curve_data(row.get("curve_data", ""))

                if len(curve_points) > max_points:
                    if not truncate:
                        raise ValueError(
                            f"Row {row_count} has {len(curve_points)} curve points, "
                            f"but max configured points is {max_points}. "
                            "Use --truncate or increase --max-stroke-length-cm."
                        )
                    curve_points = curve_points[:max_points]
                    truncated_rows += 1

                out_row = {col: row.get(col, "") for col in keep_columns}
                for i, col in enumerate(force_columns):
                    out_row[col] = curve_points[i] if i < len(curve_points) else ""

                writer.writerow(out_row)

    return row_count, truncated_rows


def main() -> None:
    args = parse_args()

    input_csv = args.input_csv
    output_csv = output_path(input_csv, args.output)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows, truncated_rows = process_file(
        input_csv=input_csv,
        output_csv=output_csv,
        max_stroke_length_cm=args.max_stroke_length_cm,
        step_cm=args.step_cm,
        drop_curve_data=args.drop_curve_data,
        truncate=args.truncate,
    )

    force_cols = len(build_force_columns(args.max_stroke_length_cm, args.step_cm))
    print(f"Wrote {rows} rows to: {output_csv}")
    print(f"Force columns created: {force_cols}")
    if truncated_rows:
        print(f"Rows truncated due to max columns: {truncated_rows}")


if __name__ == "__main__":
    main()
