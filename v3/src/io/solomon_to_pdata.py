from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def read_solomon_file(raw_dir: Path, file_name: str) -> pd.DataFrame:
    file_path = raw_dir / "Solomon_25" / f"{file_name}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing Solomon file at {file_path}")

    lines = file_path.read_text().splitlines()
    header_index = next(
        i for i, line in enumerate(lines) if line.strip().startswith("CUST NO.")
    )

    data_lines = lines[header_index + 2 :]
    data = [
        [int(part) for part in line.split()] for line in data_lines if line.strip()
    ]

    df = pd.DataFrame(
        data, columns=["Index", "X", "Y", "Demand", "Start", "End", "Duration"]
    ).set_index("Index")
    return df[["X", "Y"]].copy()


def assign_event_types(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    df = df.copy()
    if len(df) > 1:
        df["Event"] = 0
        df.loc[1:, "Event"] = np.random.choice(
            np.arange(1, 101), size=len(df) - 1, replace=False
        )
    else:
        df["Event"] = 0
    return df


def euclidean_distance(row1: pd.Series, row2: pd.Series) -> float:
    return np.sqrt((row1["X"] - row2["X"]) ** 2 + (row1["Y"] - row2["Y"]) ** 2)


def create_distance_matrix(
    df: pd.DataFrame, condition: Callable[[pd.Series, pd.Series], bool]
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=df.index, columns=df.index, dtype=float)
    for i in df.index:
        for j in df.index:
            if condition(df.loc[i], df.loc[j]):
                matrix.loc[i, j] = euclidean_distance(df.loc[i], df.loc[j])
    matrix.dropna(how="all", inplace=True)
    matrix.dropna(axis=1, how="all", inplace=True)
    return matrix.astype(int)


def build_event_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return create_distance_matrix(
        df,
        lambda r1, r2: r1["Event"] <= 50
        and r2["Event"] <= 50
        and r1["Event"] != 0
        and r2["Event"] != 0,
    )


def build_home_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return create_distance_matrix(
        df,
        lambda r1, r2: r1["Event"] > 50
        and r2["Event"] <= 50
        and r2["Event"] != 0,
    )


def build_depot_matrix(df: pd.DataFrame, subset_index: pd.Index) -> pd.DataFrame:
    depot_candidates = df.index[df["Event"] == 0]
    if depot_candidates.empty:
        raise ValueError("No depot found (Event == 0).")
    depot_idx = depot_candidates[0]

    depot = pd.DataFrame(index=subset_index, columns=[0], dtype=float)
    for i in subset_index:
        if i != depot_idx:
            depot.loc[i, 0] = euclidean_distance(df.loc[i], df.loc[depot_idx])

    depot.dropna(how="all", inplace=True)
    depot.dropna(axis=1, how="all", inplace=True)
    return depot.astype(int)


def sample_durations_and_nurses(raw_dir: Path, event_index: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_data = pd.read_excel(raw_dir / "real-nurse-dur.xlsx", index_col=0)
    sampled_data = real_data.sample(
        n=len(event_index), replace=False, random_state=42
    )

    base = pd.DataFrame(index=event_index)
    c_dur = base.copy()
    c_dur["Duration"] = sampled_data["Duration"].values

    min_nurse = base.copy()
    min_nurse["RN"] = sampled_data["RN"].values
    min_nurse["LVN"] = sampled_data["LVN"].values

    return c_dur, min_nurse


def build_time_windows(raw_dir: Path, tw_type: str, event_index: pd.Index) -> pd.DataFrame:
    time_windows_raw = pd.read_excel(
        raw_dir / "five_time_windows.xlsx", sheet_name=tw_type, index_col=0
    )
    expected_cols = [
        "Start_1",
        "End_1",
        "Start_2",
        "End_2",
        "Start_3",
        "End_3",
        "Start_4",
        "End_4",
        "Start_5",
        "End_5",
    ]
    missing = [c for c in expected_cols if c not in time_windows_raw.columns]
    if missing:
        raise ValueError(f"Missing time window columns: {missing}")

    time_windows = pd.DataFrame(index=event_index, columns=expected_cols)
    # time_windows.loc[event_index, expected_cols] = (
    #     time_windows_raw.reindex(event_index)[expected_cols].values
    # )
    time_windows[expected_cols] = time_windows_raw[expected_cols].values

    return time_windows


def write_output(
    output_dir: Path,
    file_name: str,
    settings: pd.DataFrame,
    c_event: pd.DataFrame,
    c_home: pd.DataFrame,
    c_depot_e: pd.DataFrame,
    c_depot_h: pd.DataFrame,
    c_dur: pd.DataFrame,
    time_windows: pd.DataFrame,
    min_nurse: pd.DataFrame,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_name}.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        settings.to_excel(writer, sheet_name="Settings", index=False)
        c_event.to_excel(writer, sheet_name="C_event")
        c_home.to_excel(writer, sheet_name="C_home")
        c_depot_e.to_excel(writer, sheet_name="C_depot_e")
        c_depot_h.to_excel(writer, sheet_name="C_depot_h")
        c_dur.to_excel(writer, sheet_name="C_dur")
        time_windows.to_excel(writer, sheet_name="Time_Windows")
        min_nurse.to_excel(writer, sheet_name="Min_Nurses")
    return output_path


def main(file_name: str, type: str) -> Path:
    """
    Convert a Solomon instance into the pdata Excel format.
    type: sheet name in five_time_windows.xlsx (e.g., 'Even').
    """
    base_dir = Path(__file__).resolve().parents[2]  # .../mvt-code/v3
    raw_dir = base_dir / "data" / "raw"
    clean_dir = base_dir / "data" / "cleaned"

    df = read_solomon_file(raw_dir, file_name)
    df = assign_event_types(df)

    c_event = build_event_matrix(df)
    c_home = build_home_matrix(df)

    c_depot_e = build_depot_matrix(df, c_event.index)
    c_depot_h = build_depot_matrix(df, c_home.index)

    event_index = c_event.index
    c_dur, min_nurse = sample_durations_and_nurses(raw_dir, event_index)
    time_windows = build_time_windows(raw_dir, type, event_index)

    settings = pd.DataFrame(
        {"Parameter": ["nr", "nl", "m", "day"], "Value": [20, 30, len(event_index), 5]}
    )

    output_name = f"{file_name}_{type}"
    return write_output(
        clean_dir,
        output_name,
        settings,
        c_event,
        c_home,
        c_depot_e,
        c_depot_h,
        c_dur,
        time_windows,
        min_nurse,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Solomon instance to pdata Excel.")
    parser.add_argument("file_name", help="Solomon instance name without extension (e.g., c101).")
    parser.add_argument(
        "--type",
        dest="tw_type",
        default="Even",
        help="Time window sheet name in five_time_windows.xlsx (default: Even).",
    )

    args = parser.parse_args()
    output_path = main(args.file_name, args.tw_type)
    print(f"Wrote {output_path}")
