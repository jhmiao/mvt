from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
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


def perturb_coordinates(df: pd.DataFrame, std: float, seed: int = 42) -> pd.DataFrame:
    """
    Add Gaussian noise to event X/Y and round to integer coordinates.
    Only rows with 0 < Event <= 50 are perturbed.
    """
    if std <= 0:
        return df.copy()

    if "Event" not in df.columns:
        raise ValueError("Column 'Event' is required before perturbing coordinates.")

    rng = np.random.default_rng(seed)
    out = df.copy()
    event_mask = (out["Event"] > 0) & (out["Event"] <= 50)
    event_idx = out.index[event_mask]
    if len(event_idx) == 0:
        return out

    noise = rng.normal(loc=0.0, scale=std, size=(len(event_idx), 2))
    perturbed = np.rint(
        out.loc[event_idx, ["X", "Y"]].to_numpy(dtype=float) + noise
    ).astype(int)
    out.loc[event_idx, ["X", "Y"]] = perturbed
    return out


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


def sample_durations_and_nurses(raw_dir: Path, event_index: pd.Index, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_data = pd.read_excel(raw_dir / "real-nurse-dur.xlsx", index_col=0)
    sampled_data = real_data.sample(
        n=len(event_index), replace=False, random_state=seed
    )

    base = pd.DataFrame(index=event_index)
    c_dur = base.copy()
    c_dur["Duration"] = sampled_data["Duration"].values

    min_nurse = base.copy()
    min_nurse["RN"] = sampled_data["RN"].values
    min_nurse["LVN"] = sampled_data["LVN"].values

    return c_dur, min_nurse


def build_time_windows(
    raw_dir: Path,
    tw_type: str,
    event_index: pd.Index,
    seed: int = 42,
) -> pd.DataFrame:
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
    if len(time_windows_raw) < len(event_index):
        raise ValueError(
            f"Not enough rows in time window sheet '{tw_type}': "
            f"need {len(event_index)}, found {len(time_windows_raw)}."
        )

    shuffled_rows = (
        time_windows_raw[expected_cols]
        .sample(n=len(event_index), replace=False, random_state=seed)
        .to_numpy()
    )
    time_windows[expected_cols] = shuffled_rows

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


def write_location_plot(output_dir: Path, output_name: str, df: pd.DataFrame) -> Path:
    """
    Create and save a scatter plot for depot/events/homes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{output_name}_locations.png"

    depot_mask = df["Event"] == 0
    event_mask = (df["Event"] > 0) & (df["Event"] <= 50)
    home_mask = df["Event"] > 50

    plt.figure(figsize=(10, 6))
    plt.scatter(df.loc[event_mask, "X"], df.loc[event_mask, "Y"], c="tab:blue", alpha=0.8, label="Events")
    plt.scatter(df.loc[home_mask, "X"], df.loc[home_mask, "Y"], c="tab:orange", alpha=0.8, label="Homes")
    plt.scatter(
        df.loc[depot_mask, "X"],
        df.loc[depot_mask, "Y"],
        c="tab:red",
        alpha=1.0,
        marker="*",
        s=220,
        label="Depot",
    )
    plt.title("Location Scatter: Events, Homes, and Depot")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def main(
    file_name: str,
    type: str,
    coord_noise_std: float = 0.0,
    coord_noise_seed: int = 42,
) -> tuple[Path, Path]:
    """
    Convert a Solomon instance into the pdata Excel format.
    type: sheet name in five_time_windows.xlsx (e.g., 'Even').
    """
    base_dir = Path(__file__).resolve().parents[2]  # .../mvt-code/v3
    raw_dir = base_dir / "data" / "raw"
    clean_dir = base_dir / "data" / "cleaned" / "weeks"

    df = read_solomon_file(raw_dir, file_name)
    df = assign_event_types(df)
    df = perturb_coordinates(df, std=coord_noise_std, seed=coord_noise_seed)

    c_event = build_event_matrix(df)
    c_home = build_home_matrix(df)

    c_depot_e = build_depot_matrix(df, c_event.index)
    c_depot_h = build_depot_matrix(df, c_home.index)

    event_index = c_event.index
    c_dur, min_nurse = sample_durations_and_nurses(raw_dir, event_index, seed=coord_noise_seed)
    time_windows = build_time_windows(
        raw_dir,
        type,
        event_index,
        seed=coord_noise_seed,
    )

    settings = pd.DataFrame(
        {"Parameter": ["nr", "nl", "m", "day"], "Value": [20, 30, len(event_index), 5]}
    )

    std_token = f"{coord_noise_std:.1f}".replace(".", "p")
    output_name = f"{file_name}_{type}_{std_token}std_seed{coord_noise_seed}"
    xlsx_path = write_output(
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
    plot_path = write_location_plot(clean_dir, output_name, df)
    return xlsx_path, plot_path


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
    parser.add_argument(
        "--coord-noise-std",
        type=float,
        default=0.0,
        help="Std dev of Gaussian noise added to X/Y before rounding to int (default: 0.0).",
    )
    parser.add_argument(
        "--coord-noise-seed",
        type=int,
        default=42,
        help="Random seed used for coordinate perturbation (default: 42).",
    )

    args = parser.parse_args()
    output_path, plot_path = main(
        args.file_name,
        args.tw_type,
        coord_noise_std=args.coord_noise_std,
        coord_noise_seed=args.coord_noise_seed,
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {plot_path}")

# Example usage:
# python v3/src/io/solomon_to_pdata.py c101 --type Even --coord-noise-std 5.0 --coord-noise-seed 42
