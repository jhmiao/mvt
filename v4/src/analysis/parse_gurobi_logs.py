import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _log_dir() -> Path:
    return _project_root() / "outputs" / "v4_exact_weekly_fairness_w10_l100_mad" / "gurobi_logs"


def _parse_time(token: str) -> Optional[float]:
    token = token.strip()
    if not token or token == "-":
        return None
    if token.endswith("s"):
        try:
            return float(token[:-1])
        except ValueError:
            return None
    if ":" in token:
        parts = token.split(":")
        try:
            parts_f = [float(p) for p in parts]
        except ValueError:
            return None
        if len(parts_f) == 3:
            h, m, s = parts_f
            return h * 3600 + m * 60 + s
        if len(parts_f) == 2:
            m, s = parts_f
            return m * 60 + s
    try:
        return float(token)
    except ValueError:
        return None


def _parse_value(token: str) -> Optional[float]:
    token = token.strip()
    if token in {"-", ""}:
        return None
    token = token.replace("%", "")
    try:
        return float(token)
    except ValueError:
        return None


def _format_integer_value(token: str) -> str:
    """Format a Gurobi objective/bound token as an integer when possible."""
    value = _parse_value(token)
    return str(int(round(value))) if value is not None else token


def parse_parameters(lines: List[str]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    pattern = re.compile(r"Changed value of (\S+) .* to ([^\s]+)")
    for ln in lines:
        m = pattern.search(ln)
        if m:
            params[m.group(1)] = m.group(2)
    return params


def parse_log_file(path: Path) -> Tuple[
    Dict[str, str],
    List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]],
    Optional[str],
]:
    """
    Return (parameters, rows, title) where rows are tuples of
    (incumbent, bound, gap_percent, time_seconds) and title is derived from footer.
    """
    lines = path.read_text().splitlines()
    params = parse_parameters(lines)

    # Extract title info from the solver summary near the end of the log.
    last_lines = lines[-3:] if len(lines) >= 3 else lines
    instance_name = None
    footer_result = None
    if last_lines:
        instance_candidate = last_lines[-1].strip().split()
        if instance_candidate:
            instance_name = instance_candidate[-1]
    summary_pattern = re.compile(
        r"Best objective\s+(\S+),\s+best bound\s+(\S+),\s+gap\s+(.+)"
    )
    for line in reversed(lines):
        match = summary_pattern.search(line)
        if match:
            footer_result = (
                f"Best objective {_format_integer_value(match.group(1))}, "
                f"best bound {_format_integer_value(match.group(2))}, "
                f"gap {match.group(3)}"
            )
            break
    title = None
    if instance_name and footer_result:
        title = f"{instance_name}: {footer_result}"

    # Some failed runs never reach the MIP progress table.  Do not use the
    # table heading as a marker: the useful part of these logs starts at the
    # first progress row, which is preceded by a blank line and starts with
    # the two node counters "0 0".
    first_progress_row = re.compile(r"^\s*0\s+0\s+")
    progress_row = re.compile(r"^\s*(?:[H*]\s+)?\d+\s+\d+\s+")
    start_index = next(
        (index for index, line in enumerate(lines) if first_progress_row.match(line)),
        None,
    )

    if start_index is None:
        return params, [], None

    rows: List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = []

    for ln in lines[start_index:]:
        if not progress_row.match(ln):
            continue

        tokens = ln.strip().split()
        # The final five columns are always Incumbent, BestBd, Gap, It/Node,
        # and Time, even for heuristic (H) rows with omitted node fields.
        if len(tokens) < 5:
            continue

        last_five = tokens[-5:]
        time_val = _parse_time(last_five[-1])
        if time_val is None:
            continue

        incumbent = _parse_value(last_five[0])
        bound = _parse_value(last_five[1])
        gap = _parse_value(last_five[2])

        rows.append((incumbent, bound, gap, time_val))

    return params, rows, title


def plot_progress(
    rows: List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]],
    title: str,
    out_path: Path,
) -> None:
    inc_points = [(t, inc) for inc, _, _, t in rows if inc is not None and t is not None]
    bd_points = [(t, bd) for _, bd, _, t in rows if bd is not None and t is not None]

    plt.figure(figsize=(8, 4))
    if inc_points:
        plt.plot([t for t, _ in inc_points], [v for _, v in inc_points], label="Incumbent")
    if bd_points:
        plt.plot([t for t, _ in bd_points], [v for _, v in bd_points], label="Best Bound")
    plt.xlabel("Time (s)")
    plt.ylabel("Objective")
    plt.ylim(0, 20_000)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    log_dir = _log_dir()

    for log_path in sorted(log_dir.glob("*.out")):
        params, rows, title = parse_log_file(log_path)
        if not rows:
            print(f"{log_path.name}: skipped (no Gurobi progress table)")
            continue
        out_file = log_dir / f"{log_path.stem}.png"
        plot_progress(rows, title=title or log_path.name, out_path=out_file)
        if params:
            print(f"{log_path.name}: params={params}")
        print(f"{log_path.name}: parsed {len(rows)} rows; plot -> {out_file}")


if __name__ == "__main__":
    main()
