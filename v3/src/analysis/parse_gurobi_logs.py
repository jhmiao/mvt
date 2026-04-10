import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _log_dir() -> Path:
    return _project_root() / "outputs" / "v3-12h-balance-0.8-1.2" / "log"


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

    # Extract title info from last lines
    last_lines = lines[-3:] if len(lines) >= 3 else lines
    instance_name = None
    footer_result = None
    if last_lines:
        instance_candidate = last_lines[-1].strip().split()
        if instance_candidate:
            instance_name = instance_candidate[-1]
        footer_result = last_lines[0].strip() if last_lines else None
    title = None
    if instance_name and footer_result:
        title = f"{instance_name}: {footer_result}"

    header_seen = False
    rows: List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = []

    for ln in lines:
        if not header_seen:
            # Header line variations: sometimes includes "Objective Bounds", sometimes not.
            if "Expl Unexpl" in ln and "Time" in ln:
                header_seen = True
            continue

        tokens = ln.strip().split()
        if len(tokens) < 5:
            continue

        if not tokens:
            break

        last_five = tokens[-5:]
        time_val = _parse_time(last_five[-1])
        if time_val is None:
            continue

        incumbent = _parse_value(last_five[0])
        bound = _parse_value(last_five[1])
        gap = _parse_value(last_five[2])

        rows.append((incumbent, bound, gap, time_val))
    # manually drop the last row
    rows = rows[:-1]

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
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    log_dir = _log_dir()
    plot_dir = log_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for log_path in sorted(log_dir.glob("*.out")):
        params, rows, title = parse_log_file(log_path)
        if not rows:
            print(f"{log_path.name}: no progress rows parsed")
            continue
        out_file = plot_dir / f"{log_path.stem}.png"
        plot_progress(rows, title=title or log_path.name, out_path=out_file)
        if params:
            print(f"{log_path.name}: params={params}")
        print(f"{log_path.name}: parsed {len(rows)} rows; plot -> {out_file}")


if __name__ == "__main__":
    main()
