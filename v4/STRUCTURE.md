# v3 file structure

This document summarizes the layout of `v3/` and the role of each major folder.

## Top-level

- `data/` - Input datasets used by the v3 pipeline.
- `outputs/` - Generated solution artifacts, logs, and plots from runs.
- `run_all_instance.slurm` - Slurm batch script to run multiple instances.
- `run_single_instance.slurm` - Slurm batch script to run a single instance.
- `src/` - Source code for data loading, modeling, heuristics, and analysis.
- `tests/` - Unit tests.

## data/

- `data/raw/` - Raw source files (Solomon benchmarks, spreadsheets).
- `data/cleaned/` - Cleaned/processed inputs ready for experiments.

## outputs/

- `outputs/` - Results for default runs (JSON/PKL outputs).
- `outputs/v3-12h/` - Outputs for the 12-hour experiment configuration.
- `outputs/v3-12h-balance-0.8-1.2/` - Outputs and logs for the balance-hours runs.

## src/

- `src/analysis/` - Post-run parsing and analysis scripts.
- `src/experiments/` - Runners for single or batch experiment execution.
- `src/heuristics/` - Heuristic construction, selection, and improvement logic.
- `src/io/` - Data loading and dataset conversion utilities.
- `src/solutions/` - Solution serialization and helper utilities.
- `src/solver/` - Core optimization model, constraints, objectives, and runner.
- `src/structures/` - Core data structures (e.g., problem data definitions).

## tests/

- `tests/unit/` - Unit tests for core data structures and logic.
