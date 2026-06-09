"""stimulus library summarizer!

this reads a csv catalog of stimuli (i.e. images, audio clips, text passages) and
prints a summary of the library: counts by modality and category, audio
duration stats, and a tally of missing values.

need to run it with:
    python learning/stimulus_summary.py learning/sample_stimuli.csv
"""

import sys
from pathlib import Path

import pandas as pd


def load_catalog(path: Path) -> pd.DataFrame:
    """Load a stimulus CSV into a DataFrame, with friendly errors."""
    if not path.exists():
        raise FileNotFoundError(f"No CSV found at {path}")
    df = pd.read_csv(path)

    # Sanity-check that the expected columns are present.
    required = {"id", "filename", "modality", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    return df


def summarize_by_column(df: pd.DataFrame, column: str) -> dict[str, int]:
    """Return a count of how many rows fall into each value of `column`.

    Example: summarize_by_column(df, "modality") -> {"image": 52, "audio": 21, ...}
    """
    counts = df[column].value_counts()
    # TODO: convert the pandas Series 'counts' into a plain dict.
    # Hint: counts.to_dict() does it in one call.
    return counts.to_dict()


def audio_duration_stats(df: pd.DataFrame) -> dict[str, float] | None:
    """Return mean / median / min / max duration for audio rows, or None if no audio."""
    audio_rows = df[df["modality"] == "audio"]
    if audio_rows.empty:
        return None

    durations = audio_rows["duration_sec"].dropna()
    # TODO: build and return a dict with keys "mean", "median", "min", "max".
    # pandas Series have .mean(), .median(), .min(), .max() methods.
    durationdict = {
        "mean" : durations.mean(),
        "median" : durations.median(),
        "min" : durations.min(),
        "max" : durations.max(),
    }
    return durationdict


def missing_value_report(df: pd.DataFrame) -> dict[str, int]:
    """Count how many rows are missing a value in each column."""
    # This is a great place to practice a dict comprehension.
    # TODO: build {column_name: number_of_missing_values} for every column
    # that has at least one missing value.
    # Hint: df[col].isna().sum() gives the missing count for one column.
    #       df.columns is the list of column names.
    report = {
        col : df[col].isna().sum() for col in df.columns if df[col].isna().any()
    }
    return report


def print_summary(df: pd.DataFrame) -> None:
    """Print the whole summary report to the terminal."""
    print("Stimulus Library Summary")
    print("=" * 24)
    print(f"Total stimuli: {len(df)}")

    print("\nBy modality:")
    for modality, count in summarize_by_column(df, "modality").items():
        print(f"  {modality}: {count}")

    # TODO: print the same kind of breakdown for the "category" column.
    print("\nBy category:")
    for category, count in summarize_by_column(df, "category").items():
        print(f"  {category}: {count}")

    duration = audio_duration_stats(df)
    if duration is not None:
        print(
            f"\nAudio duration: mean {duration['mean']:.1f}s, "
            f"median {duration['median']:.1f}s, "
            f"range {duration['min']:.1f}–{duration['max']:.1f}s"
        )

    missing = missing_value_report(df)
    if missing:
        print("\nMissing values:")
        for column, count in missing.items():
            print(f"  {column}: {count} missing")


def main() -> None:
    # Expect one command-line argument: the path to the CSV.
    if len(sys.argv) != 2:
        print("Usage: python stimulus_summary.py <path-to-csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])

    try:
        df = load_catalog(csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print_summary(df)


if __name__ == "__main__":
    main()