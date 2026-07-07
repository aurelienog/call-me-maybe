from pathlib import Path


def validate_existing_file(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Input file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")


def validate_output_file(path: Path) -> None:
    if path.exists() and path.is_dir():
        raise ValueError(f"Output path cannot be a directory: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
