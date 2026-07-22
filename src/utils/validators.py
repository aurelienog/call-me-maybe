from pathlib import Path


def validate_existing_file(path: Path) -> None:
    """
    Validate that a path exists and refers to a file.

    Args:
        path: Path to validate.

    Raises:
        ValueError: If the path does not exist or does not
            refer to a regular file.
    """
    if not path.exists():
        raise ValueError(f"Input file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")


def validate_output_file(path: Path) -> None:
    """
    Validate an output file path.

    Ensures that the output path does not refer to an
    existing directory and creates the parent directories
    if they do not already exist.

    Args:
        path: Output file path.

    Raises:
        ValueError: If the output path refers to an existing
            directory.
    """
    if path.exists() and path.is_dir():
        raise ValueError(f"Output path cannot be a directory: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
