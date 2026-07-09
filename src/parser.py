import argparse
from pathlib import Path

from .config import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_FUNCTIONS
)


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add configuration-related command-line arguments.
    """

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help="Path to the functions definition JSON file.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the input JSON file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output JSON file.",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Function calling with Qwen 0.6B"
    )
    add_config_arguments(parser)
    return parser


def parse(arguments: list[str] | None = None):
    parser = create_parser()
    return parser.parse_args(arguments)
