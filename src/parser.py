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


def parse(arguments: list[str]| None = None):
    parser = create_parser()
    return parser.parse_args(arguments)


# Your solution will process two input files located in the data/input/ directory:
# • function_calling_tests.json: contains a JSON array of natural language prompts
# that your system must process.

# Example: function_calling_tests.json

# - functions_definition.json: contains the available functions your system can
# call. Each function includes:
# ◦ Function name
# ◦ Argument names and types
# ◦ Return type
# ◦ Description
# Example: functions_definition.json

# V.4.2 Validation Rules
# • The file must be valid JSON (no trailing commas, no comments)
# • Keys and types must match the schema in functions_definition.json exactly (input)
# • No extra keys or prose are allowed anywhere in the output
# • All required arguments must be present
# • Argument types must match the function definition (number, string, boolean, etc.)


