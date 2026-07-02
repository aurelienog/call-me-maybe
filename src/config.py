from pathlib import Path

DATA_DIR = Path("data")

INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

DEFAULT_FUNCTIONS = INPUT_DIR / "functions_definition.json"
DEFAULT_INPUT = INPUT_DIR / "function_calling_tests.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "function_calls.json"
