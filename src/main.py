from pathlib import Path
import json

from .parser import create_parser


def validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise ValueError(f"Input file not found: {input_path}")

    if input_path.is_dir():
        raise ValueError(f"Input path is a directory: {input_path}")

    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"Output path cannot be a directory: {output_path}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    try:
        validate_paths(args.input, args.output)
        content = args.input.read_text(encoding="utf-8")
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    except OSError as e:
        print(f"[SYSTEM ERROR] {e}")
        return

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print("Invalid JSON")
        return

    # args.output.parent.mkdir(parents=True, exist_ok=True)
    # args.output.write_text(result, encoding="utf-8")
    return


args.input.exists()
args.input.is_file()

with args.input.open() as file:
    ...


with args.output.open("w") as file:
    ...

if __name__ == "__main__":
    main()
