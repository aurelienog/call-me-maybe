import sys
import json

try:
    from pydantic import ValidationError
except ImportError:
    print("❌[ERROR] Missing dependency: pydantic")
    print("Install it with: pip install pydantic")
    sys.exit(1)

try:
    from .llm import Llm
except ImportError:
    print("❌[ERROR] Failed to import Small_LLM_Model.")
    print("Make sure llm_sdk and its dependencies are installed:")
    print(" - numpy")
    print(" - pydantic")
    print(" - torch")
    print(" - transformers")
    print(" - huggingface-hub")
    sys.exit(1)


from .utils import (
    validate_existing_file,
    validate_output_file,
    load_json,
    save_json,
)
from .parser import parse
from .models import FunctionRegistry, FunctionDefinition, Prompt
from .decoder.decoder import ConstrainedDecoder


def main() -> None:
    """
    Run the function-calling pipeline.

    The program validates the input files, loads the function
    definitions into the registry, initializes the language
    model and constrained decoder, processes the input prompts,
    and writes the predicted function calls to the output JSON
    file.

    All expected validation, I/O, and JSON parsing errors are
    handled gracefully and reported to the user.
    """
    args = parse()
    registry = FunctionRegistry()

    try:
        validate_existing_file(args.functions_definition)
        validate_existing_file(args.input)
        validate_output_file(args.output)

        functions_data = load_json(args.functions_definition)
        functions = FunctionDefinition.validate_many(functions_data)
        registry.load(functions)

        print("[INFO] Loading model and vocabulary...")
        llm = Llm()

        decoder = ConstrainedDecoder(
            llm=llm,
            registry=registry,
        )

        prompts_data = load_json(args.input)
        prompts = Prompt.validate_many(prompts_data)

    except OSError as e:
        print(f"[SYSTEM ERROR] {e}")
        return

    except json.JSONDecodeError as e:
        print(f"[INVALID JSON] {e}")
        return

    except ValidationError as e:
        print("[VALIDATION ERROR]")
        print("The input JSON does not match the expected schema.")
        print(e)
        return

    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    try:
        print(f"[INFO] Processing {len(prompts)} prompts...")
        results = decoder.run(prompts)

        save_json(
            args.output,
            [result.model_dump() for result in results],
        )
        print(f"[SUCCESS]Results successfully saved to: {args.output}")

    except json.JSONDecodeError as e:
        print(f"[INVALID JSON] {e}")

    except OSError as e:
        print(f"[SYSTEM ERROR] {e}")

    except Exception as e:
        print(f"[PROCESS ERROR] {e}")


if __name__ == "__main__":
    main()
