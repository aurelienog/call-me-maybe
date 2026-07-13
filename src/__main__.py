import json
from pydantic import ValidationError    # type: ignore

from .utils import (validate_existing_file, validate_output_file,
                    load_json, save_json)
from .parser import parse
from .models import FunctionRegistry, FunctionDefinition, Prompt
from .decoder.constrained_decoder import ConstrainedDecoder


def main() -> None:
    args = parse()
    registry = FunctionRegistry()
    decoder = ConstrainedDecoder()
    decoder.registry = registry

    try:
        validate_existing_file(args.functions_definition)
        validate_existing_file(args.input)
        validate_output_file(args.output)

        functions_data = load_json(args.functions_definition)
        functions = FunctionDefinition.validate_many(functions_data)
        registry.load(functions)

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
        results = decoder.run(prompts)
        save_json(args.output, results)

    except json.JSONDecodeError as e:
        print(f"[INVALID JSON] {e}")
        return

    except OSError as e:
        print(f"[SYSTEM ERROR] {e}")

    except Exception as e:
        print(f"[PROCESS ERROR] {e}")
    # for function in functions:
    #     print(f"  - {function.name}")
    print("ok")


if __name__ == "__main__":
    main()

    # print(f"\nLoaded {len(prompts)} prompts:")
    # for p in prompts:
    #     print(f"  - {p.prompt}")
# c
