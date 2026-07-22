import json
from pydantic import ValidationError  # type: ignore

from .utils import (
    validate_existing_file,
    validate_output_file,
    load_json,
    save_json,
)
from .parser import parse
from .models import FunctionRegistry, FunctionDefinition, Prompt
from .decoder import ConstrainedDecoder
from .llm import Llm


def main() -> None:
    args = parse()
    registry = FunctionRegistry()

    try:
        validate_existing_file(args.functions_definition)
        validate_existing_file(args.input)
        validate_output_file(args.output)

        functions_data = load_json(args.functions_definition)
        functions = FunctionDefinition.validate_many(functions_data)
        registry.load(functions)

        # 1. Instanciamos la LLM (carga el modelo y el vocabulario)
        print("[INFO] Cargando modelo y vocabulario...")
        llm = Llm()

        # 2. Pasamos la LLM y el Registry al Decoder
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
        # 3. Procesamos los prompts con feedback visual en terminal
        print(f"[INFO] Procesando {len(prompts)} prompts...")
        results = decoder.run(prompts)

        # 4. Guardamos los resultados
        save_json(
            args.output,
            [result.model_dump() for result in results],
        )
        print(f"[SUCCESS] Resultados guardados exitosamente en: {args.output}")

    except json.JSONDecodeError as e:
        print(f"[INVALID JSON] {e}")

    except OSError as e:
        print(f"[SYSTEM ERROR] {e}")

    except Exception as e:
        print(f"[PROCESS ERROR] {e}")


if __name__ == "__main__":
    main()
