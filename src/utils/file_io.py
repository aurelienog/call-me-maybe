from pathlib import Path
import json


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# load_vocab()

# load_prompt_file()

# save_results()

# For each prompt, add a JSON object to this file. Each object in the array
# must contain
# exactly the following keys:
# • prompt (string): The original natural-language request
# • name (string): The name of the function to call
# • parameters (object): All required arguments with the correct types
