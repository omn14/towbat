
import json
def store_dict_to_file(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_dict_from_file(filename: str) -> dict:
    with open(filename, 'r') as f:
        return json.load(f)