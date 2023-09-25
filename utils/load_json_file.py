import json
import pandas as pd

def open_json_file(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

if __name__ == "__main__":
    path = "./test_data.json"
    print(len(open_json_file(path).image_path))