import yaml

def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    
    return config

if __name__ == "__main__":
    path = "./config/clip.yaml"
    print(load_config(path))