import yaml

def read_config(file_path):
    """Reads a YAML configuration file and returns its contents as a dictionary."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config