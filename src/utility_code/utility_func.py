import yaml
from pathlib import Path
from box import ConfigBox

def read_params(config_path:Path):
    with open(config_path) as yaml_file:
        config_dict=yaml.safe_load(yaml_file)
        return ConfigBox(config_dict)
    
def main():
    config_path=Path(r"D:\End to end project\Learning_MLOps\params.yaml")
    config=read_params(config_path)
    print(config)

if __name__=="__main__":
    main()
        