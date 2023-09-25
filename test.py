from dataset import customDataset
from utils.basic_args import obtain_args
from utils.load_config import load_config
from utils.load_json_file import open_json_file
from dataset.customDataset import CustomDataset

args = obtain_args()
# print(args)
configs = load_config(args.config_paths)
# print(configs)

train_path = configs['global']['train_path']
test_path = configs['global']['test_path']
valid_path = configs['global']['valid_path']

data = open_json_file(test_path)

customData = CustomDataset(data, configs)
print(customData[1])