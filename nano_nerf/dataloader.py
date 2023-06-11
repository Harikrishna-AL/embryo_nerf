from torch.utils.data import Dataset, DataLoader
from glob import glob

train_path = glob("./generated_views/*.png")
class DatasetPro(Dataset):
    def __init__(self,train_path,json_filename,transform=None):
        self.transform = transform
        self.directory = train_path
        self.json_filename = json_filename
    def __len__(self):
        len(self.directory)
    def __getitem__(self,i):
        image = self.directory[i]
        return image
