from torch.utils.data import Dataset, DataLoader

class DatasetPro(Dataset):
    def __init__(self,folder,transform=None):
        self.transform = transform
        pass
    def __len__(self):
        pass
    def __getitem__(self,i):
        pass
