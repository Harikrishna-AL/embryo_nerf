from torch.utils.data import Dataset, DataLoader
from glob import glob
import json 
import matplotlib.pyplot as plt

file = open("./sphere_data.json")
data = json.load(file)
train_path = glob("./generated_views/*.png")
class DatasetPro(Dataset):
    def __init__(self,train_path,json_filename,transform_img=None,transform_matrix=None):
        self.transform_img = transform_img
        self.transform_matrix = transform_matrix
        self.directory = train_path
        self.json_filename = json_filename
    def __len__(self):
        len(self.directory)
    def __getitem__(self,i):
        image_path = self.directory[i]
        json_index = image_path[-5]
        trans_matrix = data[json_index]
        img = plt.imread(image_path)
        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_matrix:
            trans_matrix = self.transform_matrix(img)
        return img, trans_matrix
