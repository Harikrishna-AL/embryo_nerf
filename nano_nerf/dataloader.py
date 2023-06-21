from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from glob import glob
from torchvision import transforms
import numpy as np
import re

file = open("./sphere_data.json")
data = json.load(file)


class DatasetPro(Dataset):
    def __init__(self, train_path, transform_img=None, transform_matrix=None):
        self.transform_img = transform_img
        self.transform_matrix = transform_matrix
        self.directory = train_path
        # self.json_filename = json_filename

    def __len__(self):
        return len(self.directory)

    def __getitem__(self, i):
        image_path = self.directory[i]
        json_index = re.findall("[0-9]+", image_path)
        json_index = int(json_index[0])
        trans_matrix = np.array(data[json_index])
        img = plt.imread(image_path)
        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_matrix:
            trans_matrix = self.transform_matrix(trans_matrix)
        return img, trans_matrix


transform_img = transforms.Compose(
    [
        # transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Resize((100, 100)),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)

transform_matrix = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def show_image(image):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.detach().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)


BATCH_SIZE = 1
train_path = glob("./generated_views/*.png")
train_data = DatasetPro(
    train_path, transform_img=transform_img, transform_matrix=transform_matrix
)
dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
