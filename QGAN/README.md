# Quantum Generative Adversarial Networks (QGANs) to generate handwritten digits of zero

This notebook implements a quantum GANs using the patch method to generate handwritten digits of zero. The Pennylane quantum software framework and PyTorch are used to build the model.

## Installation

This code block installs the pennylane package using pip and upgrades it to the latest version. Pennylane is an open-source software for quantum machine learning, quantum computing, and quantum chemistry.
```bash
!pip install pennylane --upgrade
```

## Libraries

The following libraries are imported to run the code: math, random, numpy, pandas, matplotlib, pennylane, torch, torch.nn, torch.optim, torchvision.transforms, and torch.utils.data. These libraries are commonly used for scientific computing, machine learning, and data analysis in Python.
The following series of commands shows how to run the platform:
```bash
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
```
## Running the Code

To run the Quantum GAN code, follow these steps:

* Open Google Drive and create a new folder.
* Add the two required files, optdigits.tra and quantumGAN.ipynb, to the newly created folder.
* Right-click on the quantumGAN.ipynb file and select "Open with" -> "Google Colaboratory." This will open the notebook in Google Colaboratory.
* Follow the instructions provided in the notebook to execute the code.

Note that you will need a Google account to access Google Drive and Google Colaboratory.

## Dataset

The DigitsDataset class is defined next, which is a subclass of torch.utils.data.Dataset. The class reads an input CSV file containing images and their corresponding labels and initializes a dataset that can be used with PyTorch's DataLoader. The class also contains methods that allow filtering of the images based on the label.
```bash
class DigitsDataset(Dataset):
    def __init__(self, csv_file, label=0, transform=None):
        self.csv_file = csv_file
        self.transform = transform
        self.df = self.filter_by_label(label)

    def filter_by_label(self, label):
        # Use pandas to return a dataframe of only zeros
        df = pd.read_csv(self.csv_file)
        df = df.loc[df.iloc[:, -1] == label]
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.df.iloc[idx, :-1] / 16
        image = np.array(image)
        image = image.astype(np.float32).reshape(8, 8)

        if self.transform:
            image = self.transform(image)

        # Return image + label
        return image, 0
```
In this implementation, image_size is set to 8, and batch_size is set to 1. The transform is a composition of the ToTensor() function. The dataset is initialized with the DigitsDataset class, with the csv_file argument set to "optdigits.tra", and the transform argument set to transform. Finally, a DataLoader is created with the dataset, batch_size, shuffle, and drop_last arguments set.
```bash
#Height square images
image_size = 8  
#Width square images
batch_size = 1

transform = transforms.Compose([transforms.ToTensor()])
dataset = DigitsDataset(csv_file="optdigits.tra", transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
```
The code block below shows the first eight images from the dataset.
```bash
plt.figure(figsize=(8,2))

for i in range(8):
    image = dataset[i][0].
```