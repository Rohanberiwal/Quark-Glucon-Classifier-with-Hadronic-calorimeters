import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Define custom dataset
class ParquetDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx]['image']  # Assuming 'image' column contains image data
        label = self.data.iloc[idx]['label']  # Assuming 'label' column contains labels

        if self.transform:
            image = self.transform(image)

        return image, label

# VGG-like model with 12 layers
class VGG12(nn.Module):
    def __init__(self):
        super(VGG12, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2)  # Assuming binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def read_and_split_parquet_file(file_path):
    """
    Reads the contents of a .parquet file, splits the data into 80% training and 20% validation,
    and prints the output along with the counts of the training and validation sets.

    Parameters:
    file_path (str): The path to the .parquet file.
    """
    # Check if the provided file path exists
    if not os.path.isfile(file_path):
        print(f"File '{file_path}' does not exist.")
        return

    # Check if the file has a .parquet extension
    if not file_path.endswith('.parquet'):
        print(f"File '{file_path}' is not a .parquet file.")
        return

    try:
        # Open the .parquet file and read its contents
        contents = pd.read_parquet(file_path)
        print(f"Contents of {file_path}:")
        print(contents.head())
        print("-" * 40)  # Separator line for clarity

        # Split the data into 80% training and 20% validation
        train_data, val_data = train_test_split(contents, test_size=0.2, random_state=42)
        
        # Print the number of records in the training and validation sets
        print(f"Number of records in the training set: {len(train_data)}")
        print(f"Number of records in the validation set: {len(val_data)}")

        return train_data, val_data

    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None, None

def load_data_into_vgg12(train_data, val_data, batch_size=32):
    """
    Loads the training and validation data into VGG12 model.

    Parameters:
    train_data (DataFrame): The training data.
    val_data (DataFrame): The validation data.
    batch_size (int): The batch size for DataLoader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ParquetDataset(train_data, transform=transform)
    val_dataset = ParquetDataset(val_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = VGG12()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, train_loader, val_loader, criterion, optimizer

# Example usage
file_path = r"C:\Users\rohan\OneDrive\Desktop\Quark_dataset\QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272.test.snappy.parquet"
train_data, val_data = read_and_split_parquet_file(file_path)
if train_data is not None and val_data is not None:
    model, train_loader, val_loader, criterion, optimizer = load_data_into_vgg12(train_data, val_data)
    print("Data loaded into VGG12 model.")
print("End of file processing")

