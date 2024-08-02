import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np 

class ParquetDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract matrix data and label
        matrix_data = self.data.iloc[idx]['X_jets']  # 'X_jets' is the matrix data
        label = int(self.data.iloc[idx]['y'])  # 'y' is the label

        # Convert matrix to tensor
        matrix = np.array(matrix_data, dtype=np.float32)  # Ensure numpy array conversion
        matrix = torch.tensor(matrix, dtype=torch.float32)

        # Ensure the tensor is of shape (C, H, W)
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)  # Add a channel dimension
        elif matrix.ndim == 3:
            matrix = matrix.permute(2, 0, 1)  # From (H, W, C) to (C, H, W)

        # Apply transformations if provided
        if self.transform:
            matrix = self.transform(matrix)

        return matrix, label

def read_and_split_parquet_file(file_path):
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

def load_data_into_resnet152(train_data, val_data, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ParquetDataset(train_data, transform=transform)
    val_dataset = ParquetDataset(val_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load pre-trained ResNet-152 model
    model = models.resnet152(pretrained=True)
    
    # Modify the final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, train_loader, val_loader, criterion, optimizer

# Example usage
file_path = r"C:\Users\rohan\OneDrive\Desktop\Quark_dataset\QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272.test.snappy.parquet"
train_data, val_data = read_and_split_parquet_file(file_path)
if train_data is not None and val_data is not None:
    model, train_loader, val_loader, criterion, optimizer = load_data_into_resnet152(train_data, val_data)
    print("Data loaded into ResNet-152 model.")
print("End of file processing")
