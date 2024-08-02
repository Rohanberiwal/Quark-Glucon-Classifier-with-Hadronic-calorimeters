import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class VGG12(nn.Module):
    def __init__(self, input_size=(125, 125)):
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
        conv_output_size = self._get_conv_output_size(input_size)
        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2)  # Assuming binary classification
        )

    def _get_conv_output_size(self, input_size):
        dummy_input = torch.randn(1, 3, *input_size)
        output = self.features(dummy_input)
        return output.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def generate_sample_data(num_samples=1000, image_size=(125, 125)):
    def generate_image():
        image = np.random.rand(image_size[0], image_size[1], 3) * 255
        image = image.astype(np.uint8)
        return image

    quark_list = []
    gluon_list = []

    for _ in range(num_samples):
        quark_image_array = generate_image()
        quark_list.append([quark_image_array, 1])
        
        gluon_image_array = generate_image()
        gluon_list.append([gluon_image_array, 0])

    return quark_list, gluon_list

def to_tensor(data_list):
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((125, 125)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images = []
    labels = []
    for image_array, label in data_list:
        image = transform(image_array)
        images.append(image)
        labels.append(torch.tensor(label, dtype=torch.long))
    return images, labels

class ParticleDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def split_dataset(dataset, train_ratio=0.8):
    split_idx = int(len(dataset) * train_ratio)
    train_data = Subset(dataset, range(split_idx))
    test_data = Subset(dataset, range(split_idx, len(dataset)))
    return train_data, test_data

def l1_l2_regularization(model, l1_lambda=0.0, l2_lambda=0.0):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return l1_lambda * l1_norm + l2_lambda * l2_norm

def ensemble_predict(models, dataloader):
    all_preds = []
    with torch.no_grad():
        for images, _ in dataloader:
            outputs = torch.mean(torch.stack([model(images) for model in models]), dim=0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

def main():
    # Generate sample data
    quark_list, gluon_list = generate_sample_data()

    # Convert to tensors
    quark_images, quark_labels = to_tensor(quark_list)
    gluon_images, gluon_labels = to_tensor(gluon_list)

    # Create datasets
    quark_dataset = ParticleDataset(quark_images, quark_labels)
    gluon_dataset = ParticleDataset(gluon_images, gluon_labels)

    # Split datasets
    quark_train, quark_test = split_dataset(quark_dataset)
    gluon_train, gluon_test = split_dataset(gluon_dataset)

    # Combine datasets
    train_dataset = ConcatDataset([quark_train, gluon_train])
    test_dataset = ConcatDataset([quark_test, gluon_test])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize models for ensemble
    num_models = 3
    models = [VGG12() for _ in range(num_models)]
    criterions = [nn.CrossEntropyLoss() for _ in range(num_models)]
    optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

    num_epochs = 50
    for epoch in range(num_epochs):
        for model, criterion, optimizer in zip(models, criterions, optimizers):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss += l1_l2_regularization(model, l1_lambda=0.001, l2_lambda=0.001)  # Add regularization
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Calculate training loss
            train_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Calculate validation loss and accuracy
            val_loss /= len(test_loader)
            val_accuracy = 100 * correct / total

            # Print metrics
            print(f"Model Epoch {epoch+1}/{num_epochs}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Evaluate ensemble
    ensemble_preds = ensemble_predict(models, test_loader)
    correct = (ensemble_preds == np.concatenate([labels.numpy() for _, labels in test_loader])).sum()
    accuracy = 100 * correct / len(test_dataset)

    print(f"Ensemble Validation Accuracy: {accuracy:.2f}%")

    print("Training complete.")
    print("THIS CODE END HERE BUT WITH MODIFICATION")

if __name__ == "__main__":
    main()

