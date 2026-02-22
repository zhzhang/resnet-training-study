import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, models
from torchvision import transforms
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Device configuration
def data_loader(data_dir, batch_size, shuffle=True, test=False):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if test:
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    return train_loader


@dataclass
class TrainingConfig:
    n_epochs: int = 100
    lr: float = 0.1
    batch_size: int = 250
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0001


def train_model(config: TrainingConfig):
    train_loader = data_loader(data_dir="./data", batch_size=config.batch_size)

    test_loader = data_loader(
        data_dir="./data", batch_size=config.batch_size, test=True
    )
    print(len(train_loader))
    print(len(test_loader))

    model = models.resnet18().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.n_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            print(outputs.grad)
            optimizer.step()
            print(
                f"Epoch [{epoch + 1}/{config.n_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )
        acc = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            acc += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            print(f"Validation Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch + 1}/{config.n_epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    config = TrainingConfig()
    train_model(config)
