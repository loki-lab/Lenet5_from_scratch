from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn
from model import LeNet5
from torch.utils.data import DataLoader
from torch.optim import Adam
from trainer import Trainer
import torch

train_data_path = "datasets/Training"
validation_data_path = "datasets/Testing"
num_classes = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize((512, 512), antialias=True),
                                 transforms.Normalize((0.5,), (0.5,)),
                                 ])


train_dataset = ImageFolder(train_data_path, transform=transforms)
val_dataset = ImageFolder(validation_data_path, transform=transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = LeNet5(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

trainer = Trainer(model, criterion, optimizer, device)
trainer.fit(train_loader, val_loader, 20)
