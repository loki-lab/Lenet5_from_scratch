from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn
from model import AlexNetTransfer
from torch.utils.data import DataLoader
from torch.optim import Adam
from trainer import Trainer
import torch

train_data_path = "datasets/Training"
validation_data_path = "datasets/Testing"
num_classes = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

transforms = {"training": transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((512, 512), antialias=True),
                                              transforms.Normalize((0.5,), (0.5,)),
                                              transforms.RandomPerspective(distortion_scale=0.5, p=0.1),
                                              transforms.RandomRotation(degrees=(0, 180)),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5),
                                              transforms.RandomAffine(degrees=(30, 70),
                                                                      translate=(0.1, 0.3),
                                                                      scale=(0.5, 0.75)),
                                              transforms.RandomInvert()
                                              ]),
              "validation": transforms.Compose([transforms.Resize((512, 512), antialias=True),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))])
              }

train_dataset = ImageFolder(train_data_path, transform=transforms["training"])
val_dataset = ImageFolder(validation_data_path, transform=transforms["validation"])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = AlexNetTransfer(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

trainer = Trainer(model, criterion, optimizer, device)
trainer.fit(train_loader, val_loader, 32)
