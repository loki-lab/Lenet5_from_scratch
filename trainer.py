import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_metric = 0.0

    def fit(self, train_loader, valid_loader, max_epoch):
        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in range(max_epoch):
            print('Epoch: {}/{}.. '.format(epoch + 1, max_epoch))
            running_loss = 0.0
            running_corrects = 0.0
            val_loss = 0.0
            val_corrects = 0.0

            for data, target in tqdm(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(output.data, 1)

                running_loss += loss.item()
                running_corrects += predicted.eq(target).sum().item()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects / len(train_loader.dataset)

            print('Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch_loss, epoch_acc))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': epoch_loss,
            }, "checkpoint/latest_checkpoint.pth")

            with torch.no_grad():
                self.model.eval()
                for data, target in valid_loader:
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.model(data)
                    val_loss = self.criterion(output, target)

                    _, predicted = torch.max(output.data, 1)
                    val_corrects += predicted.eq(target).sum().item()

                val_loss = val_loss / len(valid_loader.dataset)
                val_acc = val_corrects / len(valid_loader.dataset)
                print('Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(val_loss, val_acc))
                if val_acc > self.best_metric:
                    self.best_metric = val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': epoch_loss,
                    }, "checkpoints/best_checkpoint.pth")
                    print("Save best model to checkpoint")

