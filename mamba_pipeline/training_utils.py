import torch
from .data_utils import augment_data


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_model(model, optimizer, loss_fn, train_loader, val_loader, params, X_train, y_train, early_stopping):
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(params['n_epochs']):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            aug_images, aug_labels = augment_data(X_train, y_train, params['batch_size'])
            images = torch.cat((images, aug_images))
            labels = torch.cat((labels, aug_labels))
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Acc {val_accuracy:.2f}%")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return model, train_losses, val_losses, val_accuracies


def test_model(model, loss_fn, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
    return test_accuracy, test_loss
