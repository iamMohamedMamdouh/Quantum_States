import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
IMAGE_SIZE = 128
DATASET_PATH = "bloch_dataset"
CATEGORIES = ["entangled", "mixed", "pure"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "train"), transform=transform)
test_data = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "test"), transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

class_names = sorted(train_data.classes)
num_classes = len(class_names)

class QuantumCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = QuantumCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_acc_list = []

for epoch in range(EPOCHS):
    correct = 0
    total = 0
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    train_acc_list.append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Accuracy: {acc:.2f}%")

plt.plot(train_acc_list)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.savefig("training_accuracy.png")
plt.show()

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

torch.save(model.state_dict(), "quantum_bloch_cnn.pth")
print("Model saved as quantum_bloch_cnn.pth")

def show_predictions():
    model.eval()
    inputs, labels = next(iter(test_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(12, 8))
    for i in range(min(6, len(inputs))):
        plt.subplot(2, 3, i + 1)
        img = inputs[i].cpu().permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"True: {class_names[labels[i]]} | Pred: {class_names[preds[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_predictions()

#//////////////////////////////////شغال//////////////////////////////////
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
#
# # ------------------- CONFIG -------------------
# BATCH_SIZE = 32
# EPOCHS = 30
# LR = 0.001
# IMAGE_SIZE = 128
# DATASET_PATH = "bloch_dataset"
#
# # ------------------- DEVICE -------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # ------------------- TRANSFORMS -------------------
# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
# ])
#
# # ------------------- DATASET -------------------
# train_data = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "train"), transform=transform)
# test_data = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "test"), transform=transform)
#
# train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
#
# # استخدم الترتيب الصحيح للفئات
# class_names = sorted(train_data.classes)
# num_classes = len(class_names)
#
# # ------------------- MODEL -------------------
# class QuantumClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#         )
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.fc(x)
#         return x
#
# model = QuantumClassifier(num_classes=num_classes).to(device)
#
# # ------------------- TRAINING -------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)
#
# train_acc_list = []
#
# for epoch in range(EPOCHS):
#     correct = 0
#     total = 0
#     model.train()
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     acc = 100 * correct / total
#     train_acc_list.append(acc)
#     print(f"Epoch {epoch+1}/{EPOCHS} - Train Accuracy: {acc:.2f}%")
#
# # ------------------- PLOT -------------------
# plt.plot(train_acc_list)
# plt.title("Training Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy (%)")
# plt.grid()
# plt.savefig("training_accuracy.png")
# plt.show()
#
# # ------------------- EVALUATION -------------------
# model.eval()
# all_preds = []
# all_labels = []
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#
# print("📊 Classification Report:")
# print(classification_report(all_labels, all_preds, target_names=class_names))
#
# cm = confusion_matrix(all_labels, all_preds)
# plt.figure(figsize=(7, 6))
# sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt='d', cmap="Blues")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()
#
# torch.save(model.state_dict(), "quantum_bloch_cnn.pth")
# print("✅ Model saved as quantum_bloch_cnn.pth")
#
# # ------------------- VISUALIZATION -------------------
# def show_predictions():
#     model.eval()
#     inputs, labels = next(iter(test_loader))
#     inputs, labels = inputs.to(device), labels.to(device)
#     outputs = model(inputs)
#     _, preds = torch.max(outputs, 1)
#
#     plt.figure(figsize=(12, 8))
#     for i in range(min(6, inputs.shape[0])):
#         plt.subplot(2, 3, i + 1)
#         img = inputs[i].cpu().permute(1, 2, 0).numpy()
#         plt.imshow(img)
#         plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
#         plt.axis("off")
#     plt.tight_layout()
#     plt.show()
#
# show_predictions()
