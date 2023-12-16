import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(torch.__version__)
print(torch.cuda.is_available())

# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 32x32 (adjust as needed)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# Define paths to your image folders (power poles and trees)
train_path = 'C:\\Users\\Nurlan\Documents\\vs code\\ml_project\\train'
val_path = 'C:\\Users\\Nurlan\Documents\\vs code\\ml_project\\val'

# Create datasets from ImageFolder for power poles and trees
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

# Use DataLoader to create a loader for the combined dataset
batch_size = 100  # Define your batch size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda")

seq_model = nn.Sequential(
            nn.Linear(12288, 4096),
            nn.Tanh(),
            nn.Linear(4096, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 3),
            nn.LogSoftmax(dim=1))

optimizer = optim.Adam(seq_model.parameters(), lr=1e-3) # <1>

loss_fn = nn.NLLLoss()
n_epochs = 100
for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        outputs = seq_model(imgs.view(imgs.shape[0], -1))
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))


#train test
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in train_loader:
        outputs = seq_model(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

print("Train accuracy: %f" % (correct / total))


#validation

predicted_labels = []
true_labels = []

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        outputs = seq_model(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        predicted_labels.extend(predicted.numpy())
        true_labels.extend(labels.numpy())

print("Validation accuracy: %f" % (correct / total))

from sklearn.metrics import confusion_matrix, classification_report

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Classification report
class_report = classification_report(true_labels, predicted_labels)

# Print additional metrics and visualizations
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)