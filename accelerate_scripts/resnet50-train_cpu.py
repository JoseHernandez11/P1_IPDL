from accelerate import Accelerator, ProfileKwargs
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load ResNet-50 model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # Adjust output layer for 10 classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a batch of random images and labels (batch size 128, 3 color channels, 224x224 resolution)
batch_size = 128
input_images = torch.rand((batch_size, 3, 224, 224))  # Random image batch
labels = torch.randint(0, 10, (batch_size,))  # Random labels for 10 classes

dataset = TensorDataset(input_images, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define profiling kwargs for CPU activities
profile_kwargs = ProfileKwargs(
    activities=["cpu"],  # Profile CPU  activities
    record_shapes=True
)

# Initialize the accelerator for CPU
accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])

# Prepare the model, optimizer, and data loader for CPU execution
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Move model to training mode
model.train()

device = accelerator.device

# Training loop
num_epochs = 3
with accelerator.profile() as prof:
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

# Print profiling results
print("Training completed.")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
