import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator, ProfileKwargs

# ---------- Configuración del profiling ----------
def trace_handler(p):
    # Exportar trace para Chrome
    trace_path = f"/tmp/trace_{p.step_num}.json"
    p.export_chrome_trace(trace_path)
    print(f"\n🧠 Archivo de perfil guardado en: {trace_path}")

    # Imprimir resumen GPU
    print("\n--- GPU Profiling (on_trace_ready): Top 10 operaciones más costosas ---")
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    # Imprimir resumen CPU
    print("\n--- CPU Profiling (on_trace_ready): Top 10 operaciones más costosas ---")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    record_shapes=True,
    with_stack=True,
    on_trace_ready=trace_handler
)

# ---------- Inicializar Accelerator ----------
accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])
device = accelerator.device
print(f"✅ Usando dispositivo: {device}")

# ---------- Dataset ----------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

data_dir = "data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ---------- TinyVGG ----------
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super(TinyVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 16 * 16, output_shape)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------- Modelo, pérdida y optimizador ----------
num_classes = len(train_dataset.classes)
print(f"🔠 Clases: {train_dataset.classes}")

model = TinyVGG(input_shape=3, hidden_units=64, output_shape=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Preparar con Accelerator
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

# ---------- Entrenamiento con profiling ----------
epochs = 5
start_time = time.time()

with accelerator.profile() as prof:
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"📍 Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")

end_time = time.time()
print(f"\n⏱️ Entrenamiento completado en {end_time - start_time:.2f} segundos")

# ---------- Impresión explícita del profiling ----------
print("\n📊 🔍 TRAZA FINAL (al finalizar entrenamiento):")

# GPU
print("\n--- GPU Profiling: Top 10 operaciones más costosas ---")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

# CPU
print("\n--- CPU Profiling: Top 10 operaciones más costosas ---")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

# ---------- Evaluación ----------
model.eval()
test_loss = 0.0
correct = 0
total = 0

test_loader = accelerator.prepare(test_loader)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct / total

print(f"\n🧪 Test Loss: {avg_test_loss:.4f}")
print(f"✅ Test Accuracy: {test_accuracy:.2f}%")

# ---------- Guardar modelo ----------
os.makedirs("trained_models", exist_ok=True)
torch.save(model.state_dict(), "trained_models/tinyvgg_model.pth")
