import time
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b5

# Cargar EfficientNet-B5 preentrenado
model = efficientnet_b5(weights="IMAGENET1K_V1")
model.eval()  # Modo evaluación

# Definir transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((456, 456)),  # EfficientNet-B5 espera 456x456
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Crear un lote de imágenes aleatorias (batch size 128, 3 canales, 456x456)
batch_size = 128
input_images = torch.rand((batch_size, 3, 456, 456))

# Mover modelo e imágenes a CPU
device = torch.device("cpu")  # Puedes cambiar a "cuda" si tienes GPU
model.to(device)
input_images = input_images.to(device)

# Medir el tiempo de ejecución de la inferencia
start_time = time.time()
with torch.no_grad():
    outputs = model(input_images)
end_time = time.time()

execution_time = end_time - start_time
print(f"Tiempo de ejecución: {execution_time:.4f} segundos")
