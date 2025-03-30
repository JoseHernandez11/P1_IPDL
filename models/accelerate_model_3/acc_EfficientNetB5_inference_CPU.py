from accelerate import Accelerator, ProfileKwargs
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b5
import time
import os

# ---------- Inicializar profiler y accelerator ----------
profile_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True,
    with_stack=True
)

accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])
device = accelerator.device
print(f"Usando dispositivo: {device}")

# ---------- Cargar modelo ----------
model = efficientnet_b5(weights="IMAGENET1K_V1")
model.eval()

# ---------- Preparar batch de imágenes ----------
transform = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

batch_size = 128
input_images = torch.rand((batch_size, 3, 456, 456))

# ---------- Preparar todo con accelerator ----------
model, input_images = accelerator.prepare(model, input_images)

# ---------- Iniciar perfilado e inferencia ----------
start_time = time.time()

with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_images)

end_time = time.time()
execution_time = end_time - start_time

# ---------- Resultados ----------
print(f"\n✅ Tiempo total de ejecución: {execution_time:.4f} segundos")

# ---------- Mostrar resumen de perfilado ----------
print("\n--- CPU Profiling: Top 10 operaciones más costosas ---")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
