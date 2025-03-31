from accelerate import Accelerator, ProfileKwargs
import torch
import time
import os
from torchvision.models import efficientnet_b5

# ---------- Inicializar profiler y accelerator ----------
def trace_handler(p):
    trace_path = "traces/trace_0.json"
    p.export_chrome_trace(trace_path)
    print("Archivo de perfil guardado en:", trace_path)

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    record_shapes=True,
    with_stack=True,
    on_trace_ready=trace_handler
)

accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
device = accelerator.device
print(f"Usando dispositivo de la Accelerator: {device}")

# ---------- Cargar modelo ----------
model = efficientnet_b5(weights="IMAGENET1K_V1")
model.eval()

# ---------- Crear un batch de entrada ----------
batch_size = 128
input_images = torch.rand((batch_size, 3, 456, 456))

# ---------- Preparar con accelerator ----------
model, input_images = accelerator.prepare(model, input_images)

# Aquí imprimimos en qué dispositivo se encuentra todo:
print("Dispositivo del modelo:", next(model.parameters()).device)
print("Dispositivo de input_images:", input_images.device)

# ---------- Perfilado e inferencia ----------
start_time = time.time()

with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_images)

elapsed_time = time.time() - start_time
print(f"\nTiempo total: {elapsed_time:.4f} s")

print("\n--- GPU profiling (top 10) ---")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
