from accelerate import Accelerator, ProfileKwargs
import torch
import time
import os
from torchvision.models import efficientnet_b5

# ---------- Inicializar profiler y accelerator ----------
def trace_handler(p):
    # Exportar trace para Chrome
    trace_path = f"traces/trace_{p.step_num}.json"
    p.export_chrome_trace(trace_path)
    print(f"\n Archivo de perfil guardado en: {trace_path}")

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

accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
device = accelerator.device
print(f"Usando dispositivo de la Accelerator: {device}")

# ---------- Cargar modelo ----------
model = efficientnet_b5(weights="IMAGENET1K_V1")
model.eval()

# ---------- Crear un batch de entrada ----------
batch_size = 128
input_images = torch.rand((batch_size, 3, 456, 456))
input_images = input_images.to(device)
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
