from accelerate import Accelerator, ProfileKwargs
import torch
import time
import os

# 1) Callback para profiler
def trace_handler(p):
    print("Archivo de perfil guardado.")
    p.export_chrome_trace("traces/trace_0.json")

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    record_shapes=True,
    with_stack=True,
    on_trace_ready=trace_handler
)

# 2) Accelerator
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
device = accelerator.device
print(f"Usando dispositivo: {device}")

# 3) Modelo y modo eval
from torchvision.models import efficientnet_b5
model = efficientnet_b5(weights="IMAGENET1K_V1")
model.eval()

# 4) Crear datos de prueba
batch_size = 128
input_images = torch.rand((batch_size, 3, 456, 456))  # CPU, float32

# 5) Preparar con accelerator
model, input_images = accelerator.prepare(model, input_images)
# A PARTIR DE AQUÍ, modelo y input_images están en GPU

# 6) Perfilado e inferencia
start_time = time.time()
with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_images)
end_time = time.time() - start_time

print(f"\nTiempo total de ejecución: {end_time:.4f} segundos")
