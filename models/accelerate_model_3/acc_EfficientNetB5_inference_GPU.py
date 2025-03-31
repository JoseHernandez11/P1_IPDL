from accelerate import Accelerator, ProfileKwargs
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b5
import time
import os

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

# ---------- Inicializar Accelerator ----------
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
device = accelerator.device
print(f" Usando dispositivo: {device}")

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

# ---------- Preparar modelo e input con accelerator ----------
model, input_images = accelerator.prepare(model, input_images)

# ---------- Iniciar perfilado e inferencia ----------
start_time = time.time()

with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_images)

end_time = time.time()
execution_time = end_time - start_time

