from accelerate import Accelerator, ProfileKwargs
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b5
import time
import os

# ---------- Inicializar profiler y accelerator ----------
def trace_handler(p):
    # Exportar trazado para Chrome
    trace_path = f"/tmp/trace_{p.step_num}.json"
    p.export_chrome_trace(trace_path)

    print(f"\n🧠 Archivo de perfil guardado en: {trace_path}")

    # Imprimir resumen explícito de GPU
    print("\n--- GPU Profiling: Top 10 operaciones más costosas (por self_cuda_time_total) ---")
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    # Imprimir resumen explícito de CPU
    print("\n--- CPU Profiling: Top 10 operaciones más costosas (por self_cpu_time_total) ---")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],  # Ambos perfiles
    record_shapes=True,
    with_stack=True,
    on_trace_ready=trace_handler
)

# ---------- Inicializar Accelerator ----------
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])
device = accelerator.device
print(f"✅ Usando dispositivo: {device}")

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

# ---------- Tiempo total ----------
print(f"\n⏱️ Tiempo total de ejecución: {execution_time:.4f} segundos")

# ---------- Mostrar resumen explícito aquí también por si no se usa on_trace_ready ----------
print("\n✅ Resumen explícito adicional:")

# Imprimir resumen GPU
print("\n--- GPU Profiling: Top 10 operaciones más costosas (por self_cuda_time_total) ---")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

# Imprimir resumen CPU
print("\n--- CPU Profiling: Top 10 operaciones más costosas (por self_cpu_time_total) ---")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
