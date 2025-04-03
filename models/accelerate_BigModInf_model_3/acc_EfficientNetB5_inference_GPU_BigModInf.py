from accelerate import Accelerator, ProfileKwargs
from accelerate.utils import infer_auto_device_map
from accelerate.utils.modeling import dispatch_model
import torch
import time
from torchvision.models import efficientnet_b5

# ---------- Inicializar profiler y accelerator ----------
def trace_handler(p):
    trace_path = f"traces/trace_{p.step_num}.json"
    p.export_chrome_trace(trace_path)
    print(f"\n Archivo de perfil guardado en: {trace_path}")
    print("\n--- GPU Profiling (on_trace_ready): Top 10 operaciones más costosas ---")
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print("\n--- CPU Profiling (on_trace_ready): Top 10 operaciones más costosas ---")
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    record_shapes=True,
    with_stack=True,
    on_trace_ready=trace_handler
)

accelerator = Accelerator(
    kwargs_handlers=[profile_kwargs],
    device_placement=True,  # Necesario para Big Model Inference
)
device = accelerator.device
print(f"Usando dispositivo de la Accelerator: {device}")

# ---------- Cargar modelo ----------
model = efficientnet_b5(weights="IMAGENET1K_V1")
model.eval()

# ---------- Big Model Inference: Distribuir modelo automáticamente ----------
device_map = infer_auto_device_map(model)
model = dispatch_model(model, device_map)
print("Device map asignado automáticamente:", device_map)

# ---------- Crear un batch de entrada ----------
batch_size = 128
input_images = torch.rand((batch_size, 3, 456, 456), dtype=torch.float32).to(device)

# ---------- Perfilado e inferencia ----------
start_time = time.time()

with accelerator.profile() as prof:
    with torch.no_grad():
        outputs = model(input_images)

elapsed_time = time.time() - start_time

print(f"Tiempo total de inferencia: {elapsed_time:.2f} segundos")
