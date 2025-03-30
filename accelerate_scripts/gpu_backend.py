from accelerate import Accelerator, ProfileKwargs
import torch
import torchvision.models as models

# Initialize the ResNet18 model and inputs
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

# Define profiling kwargs for GPU activities
profile_kwargs = ProfileKwargs(
    activities=["cuda"],  # Profile CUDA (GPU) activities instead of CPU
    record_shapes=True
)

# Initialize the accelerator for GPU
accelerator = Accelerator(cpu=False, kwargs_handlers=[profile_kwargs])

# Prepare the model for GPU execution
model = accelerator.prepare(model)

# Move inputs to GPU
inputs = inputs.to(accelerator.device)

# Profile the model execution on the GPU
with accelerator.profile() as prof:
    with torch.no_grad():
        model(inputs)

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
