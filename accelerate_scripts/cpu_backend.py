from accelerate import Accelerator, ProfileKwargs
import torch
import torchvision.models as models

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

profile_kwargs = ProfileKwargs(
    activities=["cpu"],
    record_shapes=True,
    profile_memory=True
)

accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])
model = accelerator.prepare(model)

with accelerator.profile() as prof:
    with torch.no_grad():
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
