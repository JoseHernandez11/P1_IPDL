from accelerate import Accelerator, ProfileKwargs
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# ---------------------------
# 1. Definición del modelo
# ---------------------------
n_input = 12
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_input, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# ---------------------------
# 2. Función de perfilado
# ---------------------------
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    os.makedirs("executions", exist_ok=True)
    p.export_chrome_trace("executions/trace_" + str(p.step_num) + ".json")

profile_kwargs = ProfileKwargs(
    activities=["cpu", "cuda"],
    schedule_option={
        "wait": 5,
        "warmup": 1,
        "active": 3,
        "repeat": 2,
        "skip_first": 1,
    },
    on_trace_ready=trace_handler
)

# ---------------------------
# 3. Cargar modelo
# ---------------------------
def load_model(model_path):
    model = MLP()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model_path = "trained_models/model1_IDL.pth"
model = load_model(model_path)

# ---------------------------
# 4. Cargar datos
# ---------------------------
lags_df = pd.read_csv("data/air_quality_20202021_inference_laAljorra.csv")
X_test = torch.tensor(lags_df.iloc[:, :-1].values, dtype=torch.float32)
y_test = torch.tensor(lags_df.iloc[:, -1].values, dtype=torch.float32)

# ---------------------------
# 5. Inicializar Accelerator
# ---------------------------
accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
device = accelerator.device

# Preparar el modelo (es importante hacerlo antes de mover los tensores)
model = accelerator.prepare(model)

# Mover tensores manualmente al dispositivo correcto
X_test = X_test.to(device)
y_test = y_test.to(device)

# ---------------------------
# 6. Inferencia perfilada
# ---------------------------
criterion = nn.MSELoss()
start_time = time.time()

with accelerator.profile() as prof:
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        test_loss = criterion(y_pred, y_test)

end_time = time.time() - start_time

# ---------------------------
# 7. Resultados
# ---------------------------
print(f"Inference time: {end_time:.4f} s")
print(f"Test loss: {test_loss.item():.4f}")

# 🔍 Mostrar trazas explícitamente
if prof:
    print("\n --- GPU Profiling: Top 10 operaciones más costosas ---")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    print("\n --- CPU Profiling: Top 10 operaciones más costosas ---")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
else:
    print("Profiler was not initialized correctly or did not capture any data.")