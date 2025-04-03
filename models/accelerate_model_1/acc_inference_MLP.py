from accelerate import Accelerator, ProfileKwargs
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

os.chdir("/home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_1/")

# ---------------------------
# 1. Carga del dataset
# ---------------------------
lags_df = pd.read_csv("data/air_quality_20202021_inference_laAljorra.csv")
X_test = torch.tensor(lags_df.iloc[:, :-1].values, dtype=torch.float32)
y_test = torch.tensor(lags_df.iloc[:, -1].values, dtype=torch.float32)

# ---------------------------
# 2. Definición del modelo
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
# 3. Cargar modelo entrenado
# ---------------------------

def load_model(model_path):
    model = MLP()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


model_path = "trained_models/model1_IDL.pth"
model = load_model(model_path)

# ---------------------------
# 4. Configurar Accelerate
# ---------------------------
profile_kwargs = ProfileKwargs(
    activities=["cpu"],  # Puede incluir "cpu" y/o "cuda"
    record_shapes=True
)
accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])

# Preparar modelo y datos
model, X_test, y_test = accelerator.prepare(model, X_test, y_test)
device = accelerator.device

# ---------------------------
# 5. Inferencia con perfilado
# ---------------------------
criterion = nn.MSELoss()

start_time = time.time()
with accelerator.profile() as prof:
    with torch.no_grad():
        y_pred = model(X_test)
        print(f"Inference predictions: {y_pred[:10]}")
        test_loss = criterion(y_pred.squeeze(), y_test)
end_time = time.time() - start_time

# ---------------------------
# 6. Resultados
# ---------------------------
print(f"Inference time: {end_time:.4f} s")
print(f"Test loss: {test_loss.item():.4f}")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
