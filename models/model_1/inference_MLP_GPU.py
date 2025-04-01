import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os 


model_path = "trained_models/model1_IDL.pth"

## 0. Configuración del dispositivo de inferencia.

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
print(f"Current working directory: {os.getcwd()}")


## 1. Carga del dataset.
lags_df = pd.read_csv("data/air_quality_20202021_inference_laAljorra.csv")
seed = 123

## 2. Conversión a tensores del dataset de entrenamiento.
X_test = torch.tensor(lags_df.iloc[:, :-1].values, dtype=torch.float32).to(device)
y_test = torch.tensor(lags_df.iloc[:, -1].values, dtype=torch.float32).to(device)

## 3. Definición del modelo MLP
n_input = 12  # tamaño del lag

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_input, 512)  # Aumentamos el número de neuronas
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)  # Salida
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Dropout para ralentizar el aprendizaje

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout después de la activación
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.fc6(x)  # Sin activación final para regresión
        return x

        
## 4. Carga del modelo ya entrenado, función de pérdida y optimizador.

def load_model(model_path, device):
    model = MLP().to(device)  # Mover el modelo al dispositivo
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model(model_path, device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## 5. Inferencia del modelo

start_time = time.time()

with torch.no_grad():
    y_pred = model(X_test)
    print(f"Inference predictions: {y_pred[:10]}")
    test_loss = criterion(y_pred, y_test)

end_time = time.time() - start_time

print(f"Inference time: {end_time} s")