import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os 

print(os.getcwd())

## 0. Configuración del dispositivo
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

## 1. Carga del dataset.
lags_df = pd.read_csv("data/air_quality_2022_laAljorra.csv")
seed = 123

## 2. División de train y test.
train_df, _ = train_test_split(lags_df, test_size=0.2, random_state=seed)

## 3. Conversión a tensores del dataset de entrenamiento.
X_train = torch.tensor(train_df.iloc[:, :-1].values, dtype=torch.float32).to(device)
y_train = torch.tensor(train_df.iloc[:, -1].values, dtype=torch.float32).to(device)

## 4. Definición del modelo MLP
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

## 5. Inicializar modelo, función de pérdida y optimizador.
model = MLP().to(device)  # Mover modelo al dispositivo
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## 6. Entrenamiento del modelo.

epochs = 100
start_time = time.time()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
print(f"Tiempo de ejecución: {time.time() - start_time:.2f} s")

## 7. Guardar el modelo
torch.save(model.state_dict(), "trained_models/model1_IDL.pth")
