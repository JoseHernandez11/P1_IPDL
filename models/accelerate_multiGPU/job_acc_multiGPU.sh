#!/bin/bash

# === SLURM Job Configuration ===
#SBATCH -p cola03                            # Cola/partición
#SBATCH --gres=gpu:$(nvidia-smi -L | wc -l)   # Usa todas las GPUs disponibles en el nodo
#SBATCH -c 1                                 # Núcleos de CPU por tarea
#SBATCH --mem=4G                             # Memoria RAM por nodo
#SBATCH --nodes=1                            # Número de nodos
#SBATCH --ntasks=1                           # Tareas
#SBATCH --time=00:10:00                      # Tiempo máximo
#SBATCH --job-name=acc_multiGPU             # Nombre del trabajo
#SBATCH --output=acc_multiGPU.log           # Archivo de salida

# === Ejecución con Apptainer/Singularity + entorno Conda + Accelerate ===
apptainer exec \
  --writable-tmpfs \
  --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -c "
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate accelerate && \
    echo '==> GPUs disponibles:' && \
    nvidia-smi && \
    echo '==> Lanzando entrenamiento con configuración multigpu desde YAML' && \
    time accelerate launch \
      --config_file /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_multiGPU/config_multigpu.yaml \
      /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_multiGPU/acc_train_TinyVGG_GPU.py
  "