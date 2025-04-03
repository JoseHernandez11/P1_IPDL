#!/bin/bash
# Configuración del trabajo en SLURM

#SBATCH -p cola02                      # Selecciona la partición "cola02"
#SBATCH --gres=gpu:1                   # Solicita 1 GPU
#SBATCH -c 1                           # Solicita 1 núcleo de CPU
#SBATCH --mem=4G                       # Reserva 4 GB de memoria
#SBATCH --nodes=1                      # Utiliza 1 nodo
#SBATCH --ntasks=1                     # Ejecuta 1 tarea
#SBATCH --time=00:10:00                # Tiempo máximo de ejecución: 10 minutos
#SBATCH --job-name=acc1_GPU            # Nombre del trabajo
#SBATCH --output=acc1_GPU.log          # Archivo de salida

# Ejecución de la aplicación con Apptainer/Singularity
apptainer exec --writable-tmpfs --nv \
  /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && \
           conda activate accelerate && \
           time accelerate launch --config_file /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_1/config_gpubase.yaml \
           /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_1/acc_inference_MLP_GPU.py"
