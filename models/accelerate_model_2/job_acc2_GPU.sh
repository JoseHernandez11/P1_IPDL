#!/bin/bash
#SBATCH -p cola02
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --job-name=acc2_GPU
#SBATCH --output=acc2_GPU.log

apptainer exec --writable-tmpfs --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate accelerate && time accelerate launch --config_file /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_2/config_gpubase.yaml /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_2/acc_train_seq2seq_GPU.py"
