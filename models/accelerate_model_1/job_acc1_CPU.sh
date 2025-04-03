#!/bin/bash
#SBATCH -p cola01
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --job-name=acc1_CPU
#SBATCH --output=acc1_CPU.log

bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate accelerate && time accelerate launch --config_file /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_1/config_cpubase.yaml /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_1/acc_inference_MLP.py"
#bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate accelerate && time accelerate launch --config_file /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_1/config_ipexbase.yaml /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_1/acc_inference_MLP.py"

