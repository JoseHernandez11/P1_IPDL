#!/bin/bash
#SBATCH -p cola01
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --job-name=acc4_CPU
#SBATCH --output=acc4_CPU.log

apptainer exec --writable-tmpfs --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate accelerate && time accelerate launch --config_file /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_4/config_cpubase.yaml /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_4/acc_train_TinyVGG_CPU.py"
#apptainer exec --writable-tmpfs --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate accelerate && time accelerate launch --config_file /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_4/config_ipexbase.yaml /home/josemariahernandezn/IPDL/P1_IPDL/models/accelerate_model_4/acc_train_TinyVGG_CPU.py"

