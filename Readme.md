# P1\_IPDL - Práctica de Implementación de Modelos con PyTorch + Accelerate

Este repositorio contiene el desarrollo de varios modelos de aprendizaje profundo junto con sus versiones adaptadas para su ejecución eficiente mediante la librería Huggingface Accelerate. Se incluyen también configuraciones para pruebas de entrenamiento e inferencia tanto en máquina local como en entorno de clúster.

---

## 📁 Estructura del Proyecto

```plaintext
P1_IPDL/
├── models/
│   ├── model_1/                        → MLP
│   ├── model_2/                        → Seq2Seq Bahdanau
│   ├── model_3/                        → EfficientNet B5
│   ├── model_4/                        → Tiny-VGG
│   ├── accelerate_model_1/             → Versión Accelerate de model_1
│   ├── accelerate_model_2/             → Versión Accelerate de model_2
│   ├── accelerate_model_3/             → Versión Accelerate de model_3
│   ├── accelerate_model_4/             → Versión Accelerate de model_4
│   ├── accelerate_BigModInf_model_3/  → Inferencia BigModel para EfficientNet
│   └── accelerate_GradAccumulation_model_4/ → Entrenamiento con acumulación de gradientes (Tiny-VGG)
├── generate_times_plot.ipynb         → Notebook para generar gráficas comparativas
├── Dockerfile                        → Definición del entorno de ejecución en Docker
└── Readme.md                         → Este archivo
```

---

## 🚀 Instrucciones de Ejecución

### 2.1.1 Ejecución en máquina local

Los pasos seguidos para ejecutar de forma local los diferentes scripts de los experimentos realizados son los siguientes:

1. **Construir la imagen del Dockerfile proporcionado**:

   ```bash
   docker build -t p1_ipdl_image .
   ```

2. **Abrir una sesión interactiva del contenedor**:

   ```bash
   sudo docker container run -it -v ~/Documentos/P1_IPDL/:/workspace p1_ipdl_image bash
   ```

3. **Activar el entorno de Anaconda "accelerate"**.

4. **Ejecutar scripts de prueba**:

   ```bash
   time accelerate launch --config_file config.yaml script.py
   ```

> Nota: La ejecución local se realiza en un portátil con CPU Intel Core i7-8550U @ 1.80GHz, 16 GB RAM, sin GPU, por lo que los tiempos de ejecución son significativamente mayores que en el clúster.

---

### 2.1.2 Ejecución en el clúster (colas 01/02)

#### Especificaciones del clúster:

| Partición | CPU                         | Cores | RAM  | GPUs                |
| --------- | --------------------------- | ----- | ---- | ------------------- |
| cola01    | Intel Xeon Gold 6253CL      | 4     | 16GB | No                  |
| cola02    | Intel Xeon E5-2696V[x]/2689 | 4     | 16GB | 1 x NVIDIA Tesla T4 |

- **cola02** dispone de GPU, ideal para entrenamiento acelerado.
- **cola01** tiene una CPU más moderna, adecuada para tareas intensivas sin GPU.

#### A. Ejecución interactiva

1. **Reservar sesión interactiva**:

   ```bash
   srun --partition=cola02 --time=02:00:00 --ntasks=1 --gres=gpu:1 --cpus-per-task=1 --nodes=1 --mem=4GB --pty /bin/bash
   srun --partition=cola01 --time=02:00:00 --ntasks=1 --cpus-per-task=1 --nodes=1 --mem=4GB --pty /bin/bash
   ```

2. **Acceder al contenedor con Apptainer**:

   ```bash
   apptainer shell --nv /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif
   ```

3. **Activar entorno Anaconda "accelerate"**.

4. **Lanzar experimento**:

   ```bash
   time accelerate launch --config_file config.yaml script.py
   ```

#### B. Ejecución mediante trabajos `sbatch`

1. **Crear un script de trabajo (ej. ****`job.sh`****)** especificando:

   - partición (cola01 o cola02)
   - recursos (GPU, memoria, tiempo, CPUs)
   - comando de ejecución con Apptainer y Accelerate

2. **Enviar el trabajo a la cola**:

   ```bash
   sbatch job.sh
   ```

3. **Monitorear estado**:

   ```bash
   squeue
   ```

Para lanzar en la cola01 (sin GPU), simplemente cambiar el nombre de la partición y eliminar `--gres=gpu:1`.

---

## 📈 Visualización de Resultados

El notebook `generate_times_plot.ipynb` permite comparar gráficamente los tiempos de entrenamiento e inferencia entre modelos tradicionales y sus versiones aceleradas.

---

## ✍️ Autor

José María Hernández Nieto, Juan De Dios Rodríguez Garrido – 2025

