# Use the Hugging Face Accelerate GPU release image as the base

FROM huggingface/accelerate:gpu-nightly



# Set the working directory inside the container
WORKDIR /accelerate_gpu


# Install the required Python packages

RUN conda install -c conda-forge libstdcxx-ng=12.2.0

RUN pip install accelerate pyvim timm torchvision scikit-learn evaluate datasets transformers intel_extension_for_pytorch

# Copy all files in the current directory
COPY . . 


# Clone the Hugging Face Accelerate repository
RUN git clone https://github.com/huggingface/accelerate.git


# Set an entrypoint or command if needed (optional)

CMD ["bash"]
