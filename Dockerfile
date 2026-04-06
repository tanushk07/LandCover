# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /segment_project

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content from the local directory to the working directory
COPY ./config ./config
COPY ./data ./data
COPY ./src ./src

# command to run on container start
# comment and uncomment either of the following lines based on whether to train or test the model
CMD [ "python", "./src/train_model.py", "--model", "unet" ]
# CMD [ "python", "./src/test_models.py", "--model", "unet" ]