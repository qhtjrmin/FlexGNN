FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy binary and supporting libraries
COPY flexgnn .
COPY libtorch/ ./libtorch/

# Make the binary executable
RUN chmod +x flexgnn

# Set library path for CUDA and libtorch
ENV LD_LIBRARY_PATH=/app/libtorch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Default command
CMD ["./flexgnn"]

