# Use the base image
FROM nvidia/cuda:11.2.2-devel-ubi8

# Set working directory inside the container
WORKDIR /app

# Install Python 3.11 and pip
RUN dnf install -y python3.11 && \
    python3.11 -m ensurepip && \
    python3.11 -m pip install --upgrade pip

# Copy the requirements.txt from the host to the container
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Set any additional configurations or commands as needed

# Specify the command to run on container start (optional)
CMD ["bash"]
