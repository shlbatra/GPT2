# Create VM instance on Paperspace with GPU or Lambda with GPU

# Copy your SSH key to the VM
ls ~/.ssh
cat ~/.ssh/id_ed25519.pub -> copy to vm ssh 

# 1. SSH into your instance
ssh ubuntu@132.145.193.67

# 2. Create project directory
sudo apt update
mkdir my-gpu-project && cd my-gpu-project
sudo snap install docker 
sudo apt install nvidia-utils-570-server -y
sudo apt install nvidia-driver-570-server -y
# Download GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt update
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker -y
sudo reboot
nvidia-smi

# 3. Create your Python scripts and Dockerfile
# (upload via scp, git clone, or create directly)
scp -r gcp-key.json paperspace@184.105.4.46:~/my-gpu-project/
# 4. Pull Docker image and run container
sudo docker system prune -a
sudo docker pull shlbatra123/gpu_docker_image:latest

# 5. Run your scripts with GPU access
# sudo docker run --runtime=nvidia --rm -it shlbatra123/gpu_docker_image:latest bash 
# sudo docker run --runtime=nvidia --rm shlbatra123/gpu_docker_image:latest bash -c "uv pip install "google-cloud-storage>=2.10.0" && torchrun --nproc_per_node=2 train_gpt.py"
mkdir -p checkpoints
sudo docker run --runtime=nvidia -v $(pwd)/gcp-key.json:/app/gcp-key.json -v $(pwd)/checkpoints:/app/checkpoints -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json --rm shlbatra123/gpu_docker_image:latest bash -c "torchrun --nproc_per_node=2 train_gpt.py"

# 6. Download results to local machine (from local terminal)
scp -r ubuntu@129.213.148.102:~/my-gpu-project/results ./