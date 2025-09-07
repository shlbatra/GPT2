# Create VM instance on Paperspace with GPU or Lambda with GPU

# Copy your SSH key to the VM
ls ~/.ssh
cat ~/.ssh/id_ed25519.pub -> copy to vm ssh 

# 1. SSH into your instance
ssh ubuntu@150.136.113.47

# 2. Create project directory
sudo apt update
mkdir my-gpu-project && cd my-gpu-project
sudo snap install docker 
sudo apt install nvidia-utils-570-server
sudo apt install nvidia-driver-570-server
# Download GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt update
sudo apt install nvidia-container-toolkit
sudo reboot
nvidia-smi
# 3. Create your Python scripts and Dockerfile
# (upload via scp, git clone, or create directly)

# 4. Pull Docker image and run container
sudo docker system prune -a
sudo docker pull shlbatra123/gpu_docker_image:latest

# 5. Run your scripts with GPU access
# sudo docker run --gpus all --rm shlbatra123/gpu_docker_image:latest python data/fineweb.py && python train.py
sudo docker run --rm shlbatra123/gpu_docker_image:latest python data/data_scripts/fineweb.py && python train_gpt.py
# 6. Download results to local machine (from local terminal)
scp -r ubuntu@129.213.148.102:~/my-gpu-project/results ./