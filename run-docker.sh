#!/bin/bash
# Simple script to build and run GPT-2 Docker container

docker build -t gpt2-training .
docker run -it gpt2-training

# Run pushed docker image
docker run --rm -it shlbatra123/gpu_docker_image:latest bash