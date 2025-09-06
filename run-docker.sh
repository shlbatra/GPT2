#!/bin/bash
# Simple script to build and run GPT-2 Docker container

docker build -t gpt2-training .
docker run -it gpt2-training