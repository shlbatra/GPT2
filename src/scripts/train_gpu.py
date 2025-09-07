from gradient import JobsClient

if __name__ == "__main__":
    # Set up authentication
    api_key = "220fe7f5a568047178b130963bae25"  # or use environment variable
    project_id = "gpu-cluster"

    # Initialize client
    jobs_client = JobsClient(api_key=api_key)

    # Define job configuration
    job_spec = {
        "name": "gpu-gpt2-job",
        "projectId": project_id,
        "machineType": "A4000",
        "container": "shlbatra123/gpu_docker_image:latest",
        "command": "python data/fineweb.py && python train.py",
        "workspace": "GPT2/src",  # or local path
        "artifacts" : ["../checkpoints", "../logs"]
    }

    # Create and run job
    try:
        job = jobs_client.create(**job_spec)
        print(f"Job created successfully!")
        print(f"Job ID: {job.id}")
        print(f"Job URL: https://console.paperspace.com/projects/{project_id}/jobs/{job.id}")
        
        # Monitor job status
        status = jobs_client.get(job.id)
        print(f"Job Status: {status.state}")
        
    except Exception as e:
        print(f"Error creating job: {e}")