import tiktoken
import numpy as np
import torch
import torch.nn as nn
import os
import logging
from google.cloud import storage
import io


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        self.logger_instance = logging.getLogger(__name__)

        # GCS configuration
        gcs_bucket_name = "gpt2-training-data-sahil"
        gcs_prefix = "edu_fineweb10B/"
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)
        
        # List all blobs (files) in the GCS bucket with the prefix
        blobs = bucket.list_blobs(prefix=gcs_prefix)
        
        # Filter shards based on split and collect GCS paths
        shard_blobs = []
        for blob in blobs:
            if blob.name.endswith('.npy') and split in blob.name:
                shard_blobs.append(blob)
        
        # Sort by filename for consistent ordering
        shard_blobs = sorted(shard_blobs, key=lambda x: x.name)
        
        # Store GCS blob references instead of local paths
        self.shards = shard_blobs
        assert len(shard_blobs) > 0, f"no shards found for split {split}"
        
        if master_process:
            self.logger_instance.info(f"found {len(shard_blobs)} shards for split {split}")
            # Optional: print first few shard names for verification
            for i, blob in enumerate(shard_blobs[:3]):
                self.logger_instance.info(f"  shard {i}: {blob.name}")
        
        self.reset()


    def load_tokens(self, shard_blob):
        """Load numpy array directly from GCS bucket and return as torch tensor"""
        # Download blob content to memory
        blob_data = shard_blob.download_as_bytes()
        
        # Load numpy array from bytes
        buffer = io.BytesIO(blob_data)
        tokens = np.load(buffer)
        npt = tokens.astype(np.int32) # added after video
        ptt = torch.tensor(npt, dtype=torch.long)
        
        return ptt

    def reset(self): # reset data loader as do model eval every 100th iteration
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
