"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
from google.cloud import storage # pip install google-cloud-storage
import io
import tempfile

# ------------------------------------------
# GCS Configuration
gcs_bucket_name = "gpt2-training-data-sahil"
gcs_prefix = "edu_fineweb10B/"
storage_client = storage.Client()  # Create client
bucket = storage_client.bucket(gcs_bucket_name)  # Get bucket reference

# Local Configuration
local_dir = "../data/edu_fineweb10B"
remote_name = "sample-10BT" # 'sample-10BT'; 1 Billion tokens, 100 M token per shard and 100 shards in total -> ls edu_fineweb10B/ | wc -l
shard_size = int(1e7) # 10M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16) # save more space
    return tokens_np_uint16

def write_datafile_to_gcs(filename, tokens_np):
    """Write numpy array directly to GCS bucket"""
    # Create the full GCS path
    gcs_path = f"{gcs_prefix}{filename}.npy"
    
    # Create a bytes buffer and save numpy array to it
    buffer = io.BytesIO()
    np.save(buffer, tokens_np)
    buffer.seek(0)
    
    # Upload to GCS
    blob = bucket.blob(gcs_path)
    blob.upload_from_file(buffer, content_type='application/octet-stream')
    
    print(f"Uploaded {gcs_path} to GCS bucket {gcs_bucket_name}")

if __name__ == '__main__':

    # Verify GCS bucket access
    try:
        bucket.reload()
        print(f"Successfully connected to GCS bucket: {gcs_bucket_name}")
    except Exception as e:
        print(f"Error accessing GCS bucket {gcs_bucket_name}: {e}")
        print("Make sure you have proper authentication and bucket access.")
        exit(1)

    # download the dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train[:50%]") # 50% of data for testing
    
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() - 2) # leave some spare CPU
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = f"edufineweb_{split}_{shard_index:06d}"
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile_to_gcs(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = f"edufineweb_{split}_{shard_index:06d}"
            write_datafile_to_gcs(filename, all_tokens_np[:token_count])