import boto3
import os
from pathlib import Path

def download_from_s3(bucket_name, s3_folder, local_dir):
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    
    # Ensure the local directory exists
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        if 'Contents' in page:
            for obj in page['Contents']:
                s3_key = obj['Key']
                rel_path = s3_key[len(s3_folder):].lstrip("/")  # Remove folder prefix
                local_path = local_dir / rel_path
                
                # Ensure parent directories exist
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # print(f"Downloading {s3_key} to {local_path}...")
                s3_client.download_file(bucket_name, s3_key, str(local_path))
    print('Complete Downloading Data')

# Example Usage
download_from_s3('arj-ibdata', 'data/', './data')
# All files for training and predicting should now be local