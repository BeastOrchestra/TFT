# Send ./data/ to AWS S3 bucket arj_ibdata
import boto3
import os
from pathlib import Path

def upload_to_s3(local_path, bucket_name, s3_path):
    # Create S3 client
    s3_client = boto3.client('s3')
    
    # Handle both single files and directories
    local_path = Path(local_path)
    if local_path.is_file():
        s3_client.upload_file(str(local_path), bucket_name, s3_path)
    elif local_path.is_dir():
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                # Preserve directory structure in S3
                relative_path = file_path.relative_to(local_path)
                s3_key = str(Path(s3_path) / relative_path)
                s3_client.upload_file(str(file_path), bucket_name, s3_key)

# Example usage
upload_to_s3('./data', 'arj-ibdata', 'data/')