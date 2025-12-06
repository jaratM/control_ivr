import os
import io
import json
from minio import Minio
from minio.error import S3Error
from loguru import logger
from typing import List, Generator, Optional, Any

class MinioStorage:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket_name: str = "calls-data", secure: bool = False):
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Ensures the bucket exists."""
        try:
            # Updated to use keyword arguments for MinIO SDK compatibility
            if not self.client.bucket_exists(bucket_name=self.bucket_name):
                self.client.make_bucket(bucket_name=self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"MinIO Bucket Error: {e}")

    def list_objects(self, prefix: str = "", recursive: bool = True) -> Generator[str, None, None]:
        """Lists objects in the bucket."""
        try:
            objects = self.client.list_objects(bucket_name=self.bucket_name, prefix=prefix, recursive=recursive)
            for obj in objects:
                yield obj.object_name
        except S3Error as e:
            logger.error(f"Error listing objects: {e}")

    def get_file(self, object_name: str) -> Optional[bytes]:
        """Retrieves a file's content."""
        try:
            response = self.client.get_object(bucket_name=self.bucket_name, object_name=object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error(f"Error getting object {object_name}: {e}")
            return None
    
    def get_json(self, object_name: str) -> Optional[dict]:
        """Retrieves and parses a JSON file."""
        data = self.get_file(object_name)
        if data:
            try:
                return json.loads(data.decode('utf-8'))
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {object_name}")
                return None
        return None

    def upload_file(self, object_name: str, file_path: str):
        """Uploads a file from local path."""
        try:
            # fput_object(bucket_name, object_name, file_path, ...)
            self.client.fput_object(bucket_name=self.bucket_name, object_name=object_name, file_path=file_path)
            logger.info(f"Uploaded {file_path} to {object_name}")
        except S3Error as e:
            logger.error(f"Error uploading {object_name}: {e}")

    def download_file(self, object_name: str, file_path: str) -> bool:
        """Downloads a file to local path."""
        try:
            self.client.fget_object(bucket_name=self.bucket_name, object_name=object_name, file_path=file_path)
            return True
        except S3Error as e:
            logger.error(f"Error downloading {object_name}: {e}")
            return False
