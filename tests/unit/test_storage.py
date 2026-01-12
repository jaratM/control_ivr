"""
Unit tests for the MinIO storage client module.
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
import tempfile
import os

from storage.minio_client import MinioStorage


class TestMinioStorage:
    """Test cases for the MinioStorage class."""
    
    @pytest.fixture
    def minio_config(self):
        """Sample MinIO configuration."""
        return {
            'endpoint': 'localhost:9000',
            'access_key': 'test_access',
            'secret_key': 'test_secret',
            'bucket_name': 'test-bucket',
            'secure': False
        }
    
    @patch('storage.minio_client.Minio')
    def test_init(self, mock_minio_class, minio_config):
        """Test MinioStorage initialization."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        storage = MinioStorage(**minio_config)
        
        assert storage.bucket_name == 'test-bucket'
        assert storage.client is not None
        mock_minio_class.assert_called_once_with(
            endpoint='localhost:9000',
            access_key='test_access',
            secret_key='test_secret',
            secure=False
        )
    
    @patch('storage.minio_client.Minio')
    def test_init_creates_bucket_if_not_exists(self, mock_minio_class, minio_config):
        """Test that bucket is created if it doesn't exist."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = False
        
        storage = MinioStorage(**minio_config)
        
        mock_client.bucket_exists.assert_called_once_with(bucket_name='test-bucket')
        mock_client.make_bucket.assert_called_once_with(bucket_name='test-bucket')
    
    @patch('storage.minio_client.Minio')
    def test_bucket_exists_no_creation(self, mock_minio_class, minio_config):
        """Test that bucket is not created if it already exists."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        storage = MinioStorage(**minio_config)
        
        mock_client.bucket_exists.assert_called_once()
        mock_client.make_bucket.assert_not_called()
    
    @patch('storage.minio_client.Minio')
    def test_list_objects(self, mock_minio_class, minio_config):
        """Test listing objects in bucket."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        # Mock list_objects response
        mock_objects = [
            MagicMock(object_name='file1.csv'),
            MagicMock(object_name='file2.xlsx'),
            MagicMock(object_name='folder/file3.wav')
        ]
        mock_client.list_objects.return_value = mock_objects
        
        storage = MinioStorage(**minio_config)
        results = list(storage.list_objects(prefix='servicenow'))
        
        assert len(results) == 3
        assert 'file1.csv' in results
        assert 'file2.xlsx' in results
        assert 'folder/file3.wav' in results
        
        mock_client.list_objects.assert_called_once_with(
            bucket_name='test-bucket',
            prefix='servicenow',
            recursive=True
        )
    
    @patch('storage.minio_client.Minio')
    def test_list_objects_empty_bucket(self, mock_minio_class, minio_config):
        """Test listing objects in empty bucket."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        mock_client.list_objects.return_value = []
        
        storage = MinioStorage(**minio_config)
        results = list(storage.list_objects())
        
        assert len(results) == 0
        assert isinstance(results, list)
    
    @patch('storage.minio_client.Minio')
    def test_download_file(self, mock_minio_class, minio_config, temp_dir):
        """Test downloading a file from MinIO."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        storage = MinioStorage(**minio_config)
        
        remote_path = 'servicenow/manifest.csv'
        local_path = os.path.join(temp_dir, 'downloaded.csv')
        
        storage.download_file(remote_path, local_path)
        
        mock_client.fget_object.assert_called_once_with(
            bucket_name='test-bucket',
            object_name=remote_path,
            file_path=local_path
        )
    
    @patch('storage.minio_client.Minio')
    def test_download_file_creates_directory(self, mock_minio_class, minio_config, temp_dir):
        """Test that download_file creates parent directories."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        storage = MinioStorage(**minio_config)
        
        # Path with nested directories
        local_path = os.path.join(temp_dir, 'subdir', 'nested', 'file.csv')
        
        storage.download_file('remote.csv', local_path)
        
        # Directory should be created
        assert os.path.exists(os.path.dirname(local_path))
    
    @patch('storage.minio_client.Minio')
    def test_upload_file(self, mock_minio_class, minio_config, temp_dir):
        """Test uploading a file to MinIO."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        storage = MinioStorage(**minio_config)
        
        # Create a test file
        local_path = os.path.join(temp_dir, 'test_upload.txt')
        with open(local_path, 'w') as f:
            f.write('test content')
        
        remote_path = 'uploads/test_upload.txt'
        
        storage.upload_file(local_path, remote_path)
        
        mock_client.fput_object.assert_called_once_with(
            bucket_name='test-bucket',
            object_name=remote_path,
            file_path=local_path
        )
    
    @patch('storage.minio_client.Minio')
    def test_upload_file_not_exists_raises_error(self, mock_minio_class, minio_config):
        """Test that uploading non-existent file raises error."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        storage = MinioStorage(**minio_config)
        
        with pytest.raises(FileNotFoundError):
            storage.upload_file('/nonexistent/file.txt', 'remote.txt')
    
    @patch('storage.minio_client.Minio')
    def test_file_exists_true(self, mock_minio_class, minio_config):
        """Test checking if file exists (returns True)."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        mock_client.stat_object.return_value = MagicMock()  # File exists
        
        storage = MinioStorage(**minio_config)
        
        exists = storage.file_exists('path/to/file.csv')
        
        assert exists is True
        mock_client.stat_object.assert_called_once_with(bucket_name='test-bucket', object_name='path/to/file.csv')
    
    @patch('storage.minio_client.Minio')
    def test_file_exists_false(self, mock_minio_class, minio_config):
        """Test checking if file exists (returns False)."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        # Simulate file not found
        from minio.error import S3Error
        mock_client.stat_object.side_effect = S3Error(
            code='NoSuchKey',
            message='Object not found',
            resource='',
            request_id='',
            host_id='',
            response=MagicMock()
        )
        
        storage = MinioStorage(**minio_config)
        
        exists = storage.file_exists('nonexistent.csv')
        
        assert exists is False
    
    @patch('storage.minio_client.Minio')
    def test_delete_file(self, mock_minio_class, minio_config):
        """Test deleting a file from MinIO."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        storage = MinioStorage(**minio_config)
        
        storage.delete_file('path/to/delete.csv')
        
        mock_client.remove_object.assert_called_once_with(
            bucket_name='test-bucket',
            object_name='path/to/delete.csv'
        )
    
    @patch('storage.minio_client.Minio')
    def test_get_file_url(self, mock_minio_class, minio_config):
        """Test getting presigned URL for file."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        mock_client.presigned_get_object.return_value = 'https://minio.com/presigned-url'
        
        storage = MinioStorage(**minio_config)
        
        url = storage.get_file_url('path/to/file.csv')
        
        assert url == 'https://minio.com/presigned-url'
        mock_client.presigned_get_object.assert_called_once()
    
    @patch('storage.minio_client.Minio')
    def test_secure_connection(self, mock_minio_class):
        """Test initialization with secure connection."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        storage = MinioStorage(
            endpoint='secure-minio.com',
            access_key='key',
            secret_key='secret',
            bucket_name='bucket',
            secure=True
        )
        
        mock_minio_class.assert_called_once_with(
            endpoint='secure-minio.com',
            access_key='key',
            secret_key='secret',
            secure=True
        )
    
    @patch('storage.minio_client.Minio')
    def test_list_objects_with_filter(self, mock_minio_class, minio_config):
        """Test listing objects with prefix filter."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        mock_objects = [
            MagicMock(object_name='servicenow/2025/file1.csv'),
            MagicMock(object_name='servicenow/2025/file2.csv'),
        ]
        mock_client.list_objects.return_value = mock_objects
        
        storage = MinioStorage(**minio_config)
        results = list(storage.list_objects(prefix='servicenow/2025'))
        
        assert len(results) == 2
        mock_client.list_objects.assert_called_with(
            bucket_name='test-bucket',
            prefix='servicenow/2025',
            recursive=True
        )
    
    @patch('storage.minio_client.Minio')
    def test_batch_download(self, mock_minio_class, minio_config, temp_dir):
        """Test downloading multiple files."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        storage = MinioStorage(**minio_config)
        
        files = [
            ('remote1.csv', os.path.join(temp_dir, 'local1.csv')),
            ('remote2.csv', os.path.join(temp_dir, 'local2.csv')),
        ]
        
        for remote, local in files:
            storage.download_file(remote, local)
        
        assert mock_client.fget_object.call_count == 2
    
    @patch('storage.minio_client.Minio')
    def test_error_handling_on_download(self, mock_minio_class, minio_config, temp_dir):
        """Test error handling when download fails."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        # Simulate download error
        mock_client.fget_object.side_effect = Exception("Network error")
        
        storage = MinioStorage(**minio_config)
        
        with pytest.raises(Exception, match="Network error"):
            storage.download_file('remote.csv', os.path.join(temp_dir, 'local.csv'))
    
    @patch('storage.minio_client.Minio')
    def test_bucket_name_property(self, mock_minio_class, minio_config):
        """Test that bucket_name is accessible as property."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.bucket_exists.return_value = True
        
        storage = MinioStorage(**minio_config)
        
        assert storage.bucket_name == 'test-bucket'
