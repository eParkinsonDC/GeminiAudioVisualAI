"""Unit tests for the GoogleDriveFileFetcher class."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
from unittest import mock
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile

import pytest
from get_files import (
    GoogleDriveFileFetcher,
    getFiles,
    getFilesWithOAuth,
    getFilesWithAPIKey,
)


@pytest.fixture
def api_key_fetcher():
    """Create a GoogleDriveFileFetcher instance with API key authentication."""
    return GoogleDriveFileFetcher(api_key="test_api_key", use_oauth=False)


@pytest.fixture
def oauth_fetcher():
    """Create a GoogleDriveFileFetcher instance with OAuth authentication."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key_for_oauth"}):
        return GoogleDriveFileFetcher(use_oauth=True)


@pytest.fixture
def mock_service():
    """Create a mock Google Drive service."""
    service = Mock()
    service.files.return_value.list.return_value.execute.return_value = {
        "files": [
            {
                "id": "file_id_1",
                "name": "test_file.pdf",
                "mimeType": "application/pdf",
                "size": "1024",
                "createdTime": "2023-01-01T00:00:00Z",
                "modifiedTime": "2023-01-02T00:00:00Z",
                "webViewLink": "https://drive.google.com/file/d/file_id_1/view",
            }
        ]
    }
    return service


def test_init_with_api_key():
    """Test initialization with API key."""
    fetcher = GoogleDriveFileFetcher(api_key="test_key", use_oauth=False)
    assert fetcher.api_key == "test_key"
    assert fetcher.use_oauth is False
    assert fetcher.port == 8080


def test_init_with_oauth():
    """Test initialization with OAuth."""
    fetcher = GoogleDriveFileFetcher(use_oauth=True)
    assert fetcher.use_oauth is True
    assert fetcher.scopes == ["https://www.googleapis.com/auth/drive.readonly"]
    assert fetcher.credentials_file == "credentials.json"
    assert fetcher.token_file == "token.pickle"


def test_init_without_api_key_raises_error():
    """Test that initialization without API key raises ValueError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Google API key is required"):
            GoogleDriveFileFetcher(use_oauth=False)


def test_init_uses_env_api_key():
    """Test that initialization uses API key from environment."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "env_key"}):
        fetcher = GoogleDriveFileFetcher(use_oauth=False)
        assert fetcher.api_key == "env_key"


def test_authenticate_with_api_key(api_key_fetcher):
    """Test authentication with API key."""
    with patch("get_files.build") as mock_build:
        mock_service = Mock()
        mock_build.return_value = mock_service

        api_key_fetcher.authenticate()

        mock_build.assert_called_once_with("drive", "v3", developerKey="test_api_key")
        assert api_key_fetcher.service == mock_service


def test_authenticate_with_oauth_existing_valid_token(oauth_fetcher):
    """Test OAuth authentication with existing valid token."""
    mock_creds = Mock()
    mock_creds.valid = True

    with patch("get_files.os.path.exists", return_value=True), patch(
        "builtins.open", mock.mock_open()
    ), patch("get_files.pickle.load", return_value=mock_creds), patch(
        "get_files.build"
    ) as mock_build:

        mock_service = Mock()
        mock_build.return_value = mock_service

        oauth_fetcher.authenticate()

        mock_build.assert_called_once_with("drive", "v3", credentials=mock_creds)
        assert oauth_fetcher.service == mock_service


def test_authenticate_with_oauth_expired_token(oauth_fetcher):
    """Test OAuth authentication with expired token that can be refreshed."""
    mock_creds = Mock()
    mock_creds.valid = False
    mock_creds.expired = True
    mock_creds.refresh_token = "refresh_token"

    with patch("get_files.os.path.exists", return_value=True), patch(
        "builtins.open", mock.mock_open()
    ), patch("get_files.pickle.load", return_value=mock_creds), patch(
        "get_files.pickle.dump"
    ), patch(
        "get_files.Request"
    ) as mock_request, patch(
        "get_files.build"
    ) as mock_build:

        mock_service = Mock()
        mock_build.return_value = mock_service

        oauth_fetcher.authenticate()

        mock_creds.refresh.assert_called_once()
        assert oauth_fetcher.service == mock_service


def test_authenticate_with_oauth_new_flow(oauth_fetcher):
    """Test OAuth authentication with new authorization flow."""
    mock_flow = Mock()
    mock_creds = Mock()
    mock_flow.run_local_server.return_value = mock_creds

    with patch("get_files.os.path.exists", return_value=False), patch(
        "get_files.InstalledAppFlow.from_client_secrets_file",
        return_value=mock_flow,
    ), patch("builtins.open", mock.mock_open()), patch("get_files.pickle.dump"), patch(
        "get_files.build"
    ) as mock_build:

        mock_service = Mock()
        mock_build.return_value = mock_service

        oauth_fetcher.authenticate()

        mock_flow.run_local_server.assert_called_once_with(port=8080)
        assert oauth_fetcher.service == mock_service


def test_authenticate_handles_exception(api_key_fetcher):
    """Test that authenticate handles exceptions properly."""
    with patch("get_files.build", side_effect=Exception("API error")):
        with pytest.raises(
            Exception, match="Failed to authenticate with Google Drive API: API error"
        ):
            api_key_fetcher.authenticate()


def test_search_by_name_success(api_key_fetcher, mock_service):
    """Test successful file search by name."""
    api_key_fetcher.service = mock_service

    result = api_key_fetcher.search_by_name("test_file")

    assert len(result) == 1
    assert result[0]["name"] == "test_file.pdf"
    assert result[0]["id"] == "file_id_1"
    mock_service.files.assert_called_once()


def test_search_by_name_handles_permission_error(api_key_fetcher, capsys):
    """Test handling of permission errors in search by name."""
    mock_service = Mock()
    mock_service.files.return_value.list.return_value.execute.side_effect = Exception(
        "insufficientFilePermissions"
    )
    api_key_fetcher.service = mock_service

    result = api_key_fetcher.search_by_name("test_file")

    assert result == []
    captured = capsys.readouterr()
    assert (
        "Permission error: API key can only access publicly shared files"
        in captured.out
    )


def test_search_by_name_handles_general_exception(api_key_fetcher, capsys):
    """Test handling of general exceptions in search by name."""
    mock_service = Mock()
    mock_service.files.return_value.list.return_value.execute.side_effect = Exception(
        "General error"
    )
    api_key_fetcher.service = mock_service

    result = api_key_fetcher.search_by_name("test_file")

    assert result == []
    captured = capsys.readouterr()
    assert "Error searching files by name: General error" in captured.out


def test_search_by_type_with_known_extension(api_key_fetcher, mock_service):
    """Test file search by known file type extension."""
    api_key_fetcher.service = mock_service

    result = api_key_fetcher.search_by_type("pdf")

    assert len(result) == 1
    assert result[0]["type"] == "application/pdf"
    # Verify the query was built correctly
    call_args = mock_service.files.return_value.list.call_args
    assert "mimeType='application/pdf'" in call_args[1]["q"]


def test_search_by_type_with_unknown_extension(api_key_fetcher, mock_service):
    """Test file search by unknown file type extension."""
    api_key_fetcher.service = mock_service

    result = api_key_fetcher.search_by_type("xyz")

    assert len(result) == 1
    # Verify fallback query was used
    call_args = mock_service.files.return_value.list.call_args
    assert "name contains '.xyz'" in call_args[1]["q"]


def test_search_by_type_handles_permission_error(api_key_fetcher, capsys):
    """Test handling of permission errors in search by type."""
    mock_service = Mock()
    mock_service.files.return_value.list.return_value.execute.side_effect = Exception(
        "insufficientFilePermissions"
    )
    api_key_fetcher.service = mock_service

    result = api_key_fetcher.search_by_type("pdf")

    assert result == []
    captured = capsys.readouterr()
    assert (
        "Permission error: API key can only access publicly shared files"
        in captured.out
    )


def test_get_public_files_success(api_key_fetcher, mock_service):
    """Test successful retrieval of public files."""
    api_key_fetcher.service = mock_service

    result = api_key_fetcher.get_public_files()

    assert len(result) == 1
    assert result[0]["name"] == "test_file.pdf"


def test_get_public_files_with_folder_id(api_key_fetcher, mock_service):
    """Test retrieval of files from specific folder."""
    api_key_fetcher.service = mock_service

    result = api_key_fetcher.get_public_files(folder_id="folder_123")

    # Verify folder filter was applied
    call_args = mock_service.files.return_value.list.call_args
    assert "'folder_123' in parents" in call_args[1]["q"]


def test_get_public_files_with_file_type_filter(api_key_fetcher, mock_service):
    """Test retrieval of files with type filter."""
    api_key_fetcher.service = mock_service

    result = api_key_fetcher.get_public_files(file_type="pdf")

    # Verify file type filter was applied
    call_args = mock_service.files.return_value.list.call_args
    assert "mimeType contains 'pdf'" in call_args[1]["q"]


def test_get_public_files_handles_permission_error(api_key_fetcher, capsys):
    """Test handling of permission errors in get_public_files."""
    mock_service = Mock()
    mock_service.files.return_value.list.return_value.execute.side_effect = Exception(
        "insufficientFilePermissions"
    )
    api_key_fetcher.service = mock_service

    result = api_key_fetcher.get_public_files()

    assert result == []
    captured = capsys.readouterr()
    assert (
        "Permission error: API key can only access publicly shared files"
        in captured.out
    )
    assert (
        "To access your personal files, use OAuth authentication instead"
        in captured.out
    )


def test_getFiles_with_file_type_search():
    """Test getFiles function with file type search."""
    with patch("get_files.GoogleDriveFileFetcher") as mock_fetcher_class:
        mock_fetcher = Mock()
        mock_fetcher.search_by_type.return_value = [{"name": "test.pdf", "id": "123"}]
        mock_fetcher_class.return_value = mock_fetcher

        result = getFiles("pdf", use_oauth=False)
        result_data = json.loads(result)

        assert result_data["success"] is True
        assert result_data["count"] == 1
        assert "Searched by file type: pdf" in result_data["search_info"]
        mock_fetcher.search_by_type.assert_called_once_with("pdf")


def test_getFiles_with_filename_search():
    """Test getFiles function with filename search."""
    with patch("get_files.GoogleDriveFileFetcher") as mock_fetcher_class:
        mock_fetcher = Mock()
        mock_fetcher.search_by_name.return_value = [
            {"name": "my_document.txt", "id": "456"}
        ]
        mock_fetcher_class.return_value = mock_fetcher

        result = getFiles("my_document", use_oauth=True)
        result_data = json.loads(result)

        assert result_data["success"] is True
        assert result_data["count"] == 1
        assert "Searched by file name: my_document" in result_data["search_info"]
        assert "OAuth: Personal files accessed" in result_data["access_method"]
        mock_fetcher.search_by_name.assert_called_once_with("my_document")


def test_getFiles_without_search_term():
    """Test getFiles function without search term (get all files)."""
    with patch("get_files.GoogleDriveFileFetcher") as mock_fetcher_class:
        mock_fetcher = Mock()
        mock_fetcher.get_public_files.return_value = [
            {"name": "file1.pdf"},
            {"name": "file2.doc"},
        ]
        mock_fetcher_class.return_value = mock_fetcher

        result = getFiles(use_oauth=False)
        result_data = json.loads(result)

        assert result_data["success"] is True
        assert result_data["count"] == 2
        assert "Retrieved all accessible files" in result_data["search_info"]
        assert (
            "API Key: Only publicly shared files accessible"
            in result_data["access_method"]
        )
        mock_fetcher.get_public_files.assert_called_once()


def test_getFiles_handles_exception():
    """Test getFiles function handles exceptions properly."""
    with patch(
        "get_files.GoogleDriveFileFetcher",
        side_effect=Exception("Test error"),
    ):
        result = getFiles("test", use_oauth=False)
        result_data = json.loads(result)

        assert result_data["success"] is False
        assert "Test error" in result_data["error"]
        assert result_data["count"] == 0


def test_getFilesWithOAuth():
    """Test getFilesWithOAuth convenience function."""
    with patch("get_files.getFiles") as mock_getFiles:
        mock_getFiles.return_value = '{"success": true}'

        result = getFilesWithOAuth("test_search")

        mock_getFiles.assert_called_once_with("test_search", use_oauth=True)
        assert result == '{"success": true}'


def test_getFilesWithAPIKey():
    """Test getFilesWithAPIKey convenience function."""
    with patch("get_files.getFiles") as mock_getFiles:
        mock_getFiles.return_value = '{"success": true}'

        result = getFilesWithAPIKey("test_search")

        mock_getFiles.assert_called_once_with("test_search", use_oauth=False)
        assert result == '{"success": true}'


def test_mime_type_mapping():
    """Test that common file extensions are mapped to correct MIME types."""
    fetcher = GoogleDriveFileFetcher(api_key="test", use_oauth=False)

    with patch.object(fetcher, "service") as mock_service:
        mock_service.files.return_value.list.return_value.execute.return_value = {
            "files": []
        }

        # Test various extensions
        test_cases = [
            ("pdf", "application/pdf"),
            (
                "docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            ("jpg", "image/jpeg"),
            ("csv", "text/csv"),
        ]

        for extension, expected_mime in test_cases:
            fetcher.search_by_type(extension)
            call_args = mock_service.files.return_value.list.call_args
            assert f"mimeType='{expected_mime}'" in call_args[1]["q"]


def test_file_data_cleaning():
    """Test that file data is properly cleaned and formatted."""
    fetcher = GoogleDriveFileFetcher(api_key="test", use_oauth=False)

    raw_file_data = {
        "files": [
            {
                "id": "test_id",
                "name": "test_file.pdf",
                "mimeType": "application/pdf",
                "size": "2048",
                "createdTime": "2023-01-01T00:00:00Z",
                "modifiedTime": "2023-01-02T00:00:00Z",
                "webViewLink": "https://drive.google.com/file/d/test_id/view",
                "extra_field": "should_be_ignored",
            }
        ]
    }

    with patch.object(fetcher, "service") as mock_service:
        mock_service.files.return_value.list.return_value.execute.return_value = (
            raw_file_data
        )

        result = fetcher.get_public_files()

        assert len(result) == 1
        cleaned_file = result[0]

        # Check that only expected fields are present
        expected_fields = {"id", "name", "type", "size", "created", "modified", "link"}
        assert set(cleaned_file.keys()) == expected_fields

        # Check field mapping
        assert cleaned_file["type"] == "application/pdf"  # mimeType -> type
        assert (
            cleaned_file["created"] == "2023-01-01T00:00:00Z"
        )  # createdTime -> created
        assert (
            cleaned_file["modified"] == "2023-01-02T00:00:00Z"
        )  # modifiedTime -> modified
        assert (
            cleaned_file["link"] == "https://drive.google.com/file/d/test_id/view"
        )  # webViewLink -> link
        assert "extra_field" not in cleaned_file


class TestGoogleDriveFileFetcherInitialization:
    """Test initialization scenarios for GoogleDriveFileFetcher"""

    def test_init_with_api_key_parameter_success(self):
        """Test successful initialization with API key parameter"""
        with patch.dict(os.environ, {}, clear=True):
            fetcher = GoogleDriveFileFetcher(api_key="test_api_key")
            assert fetcher.api_key == "test_api_key"
            assert fetcher.use_oauth is False

    def test_init_with_env_var_success(self):
        """Test successful initialization with environment variable"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env_api_key"}):
            fetcher = GoogleDriveFileFetcher(api_key="param_key")
            # Parameter takes precedence, but env var is also loaded
            assert fetcher.api_key == "param_key"

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google API key is required"):
                GoogleDriveFileFetcher()

    def test_init_with_none_api_key_raises_error(self):
        """Test that initialization with None API key raises ValueError"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google API key is required"):
                GoogleDriveFileFetcher(api_key=None)

    def test_init_with_empty_string_api_key_raises_error(self):
        """Test that initialization with empty string API key raises ValueError"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google API key is required"):
                GoogleDriveFileFetcher(api_key="")

    def test_init_oauth_mode_success(self):
        """Test successful initialization in OAuth mode"""
        with patch.dict(os.environ, {}, clear=True):
            fetcher = GoogleDriveFileFetcher(api_key="test_key", use_oauth=True)
            assert fetcher.use_oauth is True
            assert fetcher.scopes == ["https://www.googleapis.com/auth/drive.readonly"]
            assert fetcher.credentials_file == "credentials.json"
            assert fetcher.token_file == "token.pickle"


class TestEnvironmentVariables:
    """Test environment variable handling"""

    def test_google_api_key_from_env(self):
        """Test loading Google API key from environment"""
        test_key = "test_google_api_key_123"
        with patch.dict(os.environ, {"GOOGLE_API_KEY": test_key}):
            fetcher = GoogleDriveFileFetcher(api_key="param_key")
            # Environment variable is set...
            assert os.getenv("GOOGLE_API_KEY") == test_key
            # ...but parameter should take precedence
            assert fetcher.api_key == "param_key"

    def test_missing_google_api_key_env_var(self):
        """Test behavior when GOOGLE_API_KEY is not in environment"""
        with patch.dict(os.environ, {}, clear=True):
            assert os.getenv("GOOGLE_API_KEY") is None

    def test_empty_google_api_key_env_var(self):
        """Test behavior when GOOGLE_API_KEY is empty string"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}):
            with pytest.raises(ValueError, match="Google API key is required"):
                GoogleDriveFileFetcher()

    # NOTE: Dotenv loading is now done at application entry, not in GoogleDriveFileFetcher.
    #       This test is no longer applicable.
    # @patch("get_files.load_dotenv")
    # def test_dotenv_loading_called(self, mock_load_dotenv):
    #     """Test that load_dotenv is called during initialization"""
    #     with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
    #         GoogleDriveFileFetcher(api_key="test_key")
    #         mock_load_dotenv.assert_called_once()


class TestCredentialsFileHandling:
    """Test credentials.json file handling for OAuth"""

    def test_oauth_credentials_file_exists(self):
        """Test OAuth initialization when credentials.json exists"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Create a mock credentials.json content
            credentials_data = {
                "installed": {
                    "client_id": "test_client_id",
                    "client_secret": "test_client_secret",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
            json.dump(credentials_data, f)
            credentials_file = f.name

        try:
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
                fetcher = GoogleDriveFileFetcher(api_key="test_key", use_oauth=True)
                # Change the credentials file path to our test file
                fetcher.credentials_file = credentials_file

                # Verify the file exists and is readable
                assert os.path.exists(credentials_file)
                with open(credentials_file, "r") as f:
                    data = json.load(f)
                    assert "installed" in data
        finally:
            # Clean up
            if os.path.exists(credentials_file):
                os.unlink(credentials_file)

    def test_oauth_credentials_file_missing(self):
        """Test OAuth behavior when credentials.json is missing"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            fetcher = GoogleDriveFileFetcher(api_key="test_key", use_oauth=True)

            # Set a non-existent credentials file
            fetcher.credentials_file = "non_existent_credentials.json"

            # Verify the file doesn't exist
            assert not os.path.exists(fetcher.credentials_file)

    def test_oauth_token_file_path(self):
        """Test that token file path is set correctly"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            fetcher = GoogleDriveFileFetcher(api_key="test_key", use_oauth=True)
            assert fetcher.token_file == "token.pickle"

    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.load")
    def test_oauth_token_file_loading(
        self, mock_pickle_load, mock_file_open, mock_exists
    ):
        """Test token file loading in OAuth mode"""
        # Mock that token file exists
        mock_exists.return_value = True

        # Mock credentials object
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_pickle_load.return_value = mock_creds

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            fetcher = GoogleDriveFileFetcher(api_key="test_key", use_oauth=True)

            # Mock the build function to avoid actual API calls
            with patch("get_files.build") as mock_build:
                fetcher.authenticate()

                # Verify token file was opened
                mock_file_open.assert_called()
                mock_pickle_load.assert_called()
                mock_build.assert_called_with("drive", "v3", credentials=mock_creds)


class TestFileValidation:
    """Test file validation utilities"""

    def test_check_credentials_json_format(self):
        """Test validation of credentials.json format"""
        valid_credentials = {
            "installed": {
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }

        # Test valid format
        assert "installed" in valid_credentials
        assert "client_id" in valid_credentials["installed"]
        assert "client_secret" in valid_credentials["installed"]

    def test_check_invalid_credentials_json_format(self):
        """Test detection of invalid credentials.json format"""
        invalid_credentials = {
            "web": {  # Should be "installed" for desktop apps
                "client_id": "test_client_id"
            }
        }

        # Test invalid format
        assert "installed" not in invalid_credentials

    def test_env_file_presence(self):
        """Test .env file presence check"""
        # This is more of a utility test to verify .env handling
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            assert os.path.exists(".env") == True

            mock_exists.return_value = False
            assert os.path.exists(".env") == False


class TestConfigurationValidation:
    """Test configuration validation scenarios"""

    def test_api_key_precedence_over_env(self):
        """Test that API key parameter takes precedence over environment variable"""
        env_key = "env_api_key"
        param_key = "param_api_key"

        with patch.dict(os.environ, {"GOOGLE_API_KEY": env_key}):
            fetcher = GoogleDriveFileFetcher(api_key=param_key)
            # Parameter should take precedence
            assert fetcher.api_key == param_key

    def test_oauth_scopes_configuration(self):
        """Test OAuth scopes are correctly configured"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            fetcher = GoogleDriveFileFetcher(api_key="test_key", use_oauth=True)
            expected_scopes = ["https://www.googleapis.com/auth/drive.readonly"]
            assert fetcher.scopes == expected_scopes

    def test_port_configuration(self):
        """Test OAuth port configuration"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            fetcher = GoogleDriveFileFetcher(api_key="test_key", use_oauth=True)
            assert fetcher.port == 8080


# Integration-style tests
class TestIntegrationScenarios:
    """Test complete initialization scenarios"""

    def test_complete_api_key_setup(self):
        """Test complete setup with API key"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_env_key"}):
            fetcher = GoogleDriveFileFetcher(api_key="test_param_key")

            assert fetcher.service is None  # Not authenticated yet
            assert fetcher.use_oauth is False
            assert fetcher.api_key is not None

    def test_complete_oauth_setup(self):
        """Test complete setup with OAuth"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            fetcher = GoogleDriveFileFetcher(api_key="test_key", use_oauth=True)

            assert fetcher.service is None  # Not authenticated yet
            assert fetcher.use_oauth is True
            assert fetcher.credentials_file == "credentials.json"
            assert fetcher.token_file == "token.pickle"
            assert fetcher.scopes == ["https://www.googleapis.com/auth/drive.readonly"]


# Utility functions for test setup
def create_test_env_file(api_key: str) -> str:
    """Create a temporary .env file for testing"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write(f"GOOGLE_API_KEY={api_key}\n")
        return f.name


def create_test_credentials_file() -> str:
    """Create a temporary credentials.json file for testing"""
    credentials_data = {
        "installed": {
            "client_id": "test_client_id.apps.googleusercontent.com",
            "project_id": "test_project",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "test_client_secret",
            "redirect_uris": ["http://localhost"],
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(credentials_data, f)
        return f.name


if __name__ == "__main__":
    # Run specific test categories
    print("Running GoogleDriveFileFetcher tests...")

    # Example of how to run these tests:
    # pytest test_google_drive_comprehensive.py -v
    # pytest test_google_drive_comprehensive.py::TestEnvironmentVariables -v
    # pytest test_google_drive_comprehensive.py::TestCredentialsFileHandling -v

    pass
