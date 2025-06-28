"""Unit tests for the GoogleDriveFileFetcher class."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from unittest.mock import patch, Mock

from main.get_files import GoogleDriveFileFetcher, getFiles


@pytest.fixture
def mock_service():
    """Create a mock Google Drive service with a files().list().execute() chain."""
    service = Mock()
    files_api = Mock()
    list_api = Mock()
    # Chained call: service.files().list().execute()
    files_api.list.return_value = list_api
    list_api.execute.return_value = {
        "files": [
            {
                "id": "abc123",
                "name": "test_file.xlsx",
                "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "size": "2048",
                "createdTime": "2023-01-01T12:00:00Z",
                "modifiedTime": "2023-01-02T12:00:00Z",
                "webViewLink": "https://drive.google.com/file/d/abc123/view",
            }
        ]
    }
    service.files.return_value = files_api
    return service


@pytest.fixture
def fetcher():
    return GoogleDriveFileFetcher(service_account_file="fake.json")


# ----- Tests -----


def test_authenticate_service_account_success(fetcher):
    """Test authenticate() with service account succeeds."""
    with patch(
        "main.get_files.Credentials.from_service_account_file"
    ) as mock_creds, patch("main.get_files.build") as mock_build:
        mock_creds.return_value = Mock()
        mock_build.return_value = "mock_service"
        fetcher.authenticate()
        assert fetcher.service == "mock_service"


def test_search_by_name_success(fetcher, mock_service):
    """Test search_by_name returns formatted file data."""
    fetcher.service = mock_service
    results = fetcher.search_by_name("test_file")
    assert len(results) == 1
    assert results[0]["name"] == "test_file.xlsx"
    assert (
        results[0]["type"]
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert results[0]["id"] == "abc123"


def test_search_by_type_known_extension(fetcher, mock_service):
    """Test search_by_type with known extension uses MIME type query."""
    fetcher.service = mock_service
    files = fetcher.search_by_type("xlsx")
    assert len(files) == 1
    assert files[0]["name"] == "test_file.xlsx"
    call = mock_service.files.return_value.list.call_args
    assert (
        "mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'"
        in call[1]["q"]
    )


def test_search_by_type_unknown_extension(fetcher, mock_service):
    """Test search_by_type with unknown extension falls back to filename."""
    fetcher.service = mock_service
    files = fetcher.search_by_type("unknownext")
    assert len(files) == 1
    call = mock_service.files.return_value.list.call_args
    assert "name contains '.unknownext'" in call[1]["q"]


def test_get_public_files_success(fetcher, mock_service):
    """Test get_public_files returns files and applies filters."""
    fetcher.service = mock_service
    files = fetcher.get_public_files(folder_id="folder123", file_type="xlsx")
    assert len(files) == 1
    call = mock_service.files.return_value.list.call_args
    assert "'folder123' in parents" in call[1]["q"]
    assert "mimeType contains 'xlsx'" in call[1]["q"]


def test_search_error_handling(fetcher):
    """Test error handling in search methods."""
    with patch.object(fetcher, "service") as service:
        # Simulate error in execute
        service.files.return_value.list.return_value.execute.side_effect = Exception(
            "Fake error"
        )
        assert fetcher.search_by_name("x") == []
        assert fetcher.search_by_type("xlsx") == []
        assert fetcher.get_public_files() == []


def test_getFiles_by_type(monkeypatch):
    mock_fetcher = Mock()
    mock_fetcher.search_by_type.return_value = [
        {"name": "file.pdf", "type": "application/pdf"},
    ]
    # Patch *class* so instantiation in getFiles uses your mock
    monkeypatch.setattr(
        "main.get_files.GoogleDriveFileFetcher", lambda *a, **kw: mock_fetcher
    )
    result = getFiles("pdf")
    assert result["success"]


def test_getFiles_by_name(monkeypatch):
    """Test getFiles function with name search."""
    mock_fetcher = Mock()
    mock_fetcher.search_by_name.return_value = [
        {
            "name": "notes.docx",
            "type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
    ]
    monkeypatch.setattr(
        "main.get_files.GoogleDriveFileFetcher", lambda *a, **kw: mock_fetcher
    )
    result = getFiles("notes")
    assert result["success"]
    assert result["count"] == 1
    assert "file name" in result["search_info"]




def test_getFiles_exception(monkeypatch):
    """Test getFiles handles exceptions gracefully."""
    monkeypatch.setattr(
        "main.get_files.GoogleDriveFileFetcher",
        lambda *a, **kw: (_ for _ in ()).throw(Exception("boom")),
    )
    result = getFiles("will_fail")
    assert not result["success"]
    assert "boom" in result["error/exception"]
