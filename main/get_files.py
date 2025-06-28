import os
import logging
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_FILE = "service_account.json"  # Path to your service account key file


class GoogleDriveFileFetcher:
    def __init__(self, service_account_file: Optional[str] = None):
        self.service_account_file = service_account_file or SERVICE_ACCOUNT_FILE
        self.service = None
        self.scopes = ["https://www.googleapis.com/auth/drive.readonly"]

    def authenticate(self) -> None:
        try:
            creds = Credentials.from_service_account_file(
                self.service_account_file, scopes=self.scopes
            )
            self.service = build("drive", "v3", credentials=creds)
            logger.info("Authenticated with service account.")
        except Exception as e:
            raise Exception(f"Failed to authenticate with Service Account: {str(e)}")

    def search_by_name(
        self, filename: str, max_results: int = 100
    ) -> List[Dict[str, Any]]:
        if not self.service:
            self.authenticate()
        query = f"name contains '{filename}' and trashed=false"
        try:
            results = (
                self.service.files()
                .list(
                    q=query,
                    pageSize=max_results,
                    fields="files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink)",
                )
                .execute()
            )
            files = results.get("files", [])
            return [
                {
                    "id": f.get("id"),
                    "name": f.get("name"),
                    "type": f.get("mimeType"),
                    "size": f.get("size"),
                    "created": f.get("createdTime"),
                    "modified": f.get("modifiedTime"),
                    "link": f.get("webViewLink"),
                }
                for f in files
            ]
        except Exception as e:
            logger.error(f"Error searching files by name: {str(e)}")
            return []

    def search_by_type(
        self, file_type: str, max_results: int = 100
    ) -> List[Dict[str, Any]]:
        if not self.service:
            self.authenticate()
        mime_type_map = {
            "pdf": "application/pdf",
            "doc": "application/msword",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "xls": "application/vnd.ms-excel",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "ppt": "application/vnd.ms-powerpoint",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "txt": "text/plain",
            "csv": "text/csv",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "mp4": "video/mp4",
            "zip": "application/zip",
        }
        mime_type = mime_type_map.get(file_type.lower())
        if mime_type:
            query = f"mimeType='{mime_type}' and trashed=false"
        else:
            query = f"name contains '.{file_type}' and trashed=false"
        try:
            results = (
                self.service.files()
                .list(
                    q=query,
                    pageSize=max_results,
                    fields="files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink)",
                )
                .execute()
            )
            files = results.get("files", [])
            return [
                {
                    "id": f.get("id"),
                    "name": f.get("name"),
                    "type": f.get("mimeType"),
                    "size": f.get("size"),
                    "created": f.get("createdTime"),
                    "modified": f.get("modifiedTime"),
                    "link": f.get("webViewLink"),
                }
                for f in files
            ]
        except Exception as e:
            logger.error(f"Error searching files by type: {str(e)}")
            return []

    def get_public_files(
        self,
        folder_id: Optional[str] = None,
        file_type: Optional[str] = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.service:
            self.authenticate()
        query_parts = []
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")
        if file_type:
            query_parts.append(f"mimeType contains '{file_type}'")
        query_parts.append("trashed=false")
        query = " and ".join(query_parts)
        try:
            results = (
                self.service.files()
                .list(
                    q=query,
                    pageSize=max_results,
                    fields="files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink)",
                )
                .execute()
            )
            files = results.get("files", [])
            return [
                {
                    "id": f.get("id"),
                    "name": f.get("name"),
                    "type": f.get("mimeType"),
                    "size": f.get("size"),
                    "created": f.get("createdTime"),
                    "modified": f.get("modifiedTime"),
                    "link": f.get("webViewLink"),
                }
                for f in files
            ]
        except Exception as e:
            logger.error(f"Error fetching files: {str(e)}")
            return []


def getFiles(search_term: str = None) -> dict:
    """
    Returns a list of Google Drive files matching a search term or file type.
    Uses Service Account for headless (no-browser) access.
    """
    logger.info(f"Called getFiles with search_term='{search_term}'")
    try:
        fetcher = GoogleDriveFileFetcher()
        files = []
        search_info = ""
        if search_term:
            common_extensions = [
                "pdf",
                "doc",
                "docx",
                "xls",
                "xlsx",
                "ppt",
                "pptx",
                "txt",
                "csv",
                "jpg",
                "png",
                "mp4",
                "zip",
            ]
            if search_term.lower() in common_extensions:
                logger.debug(f"Searching by file type: {search_term}")
                files = fetcher.search_by_type(search_term)
                search_info = f"Searched by file type: {search_term}"
            else:
                logger.debug(f"Searching by file name: {search_term}")
                files = fetcher.search_by_name(search_term)
                search_info = f"Searched by file name: {search_term}"
        else:
            logger.debug("No search_term provided, fetching all accessible files.")
            files = fetcher.get_public_files()
            search_info = "Retrieved all accessible files"
        logger.info(f"Files found: {len(files)} | Info: {search_info}")
        for f in files:
            logger.debug(f"Found file: {f['name']} (type: {f['type']})")
        return {
            "success": True,
            "files": files,
            "count": len(files),
            "search_info": search_info,
            "access_method": "Service Account: Only files/folders shared with this account are accessible.",
        }
    except Exception as e:
        logger.error(f"Exception in getFiles: {e}", exc_info=True)
        return {
            "success": False,
            "error/exception": str(e),
            "files": [],
            "count": 0,
        }


if __name__ == "__main__":
    print("=== Google Drive File Fetcher (Service Account, Headless) ===")
    print(getFiles("docx"))
    print(getFiles("xlsx"))
