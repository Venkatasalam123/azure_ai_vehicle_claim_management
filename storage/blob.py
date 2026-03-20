import os
import json
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from dotenv import load_dotenv
from pathlib import Path

# Load .env from parent directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

def get_blob_service_client():
    # Authenticate to Key Vault
    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET")
    )

    secret_client = SecretClient(
        vault_url=os.getenv("KEYVAULT_URL"),
        credential=credential
    )

    blob_conn_str = secret_client.get_secret(
        "blob-connection-string"
    ).value

    return BlobServiceClient.from_connection_string(blob_conn_str)


def upload_raw_document(
    local_file_path: str,
    claim_id: str,
    original_filename: str = None
):
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(
        "raw-documents"
    )

    # Use original filename if provided, otherwise use basename of local file
    file_name = original_filename if original_filename else os.path.basename(local_file_path)
    blob_path = f"{claim_id}/{file_name}"
    print(f"[upload_raw_document] Uploading to blob path: {blob_path}")

    with open(local_file_path, "rb") as data:
        container_client.upload_blob(
            name=blob_path,
            data=data,
            overwrite=True
        )

    print(f"[upload_raw_document] Successfully uploaded to: {blob_path}")
    return blob_path

def upload_processed_result(
    result_data: dict,
    claim_id: str,
    result_type: str
):
    """
    result_type: 'document' or 'image'
    """
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(
        "processed-results"
    )

    blob_path = f"{claim_id}/{result_type}_result.json"

    container_client.upload_blob(
        name=blob_path,
        data=json.dumps(result_data, indent=2),
        overwrite=True
    )

    return blob_path


def get_raw_document(claim_id: str, file_name: str) -> bytes:
    """
    Retrieve raw document/image from blob storage.
    Returns the file data as bytes, or None if not found.
    """
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(
            "raw-documents"
        )

        blob_path = f"{claim_id}/{file_name}"
        print(f"[get_raw_document] Looking for blob at: {blob_path}")
        
        blob_client = container_client.get_blob_client(blob_path)
        
        if blob_client.exists():
            print(f"[get_raw_document] Blob found, downloading...")
            blob_data = blob_client.download_blob()
            data = blob_data.readall()
            print(f"[get_raw_document] Successfully retrieved {len(data)} bytes")
            return data
        else:
            print(f"[get_raw_document] Blob does not exist at: {blob_path}")
            # Try to list blobs in the claim_id folder to see what's there
            try:
                blobs = list(container_client.list_blobs(name_starts_with=f"{claim_id}/"))
                print(f"[get_raw_document] Available blobs in {claim_id}/:")
                for blob in blobs[:10]:  # Limit to first 10
                    print(f"  - {blob.name}")
            except Exception as list_error:
                print(f"[get_raw_document] Error listing blobs: {list_error}")
            return None
    except Exception as e:
        print(f"[get_raw_document] Error retrieving raw document: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_processed_result(
    claim_id: str,
    result_type: str
) -> dict:
    """
    Retrieve processed result from blob storage.
    result_type: 'document' or 'image'
    Returns the result data as a dict, or None if not found.
    """
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(
            "processed-results"
        )

        blob_path = f"{claim_id}/{result_type}_result.json"
        
        blob_client = container_client.get_blob_client(blob_path)
        
        if blob_client.exists():
            blob_data = blob_client.download_blob()
            result_data = json.loads(blob_data.readall())
            return result_data
        else:
            return None
    except Exception as e:
        print(f"Error retrieving processed result: {e}")
        return None


def _ensure_container_exists(container_name: str):
    """Ensure a container exists, create it if it doesn't."""
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.get_container_properties()
    except ResourceNotFoundError:
        # Container doesn't exist, create it
        try:
            container_client.create_container()
            print(f"Created container: {container_name}")
        except Exception as e:
            print(f"Error creating container {container_name}: {e}")
            raise
    except Exception as e:
        # Other errors - log but don't fail
        print(f"Error checking container {container_name}: {e}")
    return container_client


def store_processed_claim(claim_id: str, invoice_number: str = None, claim_data: dict = None):
    """
    Store processed claim metadata for duplicate detection and listing.
    """
    from datetime import datetime
    container_client = _ensure_container_exists("processed-claims")
    
    claim_metadata = {
        "claim_id": claim_id,
        "invoice_number": invoice_number,
        "processed_at": datetime.utcnow().isoformat(),
        "claim_data": claim_data or {}
    }
    
    # Store by claim_id
    blob_path = f"{claim_id}/metadata.json"
    container_client.upload_blob(
        name=blob_path,
        data=json.dumps(claim_metadata, indent=2),
        overwrite=True
    )
    
    # Also store by invoice_number for quick lookup if invoice exists
    if invoice_number:
        invoice_path = f"invoices/{invoice_number}/{claim_id}.json"
        container_client.upload_blob(
            name=invoice_path,
            data=json.dumps({"claim_id": claim_id, "invoice_number": invoice_number}, indent=2),
            overwrite=True
        )
    
    return blob_path


def check_duplicate_invoice(invoice_number: str) -> list:
    """
    Check if invoice number already exists in processed claims.
    Returns list of claim_ids with this invoice number.
    """
    if not invoice_number:
        return []
    
    try:
        container_client = _ensure_container_exists("processed-claims")
        
        # Look for claims with this invoice number
        invoice_path_prefix = f"invoices/{invoice_number.upper()}/"
        matching_claims = []
        
        try:
            blobs = list(container_client.list_blobs(name_starts_with=invoice_path_prefix))
            for blob in blobs:
                blob_client = container_client.get_blob_client(blob.name)
                if blob_client.exists():
                    blob_data = blob_client.download_blob()
                    claim_info = json.loads(blob_data.readall())
                    matching_claims.append(claim_info.get("claim_id"))
        except Exception as e:
            print(f"Error checking duplicate invoice: {e}")
        
        return matching_claims
    except Exception as e:
        print(f"Error checking duplicate invoice: {e}")
        return []


def list_processed_claims(limit: int = 50) -> list:
    """
    List all processed claims, most recent first.
    """
    try:
        container_client = _ensure_container_exists("processed-claims")
        
        claims = []
        # List all claim metadata files
        blobs = list(container_client.list_blobs(name_starts_with=""))
        
        for blob in blobs:
            # Only process metadata.json files
            if blob.name.endswith("/metadata.json"):
                try:
                    blob_client = container_client.get_blob_client(blob.name)
                    if blob_client.exists():
                        blob_data = blob_client.download_blob()
                        claim_metadata = json.loads(blob_data.readall())
                        claims.append(claim_metadata)
                except Exception as e:
                    print(f"Error reading claim metadata {blob.name}: {e}")
        
        # Sort by processed_at (most recent first)
        claims.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
        
        return claims[:limit]
    except Exception as e:
        print(f"Error listing processed claims: {e}")
        return []

