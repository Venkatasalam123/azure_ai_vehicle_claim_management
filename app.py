import os
import uuid
from flask import Flask, request, render_template, redirect, url_for, flash, send_file, Response
import base64
import io
from pathlib import Path
from dotenv import load_dotenv
import html
from datetime import datetime

from processors.document_processor import analyze_claim_document
from processors.language_processor import analyze_claim_language, analyze_sentiment
from processors.vision_processor import analyze_claim_image
from processors.classifier import classify_claim_image
from processors.object_detection import detect_claim_damage
from processors.chat_bot import process_chat_message

from storage.blob import upload_raw_document, upload_processed_result, get_raw_document, get_processed_result, store_processed_claim, check_duplicate_invoice, list_processed_claims
from processors.search_indexer import index_document_result, index_image_result
from processors.search_query import search_claims
from processors.query_processor import process_query
from flask import jsonify, session
import re
import json
from agents.claim_process_agents import process_claim
from agents.observability import init_observability

# Load .env from parent directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

init_observability()

def safe_id(value: str) -> str:
    """
    Convert a string into a safe Azure AI Search document key.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def extract_claim_data_from_text(full_text: str) -> dict:
    """
    Extract claim data from document text using regex patterns.
    Returns dict with claim_amount, vehicle_type, etc.
    """
    extracted = {}
    
    # Extract claim amount (look for patterns like "4.5L", "450000", "₹4.5L", "$4500", etc.)
    amount_patterns = [
        r'(?:claim|amount|total|sum|value)[\s:]*[₹$]?\s*([\d.,]+[LK]?)',
        r'[₹$]\s*([\d.,]+[LK]?)',
        r'([\d.,]+)\s*(?:lakh|lac|L|thousand|k)',
    ]
    for pattern in amount_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            extracted['claim_amount'] = match.group(1).strip()
            break
    
    # Extract vehicle type
    vehicle_types = ['car', 'truck', 'bus', 'motorcycle', 'bike', 'scooter', 'auto', 'vehicle']
    for vtype in vehicle_types:
        if re.search(r'\b' + vtype + r'\b', full_text, re.IGNORECASE):
            extracted['vehicle_type'] = vtype
            break
    
    return extracted


def extract_bill_data_from_text(full_text: str) -> dict:
    """
    Extract bill/repair data from document text.
    Returns dict with repair_amount, repair_date, invoice_number, etc.
    """
    extracted = {}
    
    # Extract repair amount
    repair_patterns = [
        r'(?:repair|bill|total|amount|charge)[\s:]*[₹$]?\s*([\d.,]+)',
        r'[₹$]\s*([\d.,]+)',
    ]
    for pattern in repair_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            extracted['repair_amount'] = match.group(1).strip().replace(',', '')
            break
    
    # Extract date (look for common date formats including month names)
    # First try to find "Service Date:" or "Repair Date:" labels
    service_date_patterns = [
        r'(?:service|repair)\s+date\s*:?\s*(\d{1,2}[-/]\w+[-/]\d{2,4})',
        r'(?:service|repair)\s+date\s*:?\s*(\d{1,2}\s+\w+\s+\d{2,4})',
        r'(?:service|repair)\s+date\s*:?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        r'(?:service|repair)\s+date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
    ]
    for pattern in service_date_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            date_value = match.group(1).strip()
            extracted['repair_date'] = date_value
            # Also set service_date if the pattern matched "service date"
            if 'service' in pattern.lower():
                extracted['service_date'] = date_value
            break
    
    # If no service/repair date found, look for any date
    if 'repair_date' not in extracted:
        date_patterns = [
            r'(\d{1,2}[-/]\w+[-/]\d{2,4})',  # 02-Feb-2026, 02/Feb/2026
            r'(\d{1,2}\s+\w+\s+\d{2,4})',    # 02 Feb 2026
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # 02-02-2026, 02/02/2026
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',    # 2026-02-02
        ]
        for pattern in date_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                extracted['repair_date'] = match.group(1).strip()
                break
    
    # Extract invoice number (common patterns: "Invoice #12345", "INV-12345", "Invoice No: 12345", etc.)
    invoice_patterns = [
        r'invoice\s*(?:no|number|#|num)[\s:]*([A-Z0-9\-]+)',
        r'inv[\.\s]*no[\s:]*([A-Z0-9\-]+)',
        r'invoice[\s:]+([A-Z0-9\-]+)',
        r'inv[\s\-]+([A-Z0-9\-]+)',
        r'bill\s*(?:no|number|#)[\s:]*([A-Z0-9\-]+)',
    ]
    for pattern in invoice_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            invoice_num = match.group(1).strip()
            # Only accept if it looks like an invoice number (has alphanumeric chars)
            if len(invoice_num) >= 3 and re.search(r'[A-Z0-9]', invoice_num, re.IGNORECASE):
                extracted['invoice_number'] = invoice_num.upper()
                break
    
    return extracted


def construct_vision_signals(image_result: dict, classification_result: dict = None, detection_result: dict = None) -> dict:
    """
    Construct vision_signals dict from image analysis results.
    """
    vision_signals = {
        "vision_confidence": 0.5,  # Default
        "damage_severity": "low",  # low, medium, high
        "semantic_alignment": True,
        "damage_present": False
    }
    
    # Set vision_confidence from classification if available
    if classification_result and classification_result.get('confidence'):
        vision_signals["vision_confidence"] = classification_result['confidence']
    
    # Determine if damage is present
    damage_present = False
    
    # Check classification result
    if classification_result and classification_result.get('tag_name'):
        tag_name = classification_result.get('tag_name', '').lower()
        # Damage is present if tag contains "damage" or "damaged" but not "not damaged"
        if ('damage' in tag_name or 'damaged' in tag_name) and 'not' not in tag_name:
            damage_present = True
    
    # Check detection result (object detection confirms damage)
    if detection_result and detection_result.get('detections'):
        detections = detection_result['detections']
        if len(detections) > 0:
            damage_present = True
    
    vision_signals["damage_present"] = damage_present
    
    # Calculate damage_severity based on detection results
    if detection_result and detection_result.get('detections'):
        detections = detection_result['detections']
        if len(detections) > 0:
            confidences = [d.get('probability', 0) for d in detections]
            max_confidence = max(confidences)
            num_detections = len(detections)
            
            # Determine severity based on confidence and number of detections
            if num_detections >= 3 or max_confidence >= 0.9:
                # High severity: multiple damage areas OR very high confidence single detection
                vision_signals["damage_severity"] = "high"
            elif num_detections >= 2 or max_confidence >= 0.7:
                # Medium severity: multiple detections OR medium-high confidence
                vision_signals["damage_severity"] = "medium"
            else:
                # Low severity: single low-medium confidence detection
                vision_signals["damage_severity"] = "low"
    elif classification_result and classification_result.get('confidence'):
        # If only classification available (no object detection), use classification confidence
        class_confidence = classification_result.get('confidence', 0)
        if class_confidence >= 0.8:
            vision_signals["damage_severity"] = "high"
        elif class_confidence >= 0.6:
            vision_signals["damage_severity"] = "medium"
        else:
            vision_signals["damage_severity"] = "low"
    
    # Semantic alignment: check if caption and classification align
    caption = image_result.get('caption', '').lower()
    if classification_result and classification_result.get('tag_name'):
        tag = classification_result['tag_name'].lower()
        if 'damage' in caption or 'damage' in tag:
            vision_signals["semantic_alignment"] = True
    
    return vision_signals

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# File size limits (in bytes)
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB

# ------------------------------------------------
# Home page (Upload + Search)
# ------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ------------------------------------------------
# Unified Upload Endpoint (Document or Image)
# ------------------------------------------------
@app.route("/upload-claim", methods=["POST"])
def upload_claim():
    """
    Unified endpoint to upload either a document (PDF) or image.
    After extraction, calls the claim processing agent.
    """
    try:
        file = request.files.get("file")
        if not file or file.filename == '':
            flash("Please select a file to upload.", "error")
            return redirect(url_for("home"))

        # Determine file type
        file_ext = os.path.splitext(file.filename.lower())[1]
        is_pdf = file_ext == '.pdf'
        is_image = file_ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        
        if not (is_pdf or is_image):
            flash("Only PDF documents or image files (JPG, JPEG, PNG, GIF, BMP) are allowed.", "error")
            return redirect(url_for("home"))

        # Validate file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        max_size = MAX_DOCUMENT_SIZE if is_pdf else MAX_IMAGE_SIZE
        if file_size > max_size:
            max_mb = max_size / (1024*1024)
            flash(f"File size exceeds the maximum limit of {max_mb:.0f} MB.", "error")
            return redirect(url_for("home"))
        
        if file_size == 0:
            flash("The uploaded file is empty.", "error")
            return redirect(url_for("home"))

        claim_id = request.form.get("claim_id") or f"claim-{uuid.uuid4().hex[:8]}"
        safe_file = safe_id(file.filename)
        file_type = "document" if is_pdf else "image"
        unique_claim_id = f"{claim_id}-{file_type}-{safe_file}"
        
        # Create unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        try:
            # Store raw file
            upload_raw_document(file_path, unique_claim_id, file.filename)

            # Initialize variables for agent
            extracted_claim_data = {}
            vision_signals = {}
            bill_data = None
            doc_result = None
            image_result = None

            if is_pdf:
                # Process document
                doc_result = analyze_claim_document(file_path)
                
                # Extract full text
                full_text = " ".join(
                    line
                    for page in doc_result.get("pages", [])
                    for line in page.get("lines", [])
                )
                
                # Extract claim data from text
                extracted_claim_data = extract_claim_data_from_text(full_text)
                
                # Try to extract bill data (repair bill)
                bill_data = extract_bill_data_from_text(full_text)
                if not bill_data or not bill_data.get('repair_amount'):
                    bill_data = None  # Only set if we found bill data
                
                # Analyze with Language service
                language_result = {}
                if full_text.strip():
                    try:
                        language_result = analyze_claim_language(full_text)
                        if language_result:
                            doc_result["key_phrases"] = language_result.get("key_phrases", [])
                            doc_result["entities"] = language_result.get("entities", [])
                    except Exception as e:
                        print(f"Language analysis error: {e}")
                
                # Add document URL
                doc_result["document_url"] = url_for(
                    "serve_document",
                    claim_id=unique_claim_id,
                    filename=file.filename
                )
                
                # Store and index
                upload_processed_result(doc_result, unique_claim_id, "document")
                index_document_result(
                    claim_id=unique_claim_id,
                    source_file=file.filename,
                    doc_result=doc_result
                )
                
            else:  # Image
                # Process image
                image_result = analyze_claim_image(file_path)
                image_result["image_url"] = url_for(
                    "serve_image",
                    claim_id=unique_claim_id,
                    filename=file.filename
                )
                
                # Extract claim data from image (vehicle type from caption)
                if image_result.get('vehicle_type'):
                    extracted_claim_data['vehicle_type'] = image_result['vehicle_type']
                
                # Classify image
                classification_result = None
                try:
                    classification_result = classify_claim_image(file_path)
                except Exception as e:
                    print(f"Classification error: {e}")
                
                # Detect damage if classified as damaged
                detection_result = None
                is_damaged = False
                if classification_result and classification_result.get('tag_name'):
                    tag_name = classification_result.get('tag_name', '').lower()
                    if ('damage' in tag_name or 'damaged' in tag_name) and 'not' not in tag_name:
                        is_damaged = True
                        try:
                            detection_result = detect_claim_damage(file_path)
                        except Exception as e:
                            print(f"Object detection error: {e}")
                
                # Store classification and detection
                if classification_result:
                    image_result['classification'] = classification_result
                if detection_result:
                    image_result['object_detection'] = detection_result
                
                # Construct vision signals
                vision_signals = construct_vision_signals(
                    image_result, 
                    classification_result, 
                    detection_result
                )
                
                # Store and index
                upload_processed_result(image_result, unique_claim_id, "image")
                index_image_result(
                    claim_id=unique_claim_id,
                    source_file=file.filename,
                    image_result=image_result
                )

            # Clean up local file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Call claim processing agent
            try:
                agent_result = process_claim(
                    extracted_claim_data=extracted_claim_data,
                    vision_signals=vision_signals,
                    bill_data=bill_data
                )
                
                # Build success message with agent results
                success_msg = f"✅ File '{html.escape(file.filename)}' uploaded and processed successfully!"
                success_msg += "<br><br><strong><i class='bi bi-robot'></i> Claim Processing Results:</strong><br>"
                
                # Add claim understanding
                if agent_result.get('claim_understanding'):
                    cu = agent_result['claim_understanding']
                    success_msg += f"<div class='alert alert-info'><strong>Claim Understanding:</strong><br>"
                    if cu.get('normalized_claim'):
                        nc = cu['normalized_claim']
                        success_msg += f"Claim Amount: {nc.get('claim_amount', 'N/A')}<br>"
                        success_msg += f"Vehicle Type: {nc.get('vehicle_type', 'N/A')}<br>"
                    if cu.get('missing_fields'):
                        success_msg += f"Missing Fields: {', '.join(cu['missing_fields'])}<br>"
                    if cu.get('issues'):
                        success_msg += f"Issues: {', '.join(cu['issues'])}<br>"
                    success_msg += "</div>"
                
                # Add evidence confidence
                if agent_result.get('evidence_confidence'):
                    ec = agent_result['evidence_confidence']
                    success_msg += f"<div class='alert alert-warning'><strong>Evidence Confidence:</strong><br>"
                    success_msg += f"Strength: {ec.get('evidence_strength', 'N/A')}<br>"
                    success_msg += f"Confidence Score: {ec.get('confidence_score', 0):.2%}<br>"
                    success_msg += "</div>"
                
                # Add final decision
                if agent_result.get('final_decision'):
                    fd = agent_result['final_decision']
                    decision_type = fd.get('decision', 'N/A')
                    decision_color = 'success' if decision_type.lower() == 'approve' else 'danger' if decision_type.lower() == 'reject' else 'warning'
                    success_msg += f"<div class='alert alert-{decision_color}'><strong>Final Decision:</strong> {decision_type}<br>"
                    if fd.get('reason'):
                        success_msg += f"Reason: {fd['reason']}<br>"
                    success_msg += "</div>"
                
                flash(success_msg, "success")
                
            except Exception as agent_error:
                print(f"Agent processing error: {agent_error}")
                # Still show success for upload, but note agent error
                flash(
                    f"✅ File '{html.escape(file.filename)}' uploaded successfully, but agent processing encountered an error: {str(agent_error)}",
                    "warning"
                )

            return redirect(url_for("home"))
        
        except Exception as e:
            # Clean up local file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
    
    except Exception as e:
        return render_template("error.html", message=str(e))


# ------------------------------------------------
# Upload CLAIM DOCUMENT (PDF) - Legacy endpoint (kept for backward compatibility)
# ------------------------------------------------
@app.route("/upload-document", methods=["POST"])
def upload_document():
    try:
        file = request.files.get("file")
        if not file or file.filename == '':
            flash("Please select a file to upload.", "error")
            return redirect(url_for("home"))

        # Validate file extension
        if not file.filename.lower().endswith('.pdf'):
            flash("Only PDF files are allowed for document uploads.", "error")
            return redirect(url_for("home"))

        # Validate file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_DOCUMENT_SIZE:
            flash(f"File size exceeds the maximum limit of {MAX_DOCUMENT_SIZE / (1024*1024):.0f} MB.", "error")
            return redirect(url_for("home"))
        
        if file_size == 0:
            flash("The uploaded file is empty.", "error")
            return redirect(url_for("home"))

        claim_id = request.form.get("claim_id") or f"claim-{uuid.uuid4().hex[:8]}"
        safe_file = safe_id(file.filename)
        unique_claim_id = f"{claim_id}-document-{safe_file}"
        
        # Create unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        try:
            # 1. Store raw document (use original filename for easier retrieval)
            upload_raw_document(file_path, unique_claim_id, file.filename)

            # 2. Process with Document Intelligence
            doc_result = analyze_claim_document(file_path)

            # 3. Analyze extracted text with Azure AI Language (key phrases & entities)
            try:
                full_text = " ".join(
                    line
                    for page in doc_result.get("pages", [])
                    for line in page.get("lines", [])
                )
            except Exception:
                full_text = ""

            language_result = {}
            if full_text.strip():
                try:
                    language_result = analyze_claim_language(full_text)
                except Exception as e:
                    # Don't fail the whole flow if language analysis has issues
                    print(f"Language analysis error: {e}")

            if language_result:
                doc_result["key_phrases"] = language_result.get("key_phrases", [])
                doc_result["entities"] = language_result.get("entities", [])

            # 4. Add document URL (served from blob via Flask route) so it can be indexed and used in search results
            doc_result["document_url"] = url_for(
                "serve_document",
                claim_id=unique_claim_id,
                filename=file.filename
            )

            # 5. Store processed result
            upload_processed_result(doc_result, unique_claim_id, "document")

            # 6. Index into Azure AI Search
            index_document_result(
                claim_id=unique_claim_id,
                source_file=file.filename,
                doc_result=doc_result
            )

            # 7. Store claim_id in session for claim processing
            if 'uploaded_claims' not in session:
                session['uploaded_claims'] = {}
            session['uploaded_claims']['document_claim_id'] = unique_claim_id
            session['uploaded_claims']['base_claim_id'] = claim_id
            session.modified = True

            # Clean up local file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Build rich success message with preview
            success_msg = f"✅ Document '{html.escape(file.filename)}' uploaded and indexed successfully!"

            if full_text:
                content_preview = full_text[:500] + "..." if len(full_text) > 500 else full_text
                success_msg += "<br><br><strong><i class='bi bi-align-left'></i> Content Preview:</strong><br>"
                success_msg += (
                    "<div style='background:#f8f9fa; padding:1rem; border-radius:8px; "
                    "font-family:monospace; font-size:0.9rem; max-height:200px; overflow-y:auto;'>"
                    f"{html.escape(content_preview)}"
                    "</div>"
                )

            if language_result:
                key_phrases = language_result.get("key_phrases") or []
                entities = language_result.get("entities") or []

                if key_phrases:
                    success_msg += "<br><strong><i class='bi bi-lightbulb'></i> Key Phrases:</strong><br>"
                    success_msg += "<div class='d-flex flex-wrap gap-1 mb-2'>"
                    for phrase in key_phrases:
                        if phrase:
                            success_msg += (
                                f"<span class='badge bg-light text-dark border me-1 mb-1'>"
                                f"{html.escape(str(phrase))}</span>"
                            )
                    success_msg += "</div>"

                if entities:
                    success_msg += "<br><strong><i class='bi bi-collection'></i> Entities:</strong><br>"
                    success_msg += "<div class='d-flex flex-wrap gap-1'>"
                    for ent in entities:
                        if ent:
                            success_msg += (
                                f"<span class='badge bg-warning text-dark me-1 mb-1'>"
                                f"{html.escape(str(ent))}</span>"
                            )
                    success_msg += "</div>"

            flash(success_msg, "success")
            return redirect(url_for("home"))
        
        except Exception as e:
            # Clean up local file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
    
    except Exception as e:
        return render_template("error.html", message=str(e))


# ------------------------------------------------
# Serve Document from Blob Storage
# ------------------------------------------------
@app.route("/serve-document/<claim_id>/<path:filename>", methods=["GET"])
def serve_document(claim_id: str, filename: str):
    """Serve PDF document from blob storage for inline display/download"""
    try:
        from urllib.parse import unquote
        # Decode URL-encoded filename
        filename = unquote(filename)
        print(f"Attempting to serve document: claim_id={claim_id}, filename={filename}")
        doc_data = get_raw_document(claim_id, filename)
        if not doc_data:
            print(f"Document not found in blob storage: {claim_id}/{filename}")
            return f"Document not found: {claim_id}/{filename}", 404

        # Serve as downloadable PDF
        return Response(
            doc_data,
            mimetype="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        print(f"Error serving document: {str(e)}")
        return f"Error: {str(e)}", 500


# ------------------------------------------------
# Upload CLAIM IMAGE (Damage / ID)
# ------------------------------------------------
@app.route("/upload-image", methods=["POST"])
def upload_image():
    try:
        file = request.files.get("file")
        if not file or file.filename == '':
            flash("Please select an image file to upload.", "error")
            return redirect(url_for("home"))

        # Validate file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            flash("Only image files (JPG, JPEG, PNG, GIF, BMP) are allowed.", "error")
            return redirect(url_for("home"))

        # Validate file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_IMAGE_SIZE:
            flash(f"File size exceeds the maximum limit of {MAX_IMAGE_SIZE / (1024*1024):.0f} MB.", "error")
            return redirect(url_for("home"))
        
        if file_size == 0:
            flash("The uploaded file is empty.", "error")
            return redirect(url_for("home"))

        claim_id = request.form.get("claim_id") or f"claim-{uuid.uuid4().hex[:8]}"
        safe_file = safe_id(file.filename)
        unique_claim_id = f"{claim_id}-image-{safe_file}"

        # Create unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        try:
            # 1. Store raw image (use original filename for easier retrieval)
            upload_raw_document(file_path, unique_claim_id, file.filename)

            # 2. Process with Computer Vision
            image_result = analyze_claim_image(file_path)
            # Add image URL (served from blob via Flask route) so it can be indexed and used in search results
            image_result["image_url"] = url_for(
                "serve_image",
                claim_id=unique_claim_id,
                filename=file.filename
            )
            caption = image_result.get('caption', '')
            print(f"Caption: {caption}")

            # 3. Classify image with Custom Vision
            classification_result = None
            try:
                classification_result = classify_claim_image(file_path)
            except Exception as e:
                print(f"Classification error: {e}")
                # Continue even if classification fails

            # 4. Detect damage areas with Object Detection (ONLY if classified as damaged)
            detection_result = None
            is_damaged = False
            
            # Only proceed with object detection if classification indicates damage
            if classification_result and classification_result.get('tag_name'):
                tag_name = classification_result.get('tag_name', '').lower()
                # Check if classification indicates damage (exclude "not damaged" cases)
                # Only call object detection if tag is "damaged" or contains "damage" but NOT "not damaged"
                if ('damage' in tag_name or 'damaged' in tag_name) and 'not' not in tag_name:
                    is_damaged = True
                    # Only call detect_claim_damage if damage is detected
                    try:
                        detection_result = detect_claim_damage(file_path)
                    except Exception as e:
                        print(f"Object detection error: {e}")
                        # Continue even if detection fails
                # If not damaged, detection_result remains None and detect_claim_damage is never called

            # 5. Store processed result (include classification and detection if available)
            if classification_result:
                image_result['classification'] = classification_result
            if detection_result:
                image_result['object_detection'] = detection_result
            upload_processed_result(image_result, unique_claim_id, "image")

            # 6. Index into Azure AI Search
            index_image_result(
                claim_id=unique_claim_id,
                source_file=file.filename,
                image_result=image_result
            )

            # 7. Store claim_id in session for claim processing
            if 'uploaded_claims' not in session:
                session['uploaded_claims'] = {}
            session['uploaded_claims']['image_claim_id'] = unique_claim_id
            session['uploaded_claims']['base_claim_id'] = claim_id
            session.modified = True

            # Clean up local file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Create success message with caption, classification, and detection
            success_msg = f"✅ Image '{file.filename}' uploaded and indexed successfully!"
            
            # Display vehicle type and color if available
            vehicle_type = image_result.get('vehicle_type')
            color = image_result.get('color')
            if vehicle_type or color:
                success_msg += "<br><br><strong><i class='bi bi-car-front'></i> Vehicle Information:</strong><br>"
                if vehicle_type:
                    success_msg += f"<span class='badge bg-info me-2'><i class='bi bi-car-front-fill'></i> Type: {html.escape(vehicle_type).title()}</span>"
                if color:
                    success_msg += f"<span class='badge bg-secondary'><i class='bi bi-palette-fill'></i> Color: {html.escape(color).title()}</span>"
            
            if caption:
                success_msg += f"<br><br><strong><i class='bi bi-chat-quote'></i> Image Caption:</strong><br><em>{caption}</em>"
            
            if classification_result and classification_result.get('tag_name'):
                tag_name = classification_result['tag_name']
                confidence = classification_result['confidence']
                success_msg += f"<br><br><strong><i class='bi bi-tags'></i> Classification:</strong><br>"
                success_msg += f"<span class='badge bg-primary'>{tag_name}</span> "
                success_msg += f"<small>(Confidence: {confidence:.1%})</small>"
                
                # Show message if not damaged
                if not is_damaged:
                    success_msg += f"<br><br><div class='alert alert-info'><i class='bi bi-info-circle'></i> No damage detected. Object detection skipped.</div>"
            
            if detection_result:
                detections = detection_result.get('detections', [])
                if detections:
                    # Store detection data and image info for inline display
                    detection_data = {
                        'claim_id': unique_claim_id,
                        'filename': file.filename,
                        'detections': detections,
                        'best_detection': detection_result.get('best_detection')
                    }
                    # Escape JSON for HTML attribute
                    detection_data_json = html.escape(json.dumps(detection_data))
                    success_msg += f"<div id='image-display-container' data-claim-id='{unique_claim_id}' data-filename='{html.escape(file.filename)}' data-detections='{detection_data_json}' style='margin-top: 1rem;'></div>"
            
            flash(success_msg, "success")
            return redirect(url_for("home"))
        
        except Exception as e:
            # Clean up local file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
    
    except Exception as e:
        return render_template("error.html", message=str(e))


# ------------------------------------------------
# Upload Document API (JSON response)
# ------------------------------------------------
@app.route("/api/upload-document", methods=["POST"])
def api_upload_document():
    """Upload and process document, return JSON response"""
    try:
        file = request.files.get("file")
        if not file or file.filename == '':
            return jsonify({"success": False, "error": "Please select a file to upload."}), 400

        # Validate file extension
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"success": False, "error": "Only PDF files are allowed for document uploads."}), 400

        # Validate file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_DOCUMENT_SIZE:
            return jsonify({"success": False, "error": f"File size exceeds the maximum limit of {MAX_DOCUMENT_SIZE / (1024*1024):.0f} MB."}), 400
        
        if file_size == 0:
            return jsonify({"success": False, "error": "The uploaded file is empty."}), 400

        claim_id = request.form.get("claim_id") or f"claim-{uuid.uuid4().hex[:8]}"
        safe_file = safe_id(file.filename)
        unique_claim_id = f"{claim_id}-document-{safe_file}"
        
        # Create unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        try:
            # 1. Store raw document
            upload_raw_document(file_path, unique_claim_id, file.filename)

            # 2. Process with Document Intelligence
            doc_result = analyze_claim_document(file_path)

            # 3. Analyze extracted text with Azure AI Language (key phrases & entities)
            try:
                full_text = " ".join(
                    line
                    for page in doc_result.get("pages", [])
                    for line in page.get("lines", [])
                )
            except Exception:
                full_text = ""

            language_result = {}
            if full_text.strip():
                try:
                    language_result = analyze_claim_language(full_text)
                except Exception as e:
                    print(f"Language analysis error: {e}")

            if language_result:
                doc_result["key_phrases"] = language_result.get("key_phrases", [])
                doc_result["entities"] = language_result.get("entities", [])

            # 4. Add document URL
            doc_result["document_url"] = url_for(
                "serve_document",
                claim_id=unique_claim_id,
                filename=file.filename
            )

            # 5. Store processed result
            upload_processed_result(doc_result, unique_claim_id, "document")

            # 6. Index into Azure AI Search
            index_document_result(
                claim_id=unique_claim_id,
                source_file=file.filename,
                doc_result=doc_result
            )

            # 7. Store claim_id in session for claim processing
            if 'uploaded_claims' not in session:
                session['uploaded_claims'] = {}
            session['uploaded_claims']['document_claim_id'] = unique_claim_id
            session['uploaded_claims']['base_claim_id'] = claim_id
            session.modified = True

            # Clean up local file
            if os.path.exists(file_path):
                os.remove(file_path)

            return jsonify({
                "success": True,
                "claim_id": unique_claim_id,
                "filename": file.filename,
                "result": {
                    "key_phrases": doc_result.get("key_phrases", []),
                    "entities": doc_result.get("entities", []),
                    "content_preview": full_text[:500] + "..." if len(full_text) > 500 else full_text
                }
            })
        
        except Exception as e:
            # Clean up local file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------------------------------
# Upload Image API (JSON response)
# ------------------------------------------------
@app.route("/api/upload-image", methods=["POST"])
def api_upload_image():
    """Upload and process image, return JSON response"""
    try:
        file = request.files.get("file")
        if not file or file.filename == '':
            return jsonify({"success": False, "error": "Please select an image file to upload."}), 400

        # Validate file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({"success": False, "error": "Only image files (JPG, JPEG, PNG, GIF, BMP) are allowed."}), 400

        # Validate file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_IMAGE_SIZE:
            return jsonify({"success": False, "error": f"File size exceeds the maximum limit of {MAX_IMAGE_SIZE / (1024*1024):.0f} MB."}), 400
        
        if file_size == 0:
            return jsonify({"success": False, "error": "The uploaded file is empty."}), 400

        claim_id = request.form.get("claim_id") or f"claim-{uuid.uuid4().hex[:8]}"
        safe_file = safe_id(file.filename)
        unique_claim_id = f"{claim_id}-image-{safe_file}"

        # Create unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        try:
            # 1. Store raw image
            upload_raw_document(file_path, unique_claim_id, file.filename)

            # 2. Process with Computer Vision
            image_result = analyze_claim_image(file_path)
            image_result["image_url"] = url_for(
                "serve_image",
                claim_id=unique_claim_id,
                filename=file.filename
            )
            caption = image_result.get('caption', '')

            # 3. Classify image with Custom Vision
            classification_result = None
            try:
                classification_result = classify_claim_image(file_path)
            except Exception as e:
                print(f"Classification error: {e}")

            # 4. Detect damage areas with Object Detection (ONLY if classified as damaged)
            detection_result = None
            is_damaged = False
            
            if classification_result and classification_result.get('tag_name'):
                tag_name = classification_result.get('tag_name', '').lower()
                if ('damage' in tag_name or 'damaged' in tag_name) and 'not' not in tag_name:
                    is_damaged = True
                    try:
                        detection_result = detect_claim_damage(file_path)
                    except Exception as e:
                        print(f"Object detection error: {e}")

            # 5. Store processed result
            if classification_result:
                image_result['classification'] = classification_result
            if detection_result:
                image_result['object_detection'] = detection_result
            upload_processed_result(image_result, unique_claim_id, "image")

            # 6. Index into Azure AI Search
            index_image_result(
                claim_id=unique_claim_id,
                source_file=file.filename,
                image_result=image_result
            )

            # 7. Store claim_id in session for claim processing
            if 'uploaded_claims' not in session:
                session['uploaded_claims'] = {}
            session['uploaded_claims']['image_claim_id'] = unique_claim_id
            session['uploaded_claims']['base_claim_id'] = claim_id
            session.modified = True

            # Clean up local file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Build result
            result = {
                "filename": file.filename,
                "caption": caption,
                "vehicle_type": image_result.get('vehicle_type'),
                "color": image_result.get('color')
            }
            
            if classification_result:
                result["classification"] = {
                    "tag_name": classification_result.get('tag_name'),
                    "confidence": classification_result.get('confidence')
                }
            
            if detection_result:
                detections = detection_result.get('detections', [])
                result["object_detection"] = {
                    "detections_count": len(detections),
                    "best_detection": detection_result.get('best_detection'),
                    "detections": detections,  # Include all detections for bounding box display
                    "claim_id": unique_claim_id  # Include claim_id for image serving
                }

            return jsonify({
                "success": True,
                "claim_id": unique_claim_id,
                "result": result
            })
        
        except Exception as e:
            # Clean up local file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------------------------------
# Process Claim with Agents
# ------------------------------------------------
@app.route("/process-claim", methods=["POST"])
def process_claim_route():
    """
    Process claim using the claim processing agents.
    Retrieves document and/or image results from session and processes them.
    """
    try:
        # Get uploaded claim IDs from session
        uploaded_claims = session.get('uploaded_claims', {})
        document_claim_id = uploaded_claims.get('document_claim_id')
        image_claim_id = uploaded_claims.get('image_claim_id')
        
        if not document_claim_id and not image_claim_id:
            return jsonify({
                "success": False,
                "error": "No documents or images uploaded. Please upload at least one document or image first."
            }), 400
        
        # Retrieve processed results
        doc_result = None
        image_result = None
        
        if document_claim_id:
            doc_result = get_processed_result(document_claim_id, "document")
        
        if image_claim_id:
            image_result = get_processed_result(image_claim_id, "image")
        
        if not doc_result and not image_result:
            return jsonify({
                "success": False,
                "error": "Could not retrieve processed results. Please try uploading again."
            }), 400
        
        # Extract data for process_claim function
        extracted_claim_data = {}
        vision_signals = {}
        bill_data = None
        
        # Extract claim data from document
        if doc_result:
            # Extract full text from document
            full_text = ""
            try:
                full_text = " ".join(
                    line
                    for page in doc_result.get("pages", [])
                    for line in page.get("lines", [])
                )
            except Exception:
                pass
            
            if full_text:
                # Extract claim data from text
                extracted_claim_data = extract_claim_data_from_text(full_text)
                
                # Try to extract bill data (repair bill)
                bill_data = extract_bill_data_from_text(full_text)
                if not bill_data or not bill_data.get('repair_amount'):
                    bill_data = None
                
                # Extract invoice number from bill_data or document text
                invoice_number = None
                if bill_data and bill_data.get('invoice_number'):
                    invoice_number = bill_data.get('invoice_number')
                elif not invoice_number:
                    # Try to extract invoice number directly from text if not in bill_data
                    invoice_patterns = [
                        r'invoice\s*(?:no|number|#|num)[\s:]*([A-Z0-9\-]+)',
                        r'inv[\.\s]*no[\s:]*([A-Z0-9\-]+)',
                        r'invoice[\s:]+([A-Z0-9\-]+)',
                        r'inv[\s\-]+([A-Z0-9\-]+)',
                        r'bill\s*(?:no|number|#)[\s:]*([A-Z0-9\-]+)',
                    ]
                    for pattern in invoice_patterns:
                        match = re.search(pattern, full_text, re.IGNORECASE)
                        if match:
                            invoice_num = match.group(1).strip()
                            if len(invoice_num) >= 3 and re.search(r'[A-Z0-9]', invoice_num, re.IGNORECASE):
                                invoice_number = invoice_num.upper()
                                break
                
                # Check for duplicate invoice number BEFORE processing
                if invoice_number:
                    duplicate_claims = check_duplicate_invoice(invoice_number)
                    if duplicate_claims:
                        # Return rejected status without calling agents or storing
                        return jsonify({
                            "success": True,
                            "result": {
                                "claim_understanding": {
                                    "normalized_claim": extracted_claim_data,
                                    "missing_fields": [],
                                    "issues": [f"Duplicate invoice number: {invoice_number}"]
                                },
                                "evidence_confidence": {
                                    "evidence_strength": "low",
                                    "confidence_score": 0.0,
                                    "recommendations": [f"Duplicate invoice detected. Existing claim(s): {', '.join(duplicate_claims[:3])}"]
                                },
                                "final_decision": {
                                    "status": "REJECTED",
                                    "reason": f"Duplicate invoice number '{invoice_number}' detected. This invoice has already been processed in claim(s): {', '.join(duplicate_claims[:3])}. Claim rejected to prevent duplicate processing."
                                }
                            },
                            "duplicate": True,
                            "invoice_number": invoice_number,
                            "existing_claims": duplicate_claims,
                            "rejected_by_duplicate": True
                        })
        
        # Extract vision signals from image
        if image_result:
            classification_result = image_result.get('classification')
            detection_result = image_result.get('object_detection')
            vision_signals = construct_vision_signals(
                image_result,
                classification_result,
                detection_result
            )
            
            # Prioritize vehicle_type from image (more specific) over document extraction (which may be generic "vehicle")
            if image_result.get('vehicle_type'):
                # Image result is more specific, so always use it if available
                # Only skip if document already has a specific type (not generic "vehicle")
                current_type = extracted_claim_data.get('vehicle_type', '').lower()
                if not current_type or current_type == 'vehicle':
                    extracted_claim_data['vehicle_type'] = image_result['vehicle_type']
        
        # If no data extracted, provide defaults
        if not extracted_claim_data:
            extracted_claim_data = {}
        if not vision_signals:
            vision_signals = {
                "vision_confidence": 0.5,
                "damage_severity": "low",
                "semantic_alignment": True,
                "damage_present": False
            }
        
        # Debug: Print what's being sent to agents
        print(f"\n=== SENDING TO AGENTS ===")
        print(f"extracted_claim_data: {json.dumps(extracted_claim_data, indent=2)}")
        print(f"vision_signals: {json.dumps(vision_signals, indent=2)}")
        print(f"bill_data: {json.dumps(bill_data, indent=2) if bill_data else None}")
        
        # Get invoice number if available
        invoice_number = None
        if bill_data and bill_data.get('invoice_number'):
            invoice_number = bill_data.get('invoice_number')
        
        # Get claim IDs for storing
        uploaded_claims = session.get('uploaded_claims', {})
        document_claim_id = uploaded_claims.get('document_claim_id')
        image_claim_id = uploaded_claims.get('image_claim_id')
        claim_id = uploaded_claims.get('base_claim_id') or (document_claim_id or image_claim_id or f"claim-{uuid.uuid4().hex[:8]}")
        
        # Process claim using agents
        try:
            result = process_claim(
                extracted_claim_data=extracted_claim_data,
                vision_signals=vision_signals,
                bill_data=bill_data
            )
            
            # Store processed claim metadata
            claim_metadata = {
                "claim_amount": extracted_claim_data.get('claim_amount'),
                "vehicle_type": extracted_claim_data.get('vehicle_type'),
                "decision": result.get('final_decision', {}).get('status') or result.get('final_decision', {}).get('decision', 'Unknown'),
                "processed_at": datetime.utcnow().isoformat()
            }
            store_processed_claim(claim_id, invoice_number, claim_metadata)
            
            return jsonify({
                "success": True,
                "result": result,
                "claim_id": claim_id,
                "invoice_number": invoice_number,
                "inputs": {
                    "extracted_claim_data": extracted_claim_data,
                    "vision_signals": vision_signals,
                    "bill_data": bill_data
                }
            })
        except Exception as agent_error:
            print(f"Agent processing error: {agent_error}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": f"Error processing claim with agents: {str(agent_error)}"
            }), 500
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Error processing claim: {str(e)}"
        }), 500


# ------------------------------------------------
# Search Claims (Legacy)
# ------------------------------------------------
@app.route("/search", methods=["GET"])
def search():
    try:
        query = request.args.get("q", "").strip()
        claim_id = request.args.get("claim_id", "").strip()

        # Validate empty search query
        if not query:
            flash("Please enter a search query.", "warning")
            return render_template(
                "results.html",
                query="",
                claim_id=claim_id,
                results=[]
            )

        results = search_claims(query=query, claim_id=claim_id if claim_id else None)
        if not results:
            flash(f"No results found for '{query}'. Try different keywords.", "info")

        return render_template(
            "results.html",
            query=query,
            claim_id=claim_id,
            results=results
        )
    except Exception as e:
        return render_template("error.html", message=str(e))


# ------------------------------------------------
# Comprehensive Query Endpoint (Full AI Pipeline)
# ------------------------------------------------
@app.route("/query", methods=["POST"])
def query():
    """
    Process queries through the full AI pipeline:
    - Multimodal input (speech + text)
    - Translation
    - Input content safety
    - CLU intent & entities
    - Deterministic router
    - Azure AI Search / Custom QA / Grounded RAG
    - Output content safety
    """
    try:
        # Check if it's JSON (text) or multipart (audio)
        if request.is_json:
            data = request.get_json()
            text_input = data.get("text", "").strip()
            audio_bytes = None
        else:
            text_input = request.form.get("text", "").strip()
            audio_file = request.files.get("audio")
            audio_bytes = None
            if audio_file:
                audio_bytes = audio_file.read()
        
        if not text_input and not audio_bytes:
            return jsonify({
                "error": "Either text or audio input is required"
            }), 400
        
        # Process through full pipeline
        result = process_query(
            text_input=text_input if text_input else None,
            audio_bytes=audio_bytes,
            track_steps=True
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "result": {
                "action": "error",
                "message": f"An error occurred: {str(e)}"
            },
            "steps": [],
            "services_used": []
        }), 500


# ------------------------------------------------
# Serve Image from Blob Storage
# ------------------------------------------------
@app.route("/serve-image/<claim_id>/<path:filename>", methods=["GET"])
def serve_image(claim_id: str, filename: str):
    """Serve image from blob storage for display"""
    try:
        from urllib.parse import unquote
        # Decode URL-encoded filename
        filename = unquote(filename)
        print(f"Attempting to serve image: claim_id={claim_id}, filename={filename}")
        image_data = get_raw_document(claim_id, filename)
        if not image_data:
            print(f"Image not found in blob storage: {claim_id}/{filename}")
            return f"Image not found: {claim_id}/{filename}", 404
        
        # Determine MIME type
        file_ext = os.path.splitext(filename.lower())[1]
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp'
        }
        mime_type = mime_types.get(file_ext, 'image/jpeg')
        
        return Response(image_data, mimetype=mime_type)
    except Exception as e:
        print(f"Error serving image: {str(e)}")
        return f"Error: {str(e)}", 500


# ------------------------------------------------
# View Image with Bounding Boxes
# ------------------------------------------------
@app.route("/view-image/<claim_id>/<filename>", methods=["GET"])
def view_image_with_boxes(claim_id: str, filename: str):
    try:
        # Get the image from blob storage
        image_data = get_raw_document(claim_id, filename)
        if not image_data:
            return render_template("error.html", message="Image not found")
        
        # Get processed result to get bounding boxes
        processed_result = get_processed_result(claim_id, "image")
        if not processed_result:
            return render_template("error.html", message="Processed result not found")
        
        detection_result = processed_result.get('object_detection')
        detections = []
        if detection_result:
            detections = detection_result.get('detections', [])
        
        # Convert image to base64 for display
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Determine image MIME type
        file_ext = os.path.splitext(filename.lower())[1]
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp'
        }
        mime_type = mime_types.get(file_ext, 'image/jpeg')
        image_data_uri = f"data:{mime_type};base64,{image_base64}"
        
        return render_template(
            "image_viewer.html",
            image_data_uri=image_data_uri,
            filename=filename,
            detections=detections,
            claim_id=claim_id
        )
    except Exception as e:
        return render_template("error.html", message=str(e))


# ------------------------------------------------
# Query Results Page
# ------------------------------------------------
@app.route("/query-results", methods=["GET"])
def query_results():
    """Display query results with processing steps and services used"""
    try:
        data_json = request.args.get("data", "{}")
        data = json.loads(data_json)
        
        return render_template(
            "query_results.html",
            result=data.get("result", {}),
            plan=data.get("plan", {}),
            steps=data.get("steps", []),
            services_used=data.get("services_used", []),
            original_text=data.get("original_text", ""),
            processed_text=data.get("processed_text", "")
        )
    except Exception as e:
        return render_template("error.html", message=str(e))


# ------------------------------------------------
# Chat Bot Endpoint with Conversation History
# ------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle chat messages with conversation history maintained in session.
    """
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        clear_history = data.get("clear_history", False)
        
        if not user_message:
            return jsonify({
                "success": False,
                "error": "Message is required"
            }), 400
        
        # Initialize conversation history in session if not exists
        if "chat_history" not in session or clear_history:
            session["chat_history"] = []
        
        # Get current conversation history
        conversation_history = session["chat_history"].copy()
        
        # Process the chat message
        chat_result = process_chat_message(
            user_message=user_message,
            conversation_history=conversation_history,
            use_rag=True
        )
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_message})
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": chat_result["response"]})
        
        # Update session (keep last 10 messages to avoid session size issues)
        session["chat_history"] = conversation_history[-10:]
        
        return jsonify({
            "success": True,
            "response": chat_result["response"],
            "context_used": chat_result.get("context_used", False)
        })
        
    except Exception as e:
        print(f"Error processing chat message: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/chat/clear", methods=["POST"])
def clear_chat_history():
    """Clear the conversation history."""
    try:
        session["chat_history"] = []
        return jsonify({
            "success": True,
            "message": "Conversation history cleared"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ------------------------------------------------
# List Processed Claims
# ------------------------------------------------
@app.route("/claims", methods=["GET"])
def list_claims():
    """List all processed claims"""
    try:
        claims = list_processed_claims(limit=100)
        return render_template("claims_list.html", claims=claims)
    except Exception as e:
        return render_template("error.html", message=str(e))


# ------------------------------------------------
# Feedback Submission with Sentiment Analysis
# ------------------------------------------------
@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    """
    Accept user feedback and analyze sentiment using Azure AI Language service.
    """
    try:
        data = request.get_json()
        feedback_text = data.get("feedback", "").strip()
        
        if not feedback_text:
            return jsonify({
                "success": False,
                "error": "Feedback text is required"
            }), 400
        
        # Analyze sentiment
        sentiment_result = analyze_sentiment(feedback_text)
        
        # Log feedback (you could also store this in a database)
        print(f"[FEEDBACK] Text: {feedback_text}")
        print(f"[FEEDBACK] Sentiment: {sentiment_result['sentiment']}")
        print(f"[FEEDBACK] Confidence: {sentiment_result['overall_score']:.2%}")
        
        return jsonify({
            "success": True,
            "sentiment": sentiment_result["sentiment"],
            "confidence_scores": sentiment_result["confidence_scores"],
            "overall_score": sentiment_result["overall_score"],
            "message": f"Thank you for your feedback! We detected {sentiment_result['sentiment']} sentiment."
        })
        
    except Exception as e:
        print(f"Error processing feedback: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ------------------------------------------------
# Run locally
# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
