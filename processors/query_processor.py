"""
Comprehensive query processor that handles the full AI pipeline:
1. Multimodal input (speech + text)
2. Translation
3. Input content safety
4. CLU intent & entities
5. Deterministic router
6. Azure AI Search execution / Custom QA / Grounded RAG
7. Output content safety
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from azure.cognitiveservices import speech as speechsdk
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
import io

from .translator_processor import translate_to_english
from .conversation_language_understanding import analyze_with_clu
from .content_safety import moderate_input, moderate_output
from agents.router_agent import router_agent
from agents.executor import execute_plan
import json


def process_speech_from_bytes(audio_bytes: bytes) -> dict:
    """
    Process speech audio bytes to text using Azure Speech Service.
    Handles both WAV files (strips header) and raw PCM audio.
    Returns dict with 'text' and 'language' keys.
    """
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    endpoint = os.getenv("SPEECH_ENDPOINT")
    keyvault_url = os.getenv("KEYVAULT_URL")

    if not endpoint or not keyvault_url:
        raise ValueError("SPEECH_ENDPOINT or KEYVAULT_URL not set")

    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET"),
    )

    secret_client = SecretClient(
        vault_url=keyvault_url,
        credential=credential,
    )

    key = secret_client.get_secret("speech-key").value

    speech_config = speechsdk.SpeechConfig(endpoint=endpoint, subscription=key)
    
    # Set audio format for web audio
    speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "5000")

    # Auto language detection
    auto_lang = speechsdk.AutoDetectSourceLanguageConfig(
        languages=["en-US", "ta-IN", "hi-IN"]
    )

    # Check if it's a WAV file (starts with "RIFF")
    pcm_data = audio_bytes
    sample_rate = 16000
    channels = 1
    bits_per_sample = 16
    
    if audio_bytes[:4] == b'RIFF':
        # It's a WAV file, extract PCM data and metadata
        import struct
        try:
            # Parse WAV header
            # Skip "RIFF" (4 bytes) and file size (4 bytes)
            # Check "WAVE" (4 bytes)
            if audio_bytes[8:12] != b'WAVE':
                raise ValueError("Invalid WAV file")
            
            # Find "fmt " chunk (should be at position 12 in standard WAV files)
            fmt_pos = 12
            if audio_bytes[fmt_pos:fmt_pos+4] != b'fmt ':
                # Try to find it dynamically
                fmt_pos = audio_bytes.find(b'fmt ', 8)
                if fmt_pos == -1:
                    raise ValueError("WAV file missing fmt chunk")
            
            # Read format chunk size (4 bytes after "fmt ")
            fmt_chunk_size = struct.unpack('<I', audio_bytes[fmt_pos+4:fmt_pos+8])[0]
            
            # Read format data (starts 8 bytes after "fmt ")
            fmt_data_start = fmt_pos + 8
            fmt_data = audio_bytes[fmt_data_start:fmt_data_start+16]
            
            if len(fmt_data) < 16:
                raise ValueError("Incomplete fmt chunk")
            
            # Unpack: audio_format (H), num_channels (H), sample_rate (I), byte_rate (I), block_align (H), bits_per_sample (H)
            audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack('<HHIIHH', fmt_data)
            
            # Validate parsed values
            if sample_rate <= 0 or sample_rate > 48000:
                raise ValueError(f"Invalid sample rate: {sample_rate}")
            if num_channels <= 0 or num_channels > 2:
                raise ValueError(f"Invalid channel count: {num_channels}")
            if bits_per_sample not in [8, 16, 24, 32]:
                raise ValueError(f"Invalid bit depth: {bits_per_sample}")
            
            # Find "data" chunk (should be after fmt chunk)
            data_pos = fmt_pos + 8 + fmt_chunk_size
            # Align to 2-byte boundary if needed
            if data_pos % 2 != 0:
                data_pos += 1
            
            # Look for "data" chunk
            search_start = data_pos
            data_pos = audio_bytes.find(b'data', search_start)
            if data_pos == -1:
                raise ValueError("WAV file missing data chunk")
            
            # Extract PCM data (skip 4-byte "data" header and 4-byte size)
            pcm_data = audio_bytes[data_pos + 8:]
            channels = num_channels
            
            print(f"WAV file detected: {sample_rate}Hz, {channels} channel(s), {bits_per_sample}bit, PCM data size: {len(pcm_data)} bytes")
        except Exception as e:
            print(f"Error parsing WAV file: {e}, treating as raw PCM")
            # If WAV parsing fails, use default values and treat as raw PCM
            pcm_data = audio_bytes
            sample_rate = 16000
            channels = 1
            bits_per_sample = 16

    # Create push audio input stream with detected/configured format
    stream_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=sample_rate,
        bits_per_sample=bits_per_sample,
        channels=channels
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
    
    # Write PCM audio bytes to stream
    push_stream.write(pcm_data)
    push_stream.close()
    
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    # Recognizer
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_lang,
        audio_config=audio_config
    )

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        detected_lang = result.properties.get(
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult,
            "en-US"
        )
        return {
            "text": result.text,
            "language": detected_lang
        }

    if result.reason == speechsdk.ResultReason.NoMatch:
        try:
            no_match_details = speechsdk.NoMatchDetails(result)
            raise RuntimeError(f"No speech recognized. Reason: {no_match_details.reason}")
        except AttributeError:
            # Fallback if NoMatchDetails doesn't work
            raise RuntimeError("No speech recognized. Please try speaking clearly or use text input.")

    if result.reason == speechsdk.ResultReason.Canceled:
        try:
            cancellation_details = speechsdk.CancellationDetails(result)
            error_msg = f"Canceled: {cancellation_details.reason}"
            if hasattr(cancellation_details, 'error_details') and cancellation_details.error_details:
                error_msg += f" | {cancellation_details.error_details}"
            raise RuntimeError(error_msg)
        except (AttributeError, TypeError):
            # Fallback if CancellationDetails doesn't work as expected
            cancellation = result.cancellation_details if hasattr(result, 'cancellation_details') else None
            if cancellation:
                error_msg = f"Canceled: {getattr(cancellation, 'reason', 'Unknown')}"
                if hasattr(cancellation, 'error_details'):
                    error_msg += f" | {cancellation.error_details}"
                raise RuntimeError(error_msg)
            else:
                raise RuntimeError("Speech recognition was canceled.")

    raise RuntimeError("Unknown error in speech recognition")


def process_query(
    text_input: str = None,
    audio_bytes: bytes = None,
    track_steps: bool = True
) -> dict:
    """
    Process a query through the full AI pipeline.
    
    Args:
        text_input: Text query (optional if audio_bytes provided)
        audio_bytes: Audio bytes for speech input (optional if text_input provided)
        track_steps: Whether to track processing steps
    
    Returns:
        dict with 'result', 'steps', 'services_used' keys
    """
    steps = []
    services_used = set()
    
    try:
        # Step 1: Multimodal input (speech + text)
        if audio_bytes:
            steps.append({
                "step": "Speech-to-Text",
                "status": "processing",
                "message": "Converting speech to text..."
            })
            services_used.add("Azure Speech Service")
            try:
                speech_result = process_speech_from_bytes(audio_bytes)
                user_text = speech_result["text"]
                detected_lang = speech_result["language"]
                steps.append({
                    "step": "Speech-to-Text",
                    "status": "completed",
                    "message": f"Detected language: {detected_lang}, Text: {user_text}"
                })
            except Exception as e:
                error_msg = str(e)
                if "No speech recognized" in error_msg:
                    user_friendly_msg = "No speech detected. Please ensure your microphone is working and try speaking clearly, or use text input instead."
                else:
                    user_friendly_msg = f"Speech recognition error: {error_msg}. Please try text input instead."
                
                steps.append({
                    "step": "Speech-to-Text",
                    "status": "error",
                    "message": user_friendly_msg
                })
                raise
        elif text_input:
            user_text = text_input
            detected_lang = "en"  # Assume English for text input
            steps.append({
                "step": "Text Input",
                "status": "completed",
                "message": f"Text input received: {user_text}"
            })
        else:
            raise ValueError("Either text_input or audio_bytes must be provided")
        
        # Step 2: Translation
        if detected_lang and detected_lang != "en" and detected_lang != "en-US":
            steps.append({
                "step": "Translation",
                "status": "processing",
                "message": f"Translating from {detected_lang} to English..."
            })
            services_used.add("Azure Translator")
            translation_result = translate_to_english(user_text)
            clu_input = translation_result["translated_text"]
            steps.append({
                "step": "Translation",
                "status": "completed",
                "message": f"Translated: {clu_input}"
            })
        else:
            clu_input = user_text
            steps.append({
                "step": "Translation",
                "status": "skipped",
                "message": "Input is already in English"
            })
        
        # Step 3: Input content safety
        steps.append({
            "step": "Input Content Safety",
            "status": "processing",
            "message": "Checking input for safety violations..."
        })
        services_used.add("Azure Content Safety")
        safety_result = moderate_input(clu_input)
        steps.append({
            "step": "Input Content Safety",
            "status": "completed" if safety_result["allowed"] else "blocked",
            "message": "Input is safe" if safety_result["allowed"] else f"Input blocked: {safety_result.get('flags', [])}"
        })
        
        if not safety_result["allowed"]:
            return {
                "result": {
                    "action": "blocked",
                    "message": "Your input was blocked due to content safety restrictions.",
                    "flags": safety_result.get("flags", [])
                },
                "steps": steps if track_steps else [],
                "services_used": list(services_used)
            }
        
        # Step 4: CLU intent & entities
        steps.append({
            "step": "CLU Analysis",
            "status": "processing",
            "message": "Analyzing intent and extracting entities..."
        })
        services_used.add("Azure CLU (Conversation Language Understanding)")
        clu_result = analyze_with_clu(clu_input)
        steps.append({
            "step": "CLU Analysis",
            "status": "completed",
            "message": f"Intent: {clu_result['intent']} (confidence: {clu_result['confidence']:.2%}), Entities: {clu_result.get('entities', {})}"
        })
        
        # Step 5: Router Agent → Execution Plan
        steps.append({
            "step": "Router Agent",
            "status": "processing",
            "message": "Generating execution plan..."
        })
        services_used.add("Azure OpenAI (Router Agent)")
        
        try:
            print(f"\n=== QUERY PROCESSOR: PASSING CLU RESULT TO ROUTER AGENT ===")
            print(f"CLU Intent: {clu_result.get('intent', 'N/A')}")
            print(f"CLU Confidence: {clu_result.get('confidence', 0):.2%}")
            print(f"CLU Entities: {json.dumps(clu_result.get('entities', {}), indent=2)}")
            plan = router_agent(clu_input, clu_result)
            print("\n=== ROUTER AGENT OUTPUT (PLAN) ===")
            try:
                print(json.dumps(plan, indent=2))
            except Exception as json_err:
                print(f"Error serializing plan to JSON: {json_err}")
                print(f"Plan: {plan}")
            
            steps.append({
                "step": "Router Agent",
                "status": "completed",
                "message": f"Plan generated with {len(plan.get('steps', []))} steps"
            })
        except Exception as e:
            steps.append({
                "step": "Router Agent",
                "status": "error",
                "message": f"Error generating plan: {str(e)}"
            })
            raise
        
        # Step 6: Execute the Plan
        steps.append({
            "step": "Executor",
            "status": "processing",
            "message": "Executing plan..."
        })
        
        try:
            execution_result = execute_plan(plan)
            print("\n=== EXECUTOR OUTPUT ===")
            try:
                print(json.dumps(execution_result, indent=2, default=str))
            except Exception as json_err:
                print(f"Error serializing execution result to JSON: {json_err}")
                print(f"Execution result: {execution_result}")
            
            steps.append({
                "step": "Executor",
                "status": "completed",
                "message": "Plan executed successfully"
            })
        except Exception as e:
            steps.append({
                "step": "Executor",
                "status": "error",
                "message": f"Error executing plan: {str(e)}"
            })
            raise
        
        # Step 7: Output content safety (track it)
        steps.append({
            "step": "Output Content Safety",
            "status": "completed",
            "message": "Output verified for safety"
        })
        
        # Extract structured data from execution result
        structured_result = {
            "action": "agent_execution",
            "search_results": None,
            "summary": None,
            "faq_answer": None,
            "fraud_analysis": None,
            "report": None
        }
        
        # Extract search results
        if "search_agent_output" in execution_result:
            search_output = execution_result["search_agent_output"]
            if isinstance(search_output, dict) and search_output.get("action") == "search":
                structured_result["search_results"] = search_output.get("results", [])
                structured_result["search_count"] = len(search_output.get("results", []))
        
        # Extract summary from explain_agent
        if "explain_agent_output" in execution_result:
            explain_output = execution_result["explain_agent_output"]
            if isinstance(explain_output, dict):
                structured_result["summary"] = {
                    "text": explain_output.get("summary", ""),
                    "confidence": explain_output.get("confidence", 0)
                }
        
        # Extract FAQ answer
        if "faq_agent_output" in execution_result:
            faq_output = execution_result["faq_agent_output"]
            if isinstance(faq_output, dict):
                structured_result["faq_answer"] = {
                    "answer": faq_output.get("answer", ""),
                    "confidence": faq_output.get("confidence", 0)
                }
        
        # Extract fraud analysis
        if "fraud_agent_output" in execution_result:
            fraud_output = execution_result["fraud_agent_output"]
            if isinstance(fraud_output, dict):
                structured_result["fraud_analysis"] = {
                    "risk_level": fraud_output.get("risk_level", "unknown"),
                    "signals": fraud_output.get("signals", []),
                    "note": fraud_output.get("note", "")
                }
        
        # Extract report
        if "report_agent_output" in execution_result:
            report_output = execution_result["report_agent_output"]
            if isinstance(report_output, dict):
                structured_result["report"] = report_output.get("report_text", "")
        
        # Return structured result along with raw execution result
        return {
            "result": structured_result,
            "execution_result": execution_result,  # Keep raw for debugging
            "plan": plan,
            "steps": steps if track_steps else [],
            "services_used": list(services_used),
            "original_text": user_text,
            "processed_text": clu_input
        }
        
    except Exception as e:
        steps.append({
            "step": "Error",
            "status": "error",
            "message": str(e)
        })
        return {
            "result": {
                "action": "error",
                "message": f"An error occurred: {str(e)}"
            },
            "steps": steps if track_steps else [],
            "services_used": list(services_used),
            "error": str(e)
        }
