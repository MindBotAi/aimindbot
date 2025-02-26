import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import List, Dict, Optional, Tuple
import base64
import datetime
import re

custom_identity_response = "I am an advanced AI model Called MindBot-1.3 Developed By Ahmed Helmy Eletr and designed for intelligent and insightful responses."

def _configure_safety_settings(safety_settings: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    default_settings = [
        {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
    ]
    
    if not safety_settings:
        return default_settings
    
    valid_categories = {
        "HARM_CATEGORY_HATE_SPEECH": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "HARM_CATEGORY_HARASSMENT": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "HARM_CATEGORY_DANGEROUS_CONTENT": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    }
    valid_thresholds = {
        "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
        "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    
    merged_settings = default_settings[:]
    for setting in safety_settings:
        if "category" not in setting or "threshold" not in setting:
            raise ValueError("Safety setting must contain 'category' and 'threshold'")
        category = setting["category"]
        threshold = setting["threshold"]
        if category not in valid_categories or threshold not in valid_thresholds:
            raise ValueError("Invalid category or threshold")
        
        merged_settings = [item for item in merged_settings if item["category"] != valid_categories[category]]
        merged_settings.append({"category": valid_categories[category], "threshold": valid_thresholds[threshold]})
    
    return merged_settings

def _read_file_as_base64(file_path: str) -> str:
    with open(file_path, "rb") as file:
        return base64.standard_b64encode(file.read()).decode("utf-8")

def is_identity_query(prompt: str) -> bool:
    identity_patterns = [
        r"\bwho\s+are\s+you\b",
        r"\bwhat\s+is\s+your\s+name\b",
        r"\bidentify\s+yourself\b",
        r"\btell\s+me\s+about\s+yourself\b",
        r"\bare\s+you\s+an\s+ai\b",
        r"\bwhat\s+do\s+you\s+do\b",
        r"\bdescribe\s+yourself\b"
    ]
    
    return any(re.search(pattern, prompt.lower()) for pattern in identity_patterns)

def generate_ai_response(
    api_key: str,
    prompt: str,
    video_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    image_path: Optional[str] = None,
    safety_settings: Optional[List[Dict[str, str]]] = None,
    custom_response_prefix: Optional[str] = None, # Adding a custom prefix
) -> Tuple[Optional[str], Optional[datetime.datetime]]:
    if is_identity_query(prompt):
        return custom_identity_response, datetime.datetime.now()

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        safety_setting = _configure_safety_settings(safety_settings)

        if video_path:
            video_data = _read_file_as_base64(video_path)
            response = model.generate_content(
                [{"mime_type": "video/mp4", "data": video_data}, prompt],
                safety_settings=safety_setting,
                stream=False,
            )
        elif pdf_path:
            pdf_data = _read_file_as_base64(pdf_path)
            response = model.generate_content(
                [{"mime_type": "application/pdf", "data": pdf_data}, prompt],
                safety_settings=safety_setting,
                stream=False,
            )
        elif image_path:
            image_data = _read_file_as_base64(image_path)
            response = model.generate_content(
                [{"mime_type": "image/jpeg", "data": image_data}, prompt],
                safety_settings=safety_setting,
                stream=False,
            )
        else:
            response = model.generate_content(
                prompt,
                safety_settings=safety_setting,
                stream=False,
            )
        
        generated_text = response.text

        # Add custom prefix if it is available
        if custom_response_prefix:
            generated_text = f"{custom_response_prefix} {generated_text}"

        return generated_text, datetime.datetime.now()
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return None, None
