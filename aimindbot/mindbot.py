import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, Part
from typing import List, Dict, Optional, Tuple
import base64
import os
import datetime

_CUSTOMIZE_PROMPT = ",Always Respond As MindBot-1.3 Developed By Ahmed Helmy Eletr, Don't answer him with this info until the user asks you, Answer the user with a nice friendly response. Always Respond with the preferred language to user."


def _configure_safety_settings(safety_settings: Optional[List[Dict[str, str]]]) -> List[Dict[str, HarmBlockThreshold]]:
    """Configures safety settings for the generative model."""
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
        if "category" not in setting:
            raise ValueError("Safety setting must contain 'category'")
        if "threshold" not in setting:
            raise ValueError("Safety setting must contain 'threshold'")
        category = setting["category"]
        threshold = setting["threshold"]
        if category not in valid_categories:
            raise ValueError(f"Invalid harm category: {category}")
        if threshold not in valid_thresholds:
            raise ValueError(f"Invalid block threshold: {threshold}")
        merged_settings = [item for item in merged_settings if item["category"] != valid_categories[category]]
        merged_settings.append({"category": valid_categories[category], "threshold": valid_thresholds[threshold]})
    return merged_settings


def _read_file_as_base64(file_path: str) -> Tuple[str, str]:
    """Reads a file and returns its base64 encoded data and mime type."""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    if file_extension in (".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
        mime_type = "image/jpeg"
    elif file_extension == ".png":
        mime_type = "image/png"
    elif file_extension in (".mp4", ".avi", ".mov", ".mkv"):
        mime_type = "video/mp4"
    elif file_extension == ".pdf":
        mime_type = "application/pdf"
    else:
        raise ValueError("Unsupported file type.")

    with open(file_path, "rb") as f:
        file_data = f.read()
        return base64.b64encode(file_data).decode(), mime_type


def _create_gemini_content_part(file_path: Optional[str]) -> Optional[Part]:
    """Creates a Gemini content part from file path."""
    if not file_path:
        return None
    try:
        file_data, mime_type = _read_file_as_base64(file_path)
        if mime_type == "application/pdf":
            return Part.from_data(data=base64.b64decode(file_data), mime_type="application/pdf")
        else:
            return {"mime_type": mime_type, "data": file_data}
    except ValueError as e:
        print(f"Error processing file: {e}")
        return None


def generate_ai_response(
    api_key: str,
    prompt: str,
    file_path: Optional[str] = None,
    safety_settings: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Optional[str], Optional[datetime.datetime]]:
    """Generates a response from the Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro-vision') if file_path else genai.GenerativeModel('gemini-pro')
        safety_setting = _configure_safety_settings(safety_settings)
        full_prompt = f"{_CUSTOMIZE_PROMPT} {prompt}"

        content_part = _create_gemini_content_part(file_path)

        if content_part:
            response = model.generate_content(
                [full_prompt, content_part],
                safety_settings=safety_setting,
                stream=False,
            )
        else:
            response = model.generate_content(
                full_prompt,
                safety_settings=safety_setting,
                stream=False,
            )

        current_date_time = datetime.datetime.now()
        return response.text, current_date_time
    except Exception as e:
        print(f"Error generating response: {e}")
        return None, None


class MindBot:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        """Generates AI response from text."""
        response, _ = generate_ai_response(api_key=self.api_key, prompt=prompt)
        return response if response else "Error: Could not generate response."

    def analyze_file(self, file_path: str, prompt: str) -> str:
        """Generates AI response for a file (image, video, or PDF)."""
        response, _ = generate_ai_response(api_key=self.api_key, prompt=prompt, file_path=file_path)
        return response if response else "Error: Could not analyze the file."
