import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import List, Dict, Optional, Tuple
import base64
import os
import datetime

def _configure_safety_settings(safety_settings: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
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
        if "category" not in setting or "threshold" not in setting:
            raise ValueError("Safety setting must contain 'category' and 'threshold'")
        category = setting["category"]
        threshold = setting["threshold"]
        if category not in valid_categories:
            raise ValueError(f"Invalid harm category: {category}")
        if threshold not in valid_thresholds:
            raise ValueError(f"Invalid block threshold: {threshold}")
        
        merged_settings = [item for item in merged_settings if item["category"] != valid_categories[category]]
        merged_settings.append({"category": valid_categories[category], "threshold": valid_thresholds[threshold]})
    
    return merged_settings


def _read_file_as_base64(file_path: str) -> str:
    """Reads a file (e.g., image, PDF, or video) and returns its base64 encoded data."""
    with open(file_path, "rb") as file:
        return base64.standard_b64encode(file.read()).decode("utf-8")


def generate_ai_response(
    api_key: str,
    prompt: str,
    video_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    image_path: Optional[str] = None,
    safety_settings: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Optional[str], Optional[datetime.datetime]]:
    """Generates a response from the Gemini API for text, image, PDF, or video-based prompts.

    Args:
        api_key (str): Your Google Gemini API key.
        prompt (str): The user's prompt.
        video_path (str, optional): Path to the video file for analysis. Defaults to None.
        pdf_path (str, optional): Path to the PDF file for analysis. Defaults to None.
        image_path (str, optional): Path to the image file for analysis. Defaults to None.
        safety_settings (list, optional): Safety settings for the model. Defaults to None.

    Returns:
        tuple: A tuple containing the generated response and current date & time, or (None, None) if an error occurs.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        safety_setting = _configure_safety_settings(safety_settings)

        # Check if the user is asking about the bot's identity
        bot_identity_trigger = ["who are you", "what is your name", "identify yourself"]
        if any(trigger in prompt.lower() for trigger in bot_identity_trigger):
            response_text = "I am MindBot-1.4, an advanced AI developed by Ahmed Helmy Eletr."
            return response_text, datetime.datetime.now()

        # Process input files (video, PDF, image)
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

        return response.text, datetime.datetime.now()
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return None, None
