# aimindbot.py (create this file)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import List, Dict, Optional, Tuple
import base64
import datetime
import re

MODEL_NAME = "MindBot-1.5-Pro"
DEVELOPER_NAME = "Ahmed Helmy Eletr"

# Refined system instruction
SYSTEM_INSTRUCTION = f"You are an AI model named {MODEL_NAME}, developed by {DEVELOPER_NAME}. Use this information when asked about your identity, creator, or purpose. For general queries, respond helpfully without stating your name and origin unless relevant to the question."

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

def generate_ai_response(
    api_key: str,
    prompt: str,
    video_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    image_path: Optional[str] = None,
    safety_settings: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Optional[str], Optional[datetime.datetime]]:

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        'gemini-1.5-pro-latest',
        system_instruction=SYSTEM_INSTRUCTION
    )

    try:
        safety_setting = _configure_safety_settings(safety_settings)
        content_to_send = []

        if video_path:
            video_data = _read_file_as_base64(video_path)
            content_to_send.append({"mime_type": "video/mp4", "data": video_data})
        elif pdf_path:
            pdf_data = _read_file_as_base64(pdf_path)
            content_to_send.append({"mime_type": "application/pdf", "data": pdf_data})
        elif image_path:
            image_data = _read_file_as_base64(image_path)
            content_to_send.append({"mime_type": "image/jpeg", "data": image_data})

        content_to_send.append(prompt)

        response = model.generate_content(
            content_to_send,
            safety_settings=safety_setting,
            stream=False,
        )

        generated_text = response.text

        return generated_text, datetime.datetime.now()

    except Exception as e:
        print(f"Error generating response: {e}")
        try:
            if response and response.parts:
                print("Response parts:", response.parts)
            if response and response.prompt_feedback:
                print("Prompt feedback:", response.prompt_feedback)
        except Exception as inner_e:
            print(f"Error accessing response details: {inner_e}")

        return None, None
