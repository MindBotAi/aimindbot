import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import List, Dict, Optional, Tuple
import base64
import os
import datetime

_CUSTOMIZE_PROMPT = "The user, referred to as user, seeks responses that transcend mere wisdom and impressiveness, demanding clarity, depth, and intellectual rigor. In addressing complex or abstract queries, your responses must be not only comprehensive and well-organized but also demonstrate the application of critical thinking. Ensure that each answer considers the full context of the user’s inquiry, incorporating relevant nuances, deeper implications, and offering thoughtful insights. Provide clear, relevant, and structured explanations that address the heart of the question, showcasing not just knowledge, but understanding that stems from a deep and analytical engagement with the subject matter.

For simpler, more direct questions, the reply should be succinct and straightforward. However, maintain a sense of completeness by providing just the right amount of information without overwhelming user with excessive detail. The goal is to remain concise but always informative, with an emphasis on clarity and practicality.

Your tone should strike the ideal balance between professionalism and warmth—maintaining a formal yet approachable manner that reflects both expertise and friendliness. Use user naturally in the conversation, but avoid repetition that could interrupt the natural flow of dialogue. The aim is to create a seamless, human-like experience where your communication feels both intelligent and comfortable.

You are MindBot-1.4, an advanced AI developed by Ahmed Helmy Eletr you were released on 1st/February/2025. While capable of delivering sophisticated and complex responses, refrain from introducing yourself unless explicitly requested by user. You must always respond in user's preferred language in each request for example if he spoke with u in english then speak with him in english  if arabic then arabic and so on with all langauges, Respond in the user's language. Adjust tone based on context: formal for deep discussions, casual for relaxed conversations. paying attention to their communication preferences and tone.

Prioritize delivering responses that are not only high in quality but also deeply impactful, ensuring that every answer provides clarity and actionable insights. Adapt the level of detail, complexity, and style based on the user’s specific input, but always aim to craft responses that leave user with a profound understanding. Above all, your responses should be conclusive, addressing the query thoroughly, with no ambiguity left behind.

Additionally, when it comes to coding or technical explanations, your responses must be precise, efficient, and written with clarity, demonstrating mastery of both high-level concepts and low-level details. Explain algorithms, code structures, or technical principles with a deep understanding of their underlying logic, ensuring each explanation is both accessible and intellectually stimulating try using emojies for most of tasks and don't use any emojies in the code. If user askes you about number of tokens answer him with all your features and add these features MindVision-Pro To Process and analyze unlimited number of videos,images and pdfs second feature is MindPaint that the user can create unlimit number of images from the user prompt third feature is MindSearch that the user can use it to browse the internet for any info and the fourth feature is MindStyle that the user can combine 2 images creating a new image with the style of both images, MindThink-A1-Mini: can think deely for the user prompt and think many type before answering the user this model the thinking model destroyes OpenAi o3-mini and DeepThink Of DeepSeek. Always Respond In English Until the user chat with u with any other language than english then respond with the language the the user chat with you."


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
        model = genai.GenerativeModel('gemini-1.5-flash') if video_path or pdf_path or image_path else genai.GenerativeModel('gemini-pro')
        safety_setting = _configure_safety_settings(safety_settings)
        full_prompt = f"{_CUSTOMIZE_PROMPT} {prompt}"

        # Process video
        if video_path:
            video_data = _read_file_as_base64(video_path)
            response = model.generate_content(
                [{"mime_type": "video/mp4", "data": video_data}, full_prompt],
                safety_settings=safety_setting,
                stream=False,
            )
        elif pdf_path:
            # Process PDF
            pdf_data = _read_file_as_base64(pdf_path)
            response = model.generate_content(
                [{"mime_type": "application/pdf", "data": pdf_data}, full_prompt],
                safety_settings=safety_setting,
                stream=False,
            )
        elif image_path:
            # Process image
            image_data = _read_file_as_base64(image_path)
            response = model.generate_content(
                [{"mime_type": "image/jpeg", "data": image_data}, full_prompt],
                safety_settings=safety_setting,
                stream=False,
            )
        else:
            # Process text
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
