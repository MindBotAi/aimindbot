import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

customize = ",Always Respond As MindBot-1.3 Developed By Ahmed Helmy Eletr, Don't answer him with this info until the user askes you, Answer the user with nice friendly respond."


def configure_safety_settings(safety_settings=None):
    """Configures safety settings for the generative model.

    Args:
        safety_settings (dict, optional): A dictionary containing safety settings
          for the model. Defaults to None.

    Returns:
        list: A list of safety settings dictionaries.

    Raises:
        ValueError: If a category provided in safety settings is invalid.
    """
    default_settings = [
        {
            "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        {
            "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        {
            "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        {
            "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
    ]

    if not safety_settings:
        return default_settings

    # check if the category is valid and convert the keys to harmCategory
    valid_categories = {
        "HARM_CATEGORY_HATE_SPEECH": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "HARM_CATEGORY_HARASSMENT": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "HARM_CATEGORY_DANGEROUS_CONTENT": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    }

    # check if the threshold is valid and convert the keys to harmBlockThreshold
    valid_thresholds = {
        "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
        "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    merged_settings = []
    for settings in default_settings:
        merged_settings.append(settings)

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
            raise ValueError(f"Invalid block threshold {threshold}")

        merged_settings = [item for item in merged_settings if item["category"] != valid_categories[category]]

        merged_settings.append({
            "category": valid_categories[category],
            "threshold": valid_thresholds[threshold],
        })

    return merged_settings


def generate_ai_response(api_key, prompt, safety_settings=None):
    """Generates a response from the Gemini API.

    Args:
        api_key (str): Your Google Gemini API key.
        prompt (str): The user's prompt.
        safety_settings (dict, optional): A dictionary containing safety settings
            for the model. Defaults to None.


    Returns:
        str: The generated response, or None if an error occurs.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        safety_setting = configure_safety_settings(safety_settings)

        # Add customize to every prompt
        full_prompt = f"{customize} {prompt}"

        response = model.generate_content(
            full_prompt,
            safety_settings=safety_setting,
            stream=False,
        )
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return None