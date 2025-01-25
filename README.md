# MindBot-1.3: An AI-Powered Ai Model

![MindBot Logo](https://i.ibb.co/MR7wKr4/mindboewebsite-logo.png)  <!-- Replace with your actual logo image -->

This project is a simple interactive chatbot that allows users to input prompts and receive responses in real time.

## Overview

`MindBot-1.3` is built using:

-   **Torch** For natural language processing and AI response generation.
-   **Pillow** For Langchain as pdf, images and video processing.
-   **Python:** As the programming language.
-   **Colorama:** For colored output in the terminal.

Here's a high-level overview of how the code works:

![MindBot Architecture](https://scontent-hbe1-1.xx.fbcdn.net/v/t39.30808-6/471411222_122127522944572546_6475440340774723854_n.jpg?_nc_cat=106&ccb=1-7&_nc_sid=127cfc&_nc_ohc=-BP2Qwfp2yUQ7kNvgExdWMe&_nc_oc=AdhWXXBaLMj_0P-_Hmcm2hQXE-Tj72w5oj4GPuUovZqBLirVxzevDsMxiFq_VCkXdLU&_nc_zt=23&_nc_ht=scontent-hbe1-1.xx&_nc_gid=Aavg5ux2Wg4otQ7H9AHO-kR&oh=00_AYCcP-5cZlU2WniVt8bfQAGMD6oqJrMpn--Xqe2IYNJQTQ&oe=6776E5CF)

This diagram provides a simplified view of the interactions between the user, the MindBot application

## Installation

1.  **Clone the Repository (Optional):** If you have this in a Git repository, clone it using:
    ```bash
    git clone [your_repository_url]
    cd mindbot-project
    ```
2.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    This command installs all necessary libraries listed in `requirements.txt`.

## Code Explanation (`main.py`)

```python
import mindbotai
import datetime
import time
from colorama import init, Fore

# Initialize colorama for colored terminal output
init(autoreset=True)

if __name__ == '__main__':
    api_key = "YOUR_API_KEY"  # Replace with your actual API key
    while True:
        user_prompt = input(str(f"{Fore.GREEN}User:> {Fore.RESET}"))
        if user_prompt.lower() == 'exit':
            break

        start_time = time.time()
        submit_time = datetime.datetime.now()
        response = mindbotai.generate_ai_response(api_key, user_prompt)
        end_time = time.time()
        response_time = end_time - start_time

        if response:
           print(f"{Fore.BLUE}MindBot-1.3:> {Fore.RESET}{response}")
           print(f"Submitted at: {submit_time.strftime('%Y-%m-%d %H:%M:%S')}")
           print(f"Response Time: {response_time:.2f} seconds\n")
        else:
             print("Failed to get the MindBot-1.3 response.")
```
# Key Features

API Key Setup: Sets the api_key. Remember to replace "YOUR_API_KEY" with your actual API key.

Import mindbotai Module: Imports the logic from mindbotai.py.

User Prompt: Prompts user for input with green color.

Response Generation: Calls the generate_ai_response function to get a response from the AI model.

Timestamp and Response Time: Records the submission time and calculates response time.

Colored Output: Prints the AI's responses in blue and displays time information.

Exit Condition: The user can exit using the 'exit' keyword.

# How To Run
```
python main.py
```

# Usage Example

```commandline
User:> What is the largest planet in our solar system?
MindBot-1.3:> The largest planet in our solar system is Jupiter.
Submitted at: 2024-05-02 18:27:45
Response Time: 2.05 seconds

User:> Tell me a joke
MindBot-1.3:> Why don't scientists trust atoms? Because they make up everything!
Submitted at: 2024-05-02 18:27:52
Response Time: 1.82 seconds

User:> exit
```

# Understanding the MindBot API

![alt text](https://scontent-hbe1-1.xx.fbcdn.net/v/t39.30808-6/469557831_122123396750572546_3266587420609878967_n.jpg?_nc_cat=109&ccb=1-7&_nc_sid=127cfc&_nc_ohc=AnVSQ58UAIUQ7kNvgHKVL61&_nc_oc=Adjo1-LPZ125pHkCqQ4DxWEftY5bvV0xUJtwPvQ8i7dybHM2gMIlADmv6emMw3WMRMo&_nc_zt=23&_nc_ht=scontent-hbe1-1.xx&_nc_gid=AFSLOrCZafhnsd8tmwkbS_y&oh=00_AYBfSKJEABMURjsAV3vPU4V284sC5NDmM1oKv_tYXc7EeQ&oe=6776D1B3)

The mindbotai.generate_ai_response function takes your API key and prompt and returns a response from the MindBot API. This provides a convenient way to interact with an AI model.

# Requirements
Ensure you have all the required libraries installed in your requirements.txt file by running.

```commandline
 pip install -r requirements.txt
```
