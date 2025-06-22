# 🤖 Basic CLI ChatBot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

A powerful, yet simple, command-line chatbot that supports multiple AI providers. Enjoy real-time, streaming responses beautifully rendered in your terminal with the Rich library. This project is designed for easy setup and use, with support for Conda, Pip, and Docker environments.

## ✨ Key Features

-   **🌐 Multi-Provider Support**: Seamlessly switch between major AI providers.
-   **⚡ Streaming Responses**: Get answers word-by-word in real-time, just like you're used to.
-   **🎨 Rich Terminal UI**: A beautiful and interactive command-line interface powered by `rich` and `typer`.
-   **🤔 "Thinking" Model Indicator**: Special UI indicators for models that have a longer "thought" process.
-   **📚 Conversation History**: Maintains session context for follow-up questions.
-   **🔑 Secure Credential Management**: Loads API keys safely from a `.env` file or environment variables.
-   **📦 Containerized**: Includes a `Dockerfile` for easy deployment and isolated execution.

## 📡 Supported Providers

The application allows you to choose from a variety of leading AI providers at runtime.

| Provider        | API Key Variable      | Special "Thinking" Models |
| :-------------- | :-------------------- | :------------------------ |
| 🤖 **OpenAI**   | `OPENAI_API_KEY`      | `o1`, `o1-mini`           |
| ✨ **Anthropic**| `ANTHROPIC_API_KEY`   | Claude 3.5 Sonnet & Haiku |
| 🚀 **Gemini**   | `GEMINI_API_KEY`      | Gemini 1.5 Pro & Flash    |
| 🧠 **DeepSeek** | `DEEPSEEK_API_KEY`    | `deepseek-reasoner`       |
| 🏎️ **Groq**      | `GROQ_API_KEY`        | Llama 3.1 70B & 8B        |

> **Note:** "Thinking" models perform more complex reasoning and may have a noticeable delay before responding, which is indicated in the UI.

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### ✅ Prerequisites

Ensure you have one of the following installed on your system:
-   Conda
-   Python 3.11+ and Pip
-   Docker

### ⚙️ Installation

First, clone the repository to your local machine using the command `git clone https://github.com/your-username/Basic-CLI-ChatBot.git`, then navigate into the new directory with `cd Basic-CLI-ChatBot`.

From there, choose your preferred installation method below.

<details>
<summary><strong>🔵 Option 1: Using Conda (Recommended)</strong></summary>

1.  **Create the Environment**: Execute the command `conda env create -f environment.yml` to build the environment from the specification file.
2.  **Activate the Environment**: Once created, activate it with `conda activate cli-chatbot`.

</details>

<details>
<summary><strong>🟡 Option 2: Using Pip and a Virtual Environment</strong></summary>

1.  **Create a Virtual Environment**: It is best practice to create a virtual environment. You can do this with `python3 -m venv venv`.
2.  **Activate the Environment**:
    -   On macOS/Linux, run `source venv/bin/activate`.
    -   On Windows, run `.\venv\Scripts\activate`.
3.  **Install Packages**: Install all required dependencies by running `pip install "openai" "anthropic" "google-generativeai" "groq" "rich" "typer[all]" "prompt-toolkit" "python-dotenv" "nest-asyncio"`.

</details>

<details>
<summary><strong>⚪ Option 3: Using Docker</strong></summary>

1.  **Build the Image**: From the project's root directory, build the Docker image by running the command `docker build -t cli-chatbot .`.
2.  You can run the container after completing the configuration step below.

</details>

### 🔑 Configuration

The chatbot needs API keys to communicate with the AI providers.

1.  Create your personal configuration file by making a copy of the example: `cp .env.example .env`.
2.  Open the newly created `.env` file in a text editor and add your API keys for the services you wish to use. You only need to fill in the ones you have access to.

    ```
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GEMINI_API_KEY=AIza...
    DEEPSEEK_API_KEY=sk-...
    GROQ_API_KEY=gsk_...
    ```

> The application will also automatically detect keys set as environment variables in your system, which will take precedence over the `.env` file.

---

## 💬 Usage

Once you have installed the dependencies and configured your API keys, you can start the chatbot.

-   If you used **Conda or Pip**, run the application with `python chatbot.py chat`.
-   If you are using **Docker**, launch the interactive container with `docker run --env-file .env -it cli-chatbot`.

Upon starting, you will be prompted to select a provider and a model. After that, you're ready to chat!

### In-Chat Commands

| Command                 | Action                                       |
| :---------------------- | :------------------------------------------- |
| `clear`                 | Clears the current conversation history.     |
| `quit`, `exit`, or `q`  | Exits the chatbot application.               |
| `Ctrl+C`                | Interrupts a streaming response or exits.    |

---

## 🧪 Development & Testing

This project uses `pytest` for testing. To validate the functionality, ensure you have the development dependencies installed and simply run `pytest` in your terminal from the project's root directory.

## 📄 License

Distributed under the MIT License.