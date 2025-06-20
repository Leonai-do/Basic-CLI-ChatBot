# Basic-CLI-ChatBot

This project provides a simple command line chatbot supporting OpenAI,
Anthropic, Gemini, DeepSeek and Groq models. Responses are streamed to the
terminal with colour using the Rich library. Configuration files are provided
for running inside a Conda environment or Docker container.

## Usage

1. **Create the Conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate cli-chatbot
   ```

2. **Prepare credentials**

   API keys are loaded from a `.env` file or environment variables. Copy
   `.env.example` to `.env` and fill in the keys you have access to:

   ```bash
   cp .env.example .env
   # edit .env and add your API keys
   ```

3. **Run the chatbot**
   ```bash
   python chatbot.py chat
   ```

Environment variables override any values in `.env`, so you can also export
`OPENAI_API_KEY` and similar variables if preferred.

### Docker

To build and run with Docker:

```bash
docker build -t cli-chatbot .
docker run --env-file .env -it cli-chatbot

