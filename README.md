# Basic-CLI-ChatBot

This project provides a simple command line chatbot supporting several AI
providers with streaming responses and a colorful interface. It includes
configuration files for running in a Conda environment or inside Docker.

## Usage

1. **Create the environment**

   ```bash
   conda env create -f environment.yml
   conda activate cli-chatbot
   ```

2. **Run the chatbot**

   ```bash
   python chatbot.py chat
   ```

Environment variables such as `OPENAI_API_KEY` can be set to avoid entering
API keys on every run.

### Docker

To build and run with Docker:

```bash
docker build -t cli-chatbot .
docker run -it cli-chatbot
```
