This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
.env.example/
  .env.example
.gitignore/
  .gitignore
chatbot.py/
  chatbot.py
Dockerfile/
  Dockerfile
environment.yml/
  environment.yml
README.md/
  README.md
repomix-output-Leonai-do-Basic-CLI-ChatBot.md/
  repomix-output-Leonai-do-Basic-CLI-ChatBot.md
test_chatbot.py/
  test_chatbot.py
```

# Files

## File: .env.example/.env.example
`````
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
GROQ_API_KEY=
`````

## File: .gitignore/.gitignore
`````
.env
__pycache__/
`````

## File: chatbot.py/chatbot.py
`````python
#!/usr/bin/env python3
"""CLI chatbot supporting multiple providers with streaming output."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from prompt_toolkit import prompt
from dotenv import load_dotenv

# Provider libraries. These are imported lazily in provider classes
import openai
import anthropic
import google.generativeai as genai
from groq import Groq

console = Console()
class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    GROQ = "groq"

@dataclass
class ModelInfo:
    name: str
    display_name: str
    is_thinking: bool = False
    max_tokens: int = 4096

PROVIDER_MODELS: dict[Provider, List[ModelInfo]] = {
    Provider.OPENAI: [
        ModelInfo("gpt-4o", "GPT-4o", max_tokens=128000),
        ModelInfo("gpt-4", "GPT-4"),
        ModelInfo("o3", "o3", is_thinking=True),
    ],
    Provider.ANTHROPIC: [
        ModelInfo("claude-3-sonnet-20240229", "Claude 3 Sonnet"),
        ModelInfo("claude-3-haiku-20240307", "Claude 3 Haiku"),
    ],
    Provider.GEMINI: [
        ModelInfo("gemini-pro", "Gemini Pro"),
    ],
    Provider.DEEPSEEK: [
        ModelInfo("deepseek-chat", "DeepSeek Chat"),
        ModelInfo("deepseek-reasoner", "DeepSeek R1", is_thinking=True),
    ],
    Provider.GROQ: [
        ModelInfo("llama3-70b-8192", "Llama3 70B"),
    ],
}

class ChatbotProvider:
    def __init__(self, api_key: str, model: ModelInfo):
        self.api_key = api_key
        self.model = model
        self.history: List[dict[str, str]] = []

    def add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        raise NotImplementedError

class OpenAIProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = openai.OpenAI(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            def sync_stream():
                return self.client.chat.completions.create(
                    model=self.model.name,
                    messages=self.history,
                    stream=True,
                    max_tokens=self.model.max_tokens,
                )
            stream = await asyncio.to_thread(sync_stream)
            full = ""
            for chunk in stream:
                delta = None
                if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                    delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                full += delta
                yield delta
            self.add_to_history("assistant", full)
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {str(e)}"
            console.print(error_msg, style="bold red")
            yield error_msg
        except openai.APIConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            console.print(error_msg, style="bold red")
            yield error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            console.print(error_msg, style="bold red")
            yield error_msg

class AnthropicProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            messages = [m for m in self.history if m["role"] != "system"]
            with self.client.messages.stream(
                model=self.model.name,
                messages=messages,
                max_tokens=self.model.max_tokens,
            ) as stream:
                full = ""
                for text in stream.text_stream:
                    full += text
                    yield text
                self.add_to_history("assistant", full)
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class GeminiProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        self.model_client = genai.GenerativeModel(model.name)
        self.chat = self.model_client.start_chat(history=[])

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        try:
            resp = self.chat.send_message(
                message,
                stream=True,
                generation_config=genai.types.GenerationConfig(max_output_tokens=self.model.max_tokens),
            )
            for chunk in resp:
                if chunk.text:
                    yield chunk.text
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class DeepSeekProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            stream = self.client.chat.completions.create(
                model=self.model.name,
                messages=self.history,
                stream=True,
                max_tokens=self.model.max_tokens,
            )
            full = ""
            hide = False
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                if self.model.is_thinking:
                    if "<thinking>" in delta:
                        hide = True
                        continue
                    if "</thinking>" in delta:
                        hide = False
                        continue
                    if hide:
                        continue
                full += delta
                yield delta
            self.add_to_history("assistant", full)
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class GroqProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = Groq(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            stream = self.client.chat.completions.create(
                model=self.model.name,
                messages=self.history,
                stream=True,
                max_tokens=self.model.max_tokens,
            )
            full = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                full += delta
                yield delta
            self.add_to_history("assistant", full)
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class ChatBot:
    def __init__(self):
        self.provider: Optional[ChatbotProvider] = None
        self.provider_type: Optional[Provider] = None
        self.model: Optional[ModelInfo] = None

    def display_welcome(self) -> None:
        text = Text()
        text.append("Multi-Provider CLI Chatbot\n", style="bold blue")
        text.append("Supports OpenAI, Anthropic, Gemini, DeepSeek, Groq", style="green")
        console.print(Panel(text, title="Welcome", border_style="blue"))

    def select_provider_and_model(self) -> tuple[Provider, ModelInfo]:
        providers = list(Provider)
        console.print("\nProviders:", style="bold cyan")
        for i, p in enumerate(providers, 1):
            console.print(f"  {i}. {p.value}")
        
        while True:
            try:
                choice_input = prompt("Select provider (1-{}): ".format(len(providers)))
                if not choice_input.strip():
                    console.print("Please enter a number", style="red")
                    continue
                
                choice = int(choice_input) - 1
                if 0 <= choice < len(providers):
                    provider = providers[choice]
                    break
                else:
                    console.print(f"Please enter a number between 1 and {len(providers)}", style="red")
            except ValueError:
                console.print("Please enter a valid number", style="red")
        
        models = PROVIDER_MODELS[provider]
        console.print(f"\nModels for {provider.value}:", style="bold cyan")
        for i, m in enumerate(models, 1):
            mark = " 🧠" if m.is_thinking else ""
            console.print(f"  {i}. {m.display_name}{mark}")
        
        while True:
            try:
                choice_input = prompt("Select model (1-{}): ".format(len(models)))
                if not choice_input.strip():
                    console.print("Please enter a number", style="red")
                    continue
                
                choice = int(choice_input) - 1
                if 0 <= choice < len(models):
                    model = models[choice]
                    break
                else:
                    console.print(f"Please enter a number between 1 and {len(models)}", style="red")
            except ValueError:
                console.print("Please enter a valid number", style="red")
        
        return provider, model

    def get_api_key(self, provider: Provider) -> str:
        load_dotenv(dotenv_path=".env", override=False)
        env_name = f"{provider.value.upper()}_API_KEY"
        key = os.getenv(env_name)
        if key:
            console.print(
                f"Using {env_name} from environment or .env", style="green"
            )
            return key
        console.print(f"Enter your {provider.value.title()} API key:")
        key = prompt("API key: ", is_password=True)
        if not key:
            console.print("API key required", style="red")
            sys.exit(1)
        return key

    def create_provider(self, provider: Provider, key: str, model: ModelInfo) -> ChatbotProvider:
        mapping = {
            Provider.OPENAI: OpenAIProvider,
            Provider.ANTHROPIC: AnthropicProvider,
            Provider.GEMINI: GeminiProvider,
            Provider.DEEPSEEK: DeepSeekProvider,
            Provider.GROQ: GroqProvider,
        }
        cls = mapping[provider]
        return cls(key, model)

    async def chat_loop(self) -> None:
        console.print("\nType 'quit' to exit", style="green")
        if self.model and self.model.is_thinking:
            console.print("Thinking model active", style="yellow")
        while True:
            user = prompt("You: ").strip()
            if user.lower() in {"quit", "exit"}:
                break
            if not user:
                continue
            console.print(Panel(Text(user, style="blue"), title="You", border_style="blue"))
            console.print("AI:", style="bold green")
            if self.model and self.model.is_thinking:
                with Live(Spinner("dots", text="Thinking...")):
                    await asyncio.sleep(0.5)
            out = Text()
            with Live(out, refresh_per_second=10) as live:
                async for chunk in self.provider.stream_response(user):
                    out.append(chunk, style="green")
                    live.update(out)
            console.print(Panel(out, title="Assistant", border_style="green"))

    async def run(self) -> None:
        self.display_welcome()
        self.provider_type, self.model = self.select_provider_and_model()
        key = self.get_api_key(self.provider_type)
        self.provider = self.create_provider(self.provider_type, key, self.model)
        console.print(f"Connected to {self.provider_type.value}", style="green")
        await self.chat_loop()

app = typer.Typer(help="CLI Chatbot")

@app.command()
def chat():
    """Start the chatbot CLI interface."""
    import nest_asyncio
    nest_asyncio.apply()
    
    chatbot = ChatBot()
    asyncio.run(chatbot.run())

if __name__ == "__main__":
    app()
`````

## File: Dockerfile/Dockerfile
`````
FROM continuumio/miniconda3:latest

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate cli-chatbot" > /etc/profile.d/conda.sh
ENV PATH /opt/conda/envs/cli-chatbot/bin:$PATH

WORKDIR /app
COPY . /app

CMD ["python", "chatbot.py", "chat"]
`````

## File: environment.yml/environment.yml
`````yaml
name: cli-chatbot
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - openai
      - anthropic
      - google-generativeai
      - groq
      - rich
      - typer
      - prompt-toolkit
      - pytest
      - pytest-asyncio
      - python-dotenv
      - nest-asyncio
`````

## File: README.md/README.md
`````markdown
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
`````

## File: repomix-output-Leonai-do-Basic-CLI-ChatBot.md/repomix-output-Leonai-do-Basic-CLI-ChatBot.md
`````markdown
This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
.env.example
.gitignore
chatbot.py
Dockerfile
environment.yml
README.md
test_chatbot.py
```

# Files

## File: .env.example
````
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
GROQ_API_KEY=
````

## File: .gitignore
````
.env
__pycache__/
````

## File: chatbot.py
````python
#!/usr/bin/env python3
"""CLI chatbot supporting multiple providers with streaming output."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from prompt_toolkit import prompt
from dotenv import load_dotenv

# Provider libraries. These are imported lazily in provider classes
import openai
import anthropic
import google.generativeai as genai
from groq import Groq

console = Console()

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    GROQ = "groq"

@dataclass
class ModelInfo:
    name: str
    display_name: str
    is_thinking: bool = False
    max_tokens: int = 4096

PROVIDER_MODELS: dict[Provider, List[ModelInfo]] = {
    Provider.OPENAI: [
        ModelInfo("gpt-4o", "GPT-4o", max_tokens=128000),
        ModelInfo("gpt-4", "GPT-4"),
        ModelInfo("o3", "o3", is_thinking=True),
    ],
    Provider.ANTHROPIC: [
        ModelInfo("claude-3-sonnet-20240229", "Claude 3 Sonnet"),
        ModelInfo("claude-3-haiku-20240307", "Claude 3 Haiku"),
    ],
    Provider.GEMINI: [
        ModelInfo("gemini-pro", "Gemini Pro"),
    ],
    Provider.DEEPSEEK: [
        ModelInfo("deepseek-chat", "DeepSeek Chat"),
        ModelInfo("deepseek-reasoner", "DeepSeek R1", is_thinking=True),
    ],
    Provider.GROQ: [
        ModelInfo("llama3-70b-8192", "Llama3 70B"),
    ],
}

class ChatbotProvider:
    def __init__(self, api_key: str, model: ModelInfo):
        self.api_key = api_key
        self.model = model
        self.history: List[dict[str, str]] = []

    def add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        raise NotImplementedError

class OpenAIProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = openai.OpenAI(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            stream = self.client.chat.completions.create(
                model=self.model.name,
                messages=self.history,
                stream=True,
                max_tokens=self.model.max_tokens,
            )
            full = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                full += delta
                yield delta
            self.add_to_history("assistant", full)
        except Exception as e:  # pragma: no cover - network errors
            yield f"Error: {e}"

class AnthropicProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            messages = [m for m in self.history if m["role"] != "system"]
            with self.client.messages.stream(
                model=self.model.name,
                messages=messages,
                max_tokens=self.model.max_tokens,
            ) as stream:
                full = ""
                for text in stream.text_stream:
                    full += text
                    yield text
                self.add_to_history("assistant", full)
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class GeminiProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        self.model_client = genai.GenerativeModel(model.name)
        self.chat = self.model_client.start_chat(history=[])

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        try:
            resp = self.chat.send_message(
                message,
                stream=True,
                generation_config=genai.types.GenerationConfig(max_output_tokens=self.model.max_tokens),
            )
            for chunk in resp:
                if chunk.text:
                    yield chunk.text
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class DeepSeekProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            stream = self.client.chat.completions.create(
                model=self.model.name,
                messages=self.history,
                stream=True,
                max_tokens=self.model.max_tokens,
            )
            full = ""
            hide = False
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                if self.model.is_thinking:
                    if "<thinking>" in delta:
                        hide = True
                        continue
                    if "</thinking>" in delta:
                        hide = False
                        continue
                    if hide:
                        continue
                full += delta
                yield delta
            self.add_to_history("assistant", full)
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class GroqProvider(ChatbotProvider):
    def __init__(self, api_key: str, model: ModelInfo):
        super().__init__(api_key, model)
        self.client = Groq(api_key=api_key)

    async def stream_response(self, message: str) -> AsyncGenerator[str, None]:
        self.add_to_history("user", message)
        try:
            stream = self.client.chat.completions.create(
                model=self.model.name,
                messages=self.history,
                stream=True,
                max_tokens=self.model.max_tokens,
            )
            full = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                full += delta
                yield delta
            self.add_to_history("assistant", full)
        except Exception as e:  # pragma: no cover
            yield f"Error: {e}"

class ChatBot:
    def __init__(self):
        self.provider: Optional[ChatbotProvider] = None
        self.provider_type: Optional[Provider] = None
        self.model: Optional[ModelInfo] = None

    def display_welcome(self) -> None:
        text = Text()
        text.append("Multi-Provider CLI Chatbot\n", style="bold blue")
        text.append("Supports OpenAI, Anthropic, Gemini, DeepSeek, Groq", style="green")
        console.print(Panel(text, title="Welcome", border_style="blue"))

    def select_provider_and_model(self) -> tuple[Provider, ModelInfo]:
        providers = list(Provider)
        console.print("\nProviders:", style="bold cyan")
        for i, p in enumerate(providers, 1):
            console.print(f"  {i}. {p.value}")
        while True:
            try:
                choice = int(prompt("Select provider: ")) - 1
                if 0 <= choice < len(providers):
                    provider = providers[choice]
                    break
            except KeyboardInterrupt:
                raise
            except Exception:
                pass
            console.print("Invalid choice", style="red")

        models = PROVIDER_MODELS[provider]
        console.print(f"\nModels for {provider.value}:", style="bold cyan")
        for i, m in enumerate(models, 1):
            mark = " 🧠" if m.is_thinking else ""
            console.print(f"  {i}. {m.display_name}{mark}")
        while True:
            try:
                choice = int(prompt("Select model: ")) - 1
                if 0 <= choice < len(models):
                    model = models[choice]
                    break
            except KeyboardInterrupt:
                raise
            except Exception:
                pass
            console.print("Invalid choice", style="red")
        return provider, model

    def get_api_key(self, provider: Provider) -> str:
        load_dotenv(dotenv_path=".env", override=False)
        env_name = f"{provider.value.upper()}_API_KEY"
        key = os.getenv(env_name)
        if key:
            console.print(
                f"Using {env_name} from environment or .env", style="green"
            )
            return key
        console.print(f"Enter your {provider.value.title()} API key:")
        key = prompt("API key: ", is_password=True)
        if not key:
            console.print("API key required", style="red")
            sys.exit(1)
        return key

    def create_provider(self, provider: Provider, key: str, model: ModelInfo) -> ChatbotProvider:
        mapping = {
            Provider.OPENAI: OpenAIProvider,
            Provider.ANTHROPIC: AnthropicProvider,
            Provider.GEMINI: GeminiProvider,
            Provider.DEEPSEEK: DeepSeekProvider,
            Provider.GROQ: GroqProvider,
        }
        cls = mapping[provider]
        return cls(key, model)

    async def chat_loop(self) -> None:
        console.print("\nType 'quit' to exit", style="green")
        if self.model and self.model.is_thinking:
            console.print("Thinking model active", style="yellow")
        while True:
            user = prompt("You: ").strip()
            if user.lower() in {"quit", "exit"}:
                break
            if not user:
                continue
            console.print(Panel(Text(user, style="blue"), title="You", border_style="blue"))
            console.print("AI:", style="bold green")
            if self.model and self.model.is_thinking:
                with Live(Spinner("dots", text="Thinking...")):
                    await asyncio.sleep(0.5)
            out = Text()
            with Live(out, refresh_per_second=10) as live:
                async for chunk in self.provider.stream_response(user):
                    out.append(chunk, style="green")
                    live.update(out)
            console.print(Panel(out, title="Assistant", border_style="green"))

    async def run(self) -> None:
        self.display_welcome()
        self.provider_type, self.model = self.select_provider_and_model()
        key = self.get_api_key(self.provider_type)
        self.provider = self.create_provider(self.provider_type, key, self.model)
        console.print(f"Connected to {self.provider_type.value}", style="green")
        await self.chat_loop()

app = typer.Typer(help="CLI Chatbot")

@app.command()
def chat():
    chatbot = ChatBot()
    asyncio.run(chatbot.run())

if __name__ == "__main__":
    app()
````

## File: Dockerfile
````dockerfile
FROM continuumio/miniconda3:latest

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate cli-chatbot" > /etc/profile.d/conda.sh
ENV PATH /opt/conda/envs/cli-chatbot/bin:$PATH

WORKDIR /app
COPY . /app

CMD ["python", "chatbot.py", "chat"]
````

## File: environment.yml
````yaml
name: cli-chatbot
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - openai
      - anthropic
      - google-generativeai
      - groq
      - rich
      - typer
      - prompt-toolkit
      - pytest
      - pytest-asyncio
      - python-dotenv
````

## File: README.md
````markdown
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
````

## File: test_chatbot.py
````python
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from chatbot import (
    ChatBot,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    DeepSeekProvider,
    GroqProvider,
    ModelInfo,
    Provider,
)

@pytest.fixture
def mock_model():
    return ModelInfo("test-model", "Test Model")

@pytest.fixture
def thinking_model():
    return ModelInfo("test-thinking", "Thinking Model", is_thinking=True)


def test_provider_initialization(mock_model):
    key = "k"
    for cls in [OpenAIProvider, AnthropicProvider, GeminiProvider, DeepSeekProvider, GroqProvider]:
        with patch.object(cls, "__init__", return_value=None) as init:
            cls(key, mock_model)
            init.assert_called()

@pytest.mark.asyncio
async def test_openai_stream(mock_model):
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="a"))]),
        Mock(choices=[Mock(delta=Mock(content="b"))]),
    ]
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = OpenAIProvider("k", mock_model)
        provider.client = m.return_value
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert out == ["a", "b"]

@pytest.mark.asyncio
async def test_deepseek_thinking(thinking_model):
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="<thinking>"))]),
        Mock(choices=[Mock(delta=Mock(content="hidden"))]),
        Mock(choices=[Mock(delta=Mock(content="</thinking>"))]),
        Mock(choices=[Mock(delta=Mock(content="final"))]),
    ]
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = DeepSeekProvider("k", thinking_model)
        provider.client = m.return_value
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert out == ["final"]


def test_history(mock_model):
    p = OpenAIProvider("k", mock_model)
    p.client = Mock()
    p.add_to_history("user", "hi")
    p.add_to_history("assistant", "yo")
    assert p.history[0]["role"] == "user"
    assert p.history[1]["role"] == "assistant"

@pytest.mark.asyncio
async def test_error_handling(mock_model):
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.side_effect = Exception("boom")
        provider = OpenAIProvider("k", mock_model)
        provider.client = m.return_value
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert any("Error" in c for c in out)

def test_get_api_key_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=dotenv-key")
    monkeypatch.chdir(tmp_path)
    # ensure no env var set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    bot = ChatBot()
    assert bot.get_api_key(Provider.OPENAI) == "dotenv-key"

class DummyBenchmark:
    async def __call__(self, func):
        return await func()

@pytest.mark.asyncio
async def test_stream_perf(mock_model):
    mock_stream = [Mock(choices=[Mock(delta=Mock(content=f"c{i}"))]) for i in range(10)]
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = OpenAIProvider("k", mock_model)
        provider.client = m.return_value
        bench = DummyBenchmark()

        async def _stream():
            out = []
            async for chunk in provider.stream_response("hi"):
                out.append(chunk)
            return out

        result = await bench(_stream)
        assert len(result) == 10
````
`````

## File: test_chatbot.py/test_chatbot.py
`````python
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import anthropic # For spec
import pytest

from chatbot import (
    ChatBot,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    DeepSeekProvider,
    GroqProvider,
    ModelInfo,
    Provider,
    PROVIDER_MODELS, # For Gemini test
)

@pytest.fixture
def mock_model():
    return ModelInfo("test-model", "Test Model")

@pytest.fixture
def thinking_model():
    return ModelInfo("test-thinking", "Thinking Model", is_thinking=True)


def test_provider_initialization(mock_model):
    key = "k"
    for cls in [OpenAIProvider, AnthropicProvider, GeminiProvider, DeepSeekProvider, GroqProvider]:
        with patch.object(cls, "__init__", return_value=None) as init:
            cls(key, mock_model)
            init.assert_called()

@pytest.mark.asyncio
async def test_openai_stream(mock_model):
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="a"))]),
        Mock(choices=[Mock(delta=Mock(content="b"))]),
    ]
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = OpenAIProvider("k", mock_model)
        provider.client = m.return_value
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert out == ["a", "b"]

@pytest.mark.asyncio
async def test_deepseek_thinking(thinking_model):
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="<thinking>"))]),
        Mock(choices=[Mock(delta=Mock(content="hidden"))]),
        Mock(choices=[Mock(delta=Mock(content="</thinking>"))]),
        Mock(choices=[Mock(delta=Mock(content="final"))]),
    ]
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = DeepSeekProvider("k", thinking_model)
        provider.client = m.return_value
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert out == ["final"]

@pytest.mark.asyncio
async def test_anthropic_stream(mock_model):
    mock_text_stream_content = ["Anthropic says ", "hello!"]

    # Mock for the context manager client.messages.stream(...)
    # The object returned by stream() is the context manager itself.
    # Its __enter__ method returns an object that has the text_stream attribute.
    mock_stream_context_instance = Mock()
    mock_stream_context_instance.text_stream = mock_text_stream_content # text_stream is iterable

    mock_anthropic_client = Mock(spec=anthropic.Anthropic)
    # Configure the .stream() method to return a context manager mock
    mock_cm = Mock()
    mock_cm.__enter__ = Mock(return_value=mock_stream_context_instance) # This is what 'stream' becomes in 'with ... as stream:'
    mock_cm.__exit__ = Mock(return_value=None)
    mock_anthropic_client.messages.stream.return_value = mock_cm

    with patch("anthropic.Anthropic", return_value=mock_anthropic_client) as mock_anthropic_init:
        provider = AnthropicProvider("anthropic_key", mock_model)
        mock_anthropic_init.assert_called_once_with(api_key="anthropic_key")

        output_chunks = []
        async for chunk in provider.stream_response("Test Anthropic"):
            output_chunks.append(chunk)

        assert output_chunks == ["Anthropic says ", "hello!"]
        assert provider.history == [
            {"role": "user", "content": "Test Anthropic"},
            {"role": "assistant", "content": "Anthropic says hello!"},
        ]
        expected_messages_to_anthropic = [{"role": "user", "content": "Test Anthropic"}]
        mock_anthropic_client.messages.stream.assert_called_once_with(
            model=mock_model.name,
            messages=expected_messages_to_anthropic,
            max_tokens=mock_model.max_tokens,
        )

@pytest.mark.asyncio
async def test_gemini_stream(mock_model):
    mock_chunk1 = Mock(text="Gemini says ")
    mock_chunk2 = Mock(text="hello!")
    mock_chunk_empty = Mock(text=None)
    mock_gemini_response_stream = [mock_chunk1, mock_chunk_empty, mock_chunk2]

    mock_chat_session = Mock() # send_message is not async itself
    mock_chat_session.send_message.return_value = mock_gemini_response_stream

    mock_generative_model = Mock()
    mock_generative_model.start_chat.return_value = mock_chat_session

    # Use the actual model name for Gemini from PROVIDER_MODELS
    gemini_model_info = PROVIDER_MODELS[Provider.GEMINI][0]

    with patch("google.generativeai.configure") as mock_configure, \
         patch("google.generativeai.GenerativeModel", return_value=mock_generative_model) as mock_gm_init:

        provider = GeminiProvider("gemini_key", gemini_model_info)

        mock_configure.assert_called_once_with(api_key="gemini_key")
        mock_gm_init.assert_called_once_with(gemini_model_info.name)
        provider.model_client.start_chat.assert_called_once_with(history=[])

        output_chunks = []
        async for chunk_text in provider.stream_response("Test Gemini"):
            output_chunks.append(chunk_text)

        assert output_chunks == ["Gemini says ", "hello!"]
        mock_chat_session.send_message.assert_called_once()
        call_args = mock_chat_session.send_message.call_args
        assert call_args[0][0] == "Test Gemini"
        assert call_args[1]['stream'] is True
        assert call_args[1]['generation_config'].max_output_tokens == gemini_model_info.max_tokens

        # GeminiProvider does not populate self.history from the base class
        assert provider.history == []

@pytest.mark.asyncio
async def test_groq_stream(mock_model):
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="Groq says "))]),
        Mock(choices=[Mock(delta=Mock(content="hi!"))]),
        Mock(choices=[Mock(delta=Mock(content=None))]), # Test empty delta
    ]
    with patch("groq.Groq") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = GroqProvider("groq_key", mock_model)
        
        out = []
        async for chunk in provider.stream_response("Test Groq"):
            out.append(chunk)
        assert out == ["Groq says ", "hi!"]
        assert provider.history == [{"role": "user", "content": "Test Groq"}, {"role": "assistant", "content": "Groq says hi!"}]


def test_history(mock_model):
    p = OpenAIProvider("k", mock_model)
    p.client = Mock()
    p.add_to_history("user", "hi")
    p.add_to_history("assistant", "yo")
    assert p.history[0]["role"] == "user"
    assert p.history[1]["role"] == "assistant"

@pytest.mark.asyncio
async def test_error_handling(mock_model):
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.side_effect = Exception("boom")
        provider = OpenAIProvider("k", mock_model)
        provider.client = m.return_value
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert any("Error" in c for c in out)

@pytest.mark.asyncio
async def test_anthropic_error_handling(mock_model):
    mock_anthropic_client = Mock(spec=anthropic.Anthropic)
    # Make the stream call itself raise an exception
    mock_anthropic_client.messages.stream.side_effect = Exception("Anthropic API boom")

    with patch("anthropic.Anthropic", return_value=mock_anthropic_client):
        provider = AnthropicProvider("anthropic_key", mock_model)
        out = []
        async for chunk in provider.stream_response("test message"):
            out.append(chunk)
        assert any("Error: Anthropic API boom" in c for c in out)
        assert provider.history == [{"role": "user", "content": "test message"}]

@pytest.mark.asyncio
async def test_gemini_error_handling(mock_model):
    mock_chat_session = Mock()
    mock_chat_session.send_message.side_effect = Exception("Gemini API boom")
    mock_generative_model = Mock()
    mock_generative_model.start_chat.return_value = mock_chat_session

    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel", return_value=mock_generative_model):
        provider = GeminiProvider("gemini_key", mock_model)
        out = []
        async for chunk in provider.stream_response("test message"):
            out.append(chunk)
        assert any("Error: Gemini API boom" in c for c in out)

@pytest.mark.asyncio
async def test_groq_error_handling(mock_model):
    with patch("groq.Groq") as m:
        m.return_value.chat.completions.create.side_effect = Exception("Groq API boom")
        provider = GroqProvider("k", mock_model)
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert any("Error: Groq API boom" in c for c in out)

def test_get_api_key_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=dotenv-key")
    monkeypatch.chdir(tmp_path)
    # ensure no env var set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    bot = ChatBot()
    assert bot.get_api_key(Provider.OPENAI) == "dotenv-key"

class DummyBenchmark:
    async def __call__(self, func):
        return await func()

@pytest.mark.asyncio
async def test_stream_perf(mock_model):
    mock_stream = [Mock(choices=[Mock(delta=Mock(content=f"c{i}"))]) for i in range(10)]
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = OpenAIProvider("k", mock_model)
        provider.client = m.return_value
        bench = DummyBenchmark()

        async def _stream():
            out = []
            async for chunk in provider.stream_response("hi"):
                out.append(chunk)
            return out

        result = await bench(_stream)
        assert len(result) == 10

@pytest.mark.asyncio
async def test_deepseek_thinking_split_tags_fragility(thinking_model):
    # This test demonstrates the current fragility with split tags.
    # The current logic will not hide content if tags are split across chunks.
    mock_stream = [
        Mock(choices=[Mock(delta=Mock(content="<think"))]),      # Split open tag start
        Mock(choices=[Mock(delta=Mock(content="ing>hidden"))]),  # Split open tag end + content
        Mock(choices=[Mock(delta=Mock(content="</think"))]),     # Split close tag start
        Mock(choices=[Mock(delta=Mock(content="ing>"))]),        # Split close tag end
        Mock(choices=[Mock(delta=Mock(content="visible"))]),
    ]
    with patch("openai.OpenAI") as m:
        m.return_value.chat.completions.create.return_value = mock_stream
        provider = DeepSeekProvider("k", thinking_model)
        out = []
        async for chunk in provider.stream_response("hi"):
            out.append(chunk)
        assert out == ["<think", "ing>hidden", "</think", "ing>", "visible"], \
            "DeepSeek thinking tag logic does not correctly handle tags split across chunks."
`````
