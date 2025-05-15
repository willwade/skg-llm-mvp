#!/bin/bash

# Install Simon Willison's LLM library and plugins
pip install llm
pip install llm-gemini
pip install llm-openai
pip install llm-ollama

# Set up environment variables
echo "Setting up environment variables..."
if [ -n "$GEMINI_API_KEY" ]; then
    echo "GEMINI_API_KEY is set"
    # Configure LLM to use Gemini
    llm keys set gemini "$GEMINI_API_KEY"
else
    echo "GEMINI_API_KEY is not set"
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY is set"
    # Configure LLM to use OpenAI
    llm keys set openai "$OPENAI_API_KEY"
else
    echo "OPENAI_API_KEY is not set"
fi

if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN is set"
    # Configure Hugging Face token
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "HF_TOKEN is not set"
fi

# List available models
echo "Available LLM models:"
llm models

echo "Setup complete!"
