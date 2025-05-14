from transformers import pipeline
import json
import argparse
import os
import sys
import subprocess
import requests
from typing import List, Dict, Any, Optional, Union
import time


# Check for Hugging Face token
def check_hf_token():
    """Check if a Hugging Face token is properly set up."""
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    if not token:
        print("\nWarning: No Hugging Face token found in environment variables.")
        print(
            "To use gated models like Gemma, you need to set up a token with the right permissions."
        )
        print("1. Create a token at https://huggingface.co/settings/tokens")
        print("2. Make sure to enable 'Access to public gated repositories'")
        print("3. Set it as an environment variable:")
        print("   export HUGGING_FACE_HUB_TOKEN=your_token_here")
        return False

    return True


def load_social_graph(file_path="social_graph.json"):
    """Load the social graph from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def get_person_info(social_graph, person_id):
    """Get information about a person from the social graph."""
    if person_id in social_graph["people"]:
        return social_graph["people"][person_id]
    else:
        available_people = ", ".join(social_graph["people"].keys())
        raise ValueError(
            f"Person '{person_id}' not found in social graph. Available people: {available_people}"
        )


def build_enhanced_prompt(social_graph, person_id, topic=None, user_message=None):
    """Build an enhanced prompt using social graph information."""
    # Get AAC user information
    aac_user = social_graph["aac_user"]

    # Get conversation partner information
    person = get_person_info(social_graph, person_id)

    # Start building the prompt with AAC user information
    prompt = f"""I am {aac_user['name']}, a {aac_user['age']}-year-old with MND (Motor Neuron Disease) from {aac_user['location']}.
{aac_user['background']}

My communication needs: {aac_user['communication_needs']}

I am talking to {person['name']}, who is my {person['role']}.
About {person['name']}: {person['context']}
We typically talk about: {', '.join(person['topics'])}
We communicate {person['frequency']}.
"""

    # Add places information if available
    if "places" in social_graph:
        relevant_places = social_graph["places"][
            :3
        ]  # Just use a few places for context
        prompt += f"\nPlaces important to me: {', '.join(relevant_places)}\n"

    # Add communication style based on relationship
    if person["role"] in ["wife", "son", "daughter", "mother", "father"]:
        prompt += "I communicate with my family in a warm, loving way, sometimes using inside jokes.\n"
    elif person["role"] in ["doctor", "therapist", "nurse"]:
        prompt += (
            "I communicate with healthcare providers in a direct, informative way.\n"
        )
    elif person["role"] in ["best mate", "friend"]:
        prompt += "I communicate with friends casually, often with humor and sometimes swearing.\n"
    elif person["role"] in ["work colleague", "boss"]:
        prompt += "I communicate with colleagues professionally but still friendly.\n"

    # Add common utterances by category if available
    if "common_utterances" in social_graph:
        # Try to find relevant utterance category based on topic
        utterance_category = None
        if topic == "football" or topic == "sports":
            utterance_category = "sports_talk"
        elif topic == "programming" or topic == "tech news":
            utterance_category = "tech_talk"
        elif topic in ["family plans", "children's activities"]:
            utterance_category = "family_talk"

        # Add relevant utterances if category exists
        if (
            utterance_category
            and utterance_category in social_graph["common_utterances"]
        ):
            utterances = social_graph["common_utterances"][utterance_category][:2]
            prompt += f"\nI might say things like: {' or '.join(utterances)}\n"

    # Add topic information if provided
    if topic and topic in person["topics"]:
        prompt += f"\nWe are currently discussing {topic}.\n"

        # Add specific context about this topic with this person
        if topic == "football" and "Manchester United" in person["context"]:
            prompt += (
                "We both support Manchester United and often discuss recent matches.\n"
            )
        elif topic == "programming" and "software developer" in person["context"]:
            prompt += (
                "We both work in software development and share technical interests.\n"
            )
        elif topic == "family plans" and person["role"] in ["wife", "husband"]:
            prompt += "We make family decisions together, considering my condition.\n"
        elif topic == "old scout adventures" and person["role"] == "best mate":
            prompt += "We often reminisce about our Scout camping trips in South East London.\n"
        elif topic == "cycling" and "cycling" in person["context"]:
            prompt += "I miss being able to cycle but enjoy talking about past cycling adventures.\n"

        # Add shared experiences based on relationship and topic
        if person["role"] == "best mate" and topic in ["football", "pub quizzes"]:
            prompt += (
                "We've watched many matches together and done countless pub quizzes.\n"
            )
        elif person["role"] == "wife" and topic in ["family plans", "weekend outings"]:
            prompt += "Emma has been amazing at keeping family life as normal as possible despite my condition.\n"
        elif person["role"] == "son" and topic == "football":
            prompt += "I try to stay engaged with Billy's football enthusiasm even as my condition progresses.\n"

    # Add the user's message if provided
    if user_message:
        prompt += f"\n{person['name']} just said to me: \"{user_message}\"\n"
    else:
        # Use a common phrase from the person if no message is provided
        if person["common_phrases"]:
            default_message = person["common_phrases"][0]
            prompt += f"\n{person['name']} just said to me: \"{default_message}\"\n"

    # Add the response prompt with specific guidance
    prompt += f"""
I want to respond to {person['name']} in a way that is natural, brief (1-2 sentences), and directly relevant to what they just said. I'll use casual language with some humor since we're close friends.

My response to {person['name']}:"""

    return prompt


class LLMInterface:
    """Base interface for language model generation."""

    def __init__(self, model_name, max_length=150, temperature=0.9):
        """Initialize the LLM interface.

        Args:
            model_name: Name or path of the model
            max_length: Maximum length of generated text
            temperature: Controls randomness (higher = more random)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature

    def generate(self, prompt, num_responses=3):
        """Generate responses for the given prompt.

        Args:
            prompt: The prompt to generate responses for
            num_responses: Number of responses to generate

        Returns:
            A list of generated responses
        """
        raise NotImplementedError("Subclasses must implement this method")

    def cleanup_response(self, text):
        """Clean up a generated response.

        Args:
            text: The raw generated text

        Returns:
            Cleaned up text
        """
        # Make sure it's a complete sentence or phrase
        # If it ends abruptly, add an ellipsis
        if text and not any(text.endswith(end) for end in [".", "!", "?", '..."']):
            if text.endswith('"'):
                text = text[:-1] + '..."'
            else:
                text += "..."

        return text


class HuggingFaceInterface(LLMInterface):
    """Interface for Hugging Face Transformers models."""

    def __init__(self, model_name="distilgpt2", max_length=150, temperature=0.9):
        """Initialize the Hugging Face interface."""
        super().__init__(model_name, max_length, temperature)
        try:
            # Check if we're dealing with a gated model
            is_gated_model = any(
                name in model_name for name in ["gemma", "llama", "mistral"]
            )

            # Get token from environment
            import os

            token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get(
                "HF_TOKEN"
            )

            if is_gated_model and token:
                print(f"Using token for gated model: {model_name}")
                from huggingface_hub import login

                login(token=token, add_to_git_credential=False)

                # Explicitly pass token to pipeline
                from transformers import AutoTokenizer, AutoModelForCausalLM

                tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
                model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
                self.pipeline = pipeline(
                    "text-generation", model=model, tokenizer=tokenizer
                )
            else:
                self.pipeline = pipeline("text-generation", model=model_name)

            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            if "gated" in str(e).lower() or "403" in str(e):
                print(
                    "\nThis appears to be a gated model that requires authentication."
                )
                print("Please make sure you:")
                print("1. Have accepted the model license on the Hugging Face Hub")
                print(
                    "2. Have created a token with 'Access to public gated repositories' permission"
                )
                print(
                    "3. Have set the token as HUGGING_FACE_HUB_TOKEN environment variable"
                )
                print("\nAlternatively, try using the Ollama backend:")
                print(
                    f"python demo.py --backend ollama --model gemma:7b-it [other args]"
                )
            raise

    def generate(self, prompt, num_responses=3):
        """Generate responses using the Hugging Face pipeline."""
        # Calculate prompt length in tokens (approximate)
        prompt_length = len(prompt.split())

        # Generate the responses
        responses = self.pipeline(
            prompt,
            max_length=prompt_length + self.max_length,
            temperature=self.temperature,
            do_sample=True,
            num_return_sequences=num_responses,
            top_p=0.92,
            top_k=50,
            truncation=True,
        )

        # Extract just the generated parts (not the prompt)
        generated_texts = []
        for resp in responses:
            # Get the text after the prompt
            generated = resp["generated_text"][len(prompt) :].strip()

            # Clean up the response
            generated = self.cleanup_response(generated)

            # Add to our list if it's not empty
            if generated:
                generated_texts.append(generated)

        return generated_texts


class OllamaInterface(LLMInterface):
    """Interface for Ollama models."""

    def __init__(self, model_name="gemma:7b", max_length=150, temperature=0.9):
        """Initialize the Ollama interface."""
        super().__init__(model_name, max_length, temperature)
        # Check if Ollama is installed and the model is available
        try:
            import requests

            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = [model["name"] for model in response.json()["models"]]
                if model_name not in models:
                    print(
                        f"Warning: Model {model_name} not found in Ollama. Available models: {', '.join(models)}"
                    )
                    print(f"You may need to run: ollama pull {model_name}")
            print(f"Ollama is available and will use model: {model_name}")
        except Exception as e:
            print(f"Warning: Ollama may not be installed or running: {e}")
            print("You can install Ollama from https://ollama.ai/")

    def generate(self, prompt, num_responses=3):
        """Generate responses using Ollama API."""
        import requests

        generated_texts = []
        for _ in range(num_responses):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": self.temperature,
                        "max_tokens": self.max_length,
                    },
                    stream=False,
                )

                if response.status_code == 200:
                    # Extract the generated text
                    generated = response.json().get("response", "").strip()

                    # Clean up the response
                    generated = self.cleanup_response(generated)

                    # Add to our list if it's not empty
                    if generated:
                        generated_texts.append(generated)
                else:
                    print(f"Error from Ollama API: {response.text}")
            except Exception as e:
                print(f"Error generating with Ollama: {e}")

        return generated_texts


class LLMToolInterface(LLMInterface):
    """Interface for Simon Willison's LLM tool."""

    def __init__(
        self, model_name="gemini-1.5-pro-latest", max_length=150, temperature=0.9
    ):
        """Initialize the LLM tool interface."""
        super().__init__(model_name, max_length, temperature)
        # Check if LLM tool is installed
        try:
            import subprocess

            result = subprocess.run(["llm", "models"], capture_output=True, text=True)
            if result.returncode == 0:
                models = [
                    line.strip() for line in result.stdout.split("\n") if line.strip()
                ]
                print(f"LLM tool is available. Found {len(models)} models.")

                # Check for specific model types
                gemini_models = [
                    m for m in models if "gemini" in m.lower() or "gemma" in m.lower()
                ]
                if gemini_models:
                    print(f"Gemini models available: {', '.join(gemini_models[:3])}...")

                # Check for Ollama models
                ollama_models = [m for m in models if "ollama" in m.lower()]
                if ollama_models:
                    print(f"Ollama models available: {', '.join(ollama_models[:3])}...")

                # Check for MLX models
                mlx_models = [m for m in models if "mlx" in m.lower()]
                if mlx_models:
                    print(f"MLX models available: {', '.join(mlx_models[:3])}...")

                # Check if the specified model is available
                if not any(self.model_name in m for m in models):
                    print(
                        f"Warning: Model '{self.model_name}' not found in available models."
                    )
                    print("You may need to install the appropriate plugin:")
                    if (
                        "gemini" in self.model_name.lower()
                        or "gemma" in self.model_name.lower()
                    ):
                        print("llm install llm-gemini")
                    elif "mlx" in self.model_name.lower():
                        print("llm install llm-mlx")
                    elif "ollama" in self.model_name.lower():
                        print("llm install llm-ollama")
                        model_name = self.model_name
                        if "/" in model_name:
                            model_name = model_name.split("/")[1]
                        print("ollama pull " + model_name)
            else:
                print("Warning: LLM tool may be installed but returned an error.")
        except Exception as e:
            print(f"Warning: Simon Willison's LLM tool may not be installed: {e}")
            print("You can install it with: pip install llm")

    def generate(self, prompt, num_responses=3):
        """Generate responses using the LLM tool."""
        import subprocess
        import os

        # Check for required environment variables
        if "gemini" in self.model_name.lower() or "gemma" in self.model_name.lower():
            if not os.environ.get("GEMINI_API_KEY"):
                print("Warning: GEMINI_API_KEY environment variable not found.")
                print("Gemini API may not work without it.")
        elif "ollama" in self.model_name.lower():
            # Check if Ollama is running
            try:
                import requests

                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code != 200:
                    print("Warning: Ollama server doesn't seem to be running.")
                    print("Start Ollama with: ollama serve")
            except Exception:
                print("Warning: Ollama server doesn't seem to be running.")
                print("Start Ollama with: ollama serve")

        # Determine the appropriate parameter name for max tokens
        if "gemini" in self.model_name.lower() or "gemma" in self.model_name.lower():
            max_tokens_param = "max_output_tokens"
        elif "ollama" in self.model_name.lower():
            max_tokens_param = "num_predict"
        else:
            max_tokens_param = "max_tokens"

        generated_texts = []
        for _ in range(num_responses):
            try:
                # Call the LLM tool
                result = subprocess.run(
                    [
                        "llm",
                        "-m",
                        self.model_name,
                        "-s",
                        f"temperature={self.temperature}",
                        "-s",
                        f"{max_tokens_param}={self.max_length}",
                        prompt,
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    # Get the generated text
                    generated = result.stdout.strip()

                    # Clean up the response
                    generated = self.cleanup_response(generated)

                    # Add to our list if it's not empty
                    if generated:
                        generated_texts.append(generated)
                else:
                    print(f"Error from LLM tool: {result.stderr}")
            except Exception as e:
                print(f"Error generating with LLM tool: {e}")

        return generated_texts


class MLXInterface(LLMInterface):
    """Interface for MLX-powered models on Mac."""

    def __init__(
        self, model_name="mlx-community/gemma-7b-it", max_length=150, temperature=0.9
    ):
        """Initialize the MLX interface."""
        super().__init__(model_name, max_length, temperature)
        # Check if MLX is installed
        try:
            import importlib.util

            if importlib.util.find_spec("mlx") is not None:
                print("MLX is available for optimized inference on Mac")
            else:
                print("Warning: MLX is not installed. Install with: pip install mlx")
        except Exception as e:
            print(f"Warning: Error checking for MLX: {e}")

    def generate(self, prompt, num_responses=3):
        """Generate responses using MLX."""
        try:
            # Dynamically import MLX to avoid errors on non-Mac platforms
            import mlx.core as mx
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Load the model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True, mx_dtype=mx.float16
            )

            generated_texts = []
            for _ in range(num_responses):
                # Tokenize the prompt
                inputs = tokenizer(prompt, return_tensors="np")

                # Generate
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=len(inputs["input_ids"][0]) + self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                )

                # Decode the generated tokens
                generated = tokenizer.decode(
                    outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
                )

                # Clean up the response
                generated = self.cleanup_response(generated)

                # Add to our list if it's not empty
                if generated:
                    generated_texts.append(generated)

            return generated_texts
        except Exception as e:
            print(f"Error generating with MLX: {e}")
            return []


def create_llm_interface(backend, model_name, max_length=150, temperature=0.9):
    """Create an appropriate LLM interface based on the backend.

    Args:
        backend: The backend to use ('hf', 'llm')
        model_name: The name of the model to use
        max_length: Maximum length of generated text
        temperature: Controls randomness (higher = more random)

    Returns:
        An LLM interface instance
    """
    if backend == "hf":
        return HuggingFaceInterface(model_name, max_length, temperature)
    elif backend == "llm":
        return LLMToolInterface(model_name, max_length, temperature)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def generate_response(
    prompt,
    model_name="distilgpt2",
    max_length=150,
    temperature=0.9,
    num_responses=3,
    backend="hf",
):
    """Generate multiple responses using the specified model and backend.

    Args:
        prompt: The prompt to generate responses for
        model_name: The name of the model to use
        max_length: Maximum number of new tokens to generate
        temperature: Controls randomness (higher = more random)
        num_responses: Number of different responses to generate
        backend: The backend to use ('hf', 'ollama', 'llm', 'mlx')

    Returns:
        A list of generated responses
    """
    # Create the appropriate interface
    interface = create_llm_interface(backend, model_name, max_length, temperature)

    # Generate responses
    return interface.generate(prompt, num_responses)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate AAC responses using social graph context"
    )
    parser.add_argument(
        "--person", default="billy", help="Person ID from the social graph"
    )
    parser.add_argument("--topic", help="Topic of conversation")
    parser.add_argument("--message", help="Message from the conversation partner")
    parser.add_argument(
        "--backend",
        default="llm",
        choices=["hf", "llm"],
        help="Backend to use for generation (hf=HuggingFace, "
        "llm=Simon Willison's LLM tool with support for Gemini/MLX/Ollama)",
    )
    parser.add_argument(
        "--model",
        default="gemini-1.5-pro-latest",
        help="Model to use for generation. Recommended models by backend:\n"
        "- hf: 'distilgpt2', 'gpt2-medium', 'google/gemma-2b-it'\n"
        "- llm: 'gemini-1.5-pro-latest', 'gemma-3-27b-it' (requires llm-gemini plugin)\n"
        "       'mlx-community/gemma-7b-it' (requires llm-mlx plugin)\n"
        "       'Ollama: gemma3:4b-it-qat', 'Ollama: llama3:8b' (requires llm-ollama plugin)",
    )
    parser.add_argument(
        "--num_responses", type=int, default=3, help="Number of responses to generate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=150,
        help="Maximum length of generated responses",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Temperature for generation (higher = more creative)",
    )
    args = parser.parse_args()

    # Check for token if using HF backend with gated models
    if args.backend == "hf" and any(
        name in args.model for name in ["gemma", "llama", "mistral"]
    ):
        if not check_hf_token():
            print("\nSuggestion: Try using the LLM tool with Gemini API instead:")
            print(
                f"python demo.py --backend llm --model gemini-1.5-pro-latest --person {args.person}"
                + (f' --topic "{args.topic}"' if args.topic else "")
                + (f' --message "{args.message}"' if args.message else "")
            )
            print("\nOr use a non-gated model:")
            print(
                f"python demo.py --backend hf --model gpt2-medium --person {args.person}"
                + (f' --topic "{args.topic}"' if args.topic else "")
                + (f' --message "{args.message}"' if args.message else "")
            )
            print("\nContinuing anyway, but expect authentication errors...\n")

    # Load the social graph
    social_graph = load_social_graph()

    # Build the prompt
    prompt = build_enhanced_prompt(social_graph, args.person, args.topic, args.message)

    print("\n=== PROMPT ===")
    print(prompt)
    print(
        f"\n=== GENERATING RESPONSE USING {args.backend.upper()} BACKEND WITH MODEL {args.model} ==="
    )

    # Generate the responses
    try:
        responses = generate_response(
            prompt,
            args.model,
            max_length=args.max_length,
            num_responses=args.num_responses,
            temperature=args.temperature,
            backend=args.backend,
        )

        print("\n=== RESPONSES ===")
        for i, response in enumerate(responses, 1):
            print(f"{i}. {response}")
            print()
    except Exception as e:
        print(f"\nError generating responses: {e}")

        if args.backend == "hf" and any(
            name in args.model for name in ["gemma", "llama", "mistral"]
        ):
            print("\nThis appears to be an authentication issue with a gated model.")
            print("Try using the LLM tool with Gemini API instead:")
            print(
                f"python demo.py --backend llm --model gemini-1.5-pro-latest --person {args.person}"
                + (f' --topic "{args.topic}"' if args.topic else "")
                + (f' --message "{args.message}"' if args.message else "")
            )
        # Ollama is now handled through the llm backend
        elif args.backend == "llm":
            if "gemini" in args.model.lower() or "gemma" in args.model.lower():
                print(
                    "\nMake sure you have the GEMINI_API_KEY environment variable set:"
                )
                print("export GEMINI_API_KEY=your_api_key")
                print("\nAnd make sure llm-gemini is installed:")
                print("llm install llm-gemini")
            elif "mlx" in args.model.lower():
                print("\nMake sure llm-mlx is installed:")
                print("llm install llm-mlx")
            elif "ollama" in args.model.lower():
                print("\nMake sure Ollama is installed and running:")
                print("1. Install from https://ollama.ai/")
                print("2. Start Ollama with: ollama serve")
                print("3. Install the llm-ollama plugin: llm install llm-ollama")
                model_name = args.model
                if "ollama:" in model_name.lower():
                    model_name = model_name.replace("Ollama: ", "")
                elif "/" in model_name:
                    model_name = model_name.split("/")[1]
                print(f"4. Pull the model: ollama pull {model_name}")
            else:
                print("\nMake sure Simon Willison's LLM tool is installed:")
                print("pip install llm")


# If running as a script
if __name__ == "__main__":
    main()
