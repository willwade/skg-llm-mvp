import json
import random
import threading
import time
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline


class SocialGraphManager:
    """Manages the social graph and provides context for the AAC system."""

    def __init__(self, graph_path: str = "social_graph.json"):
        """Initialize the social graph manager.

        Args:
            graph_path: Path to the social graph JSON file
        """
        self.graph_path = graph_path
        self.graph = self._load_graph()

        # Initialize sentence transformer for semantic matching
        try:
            self.sentence_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            self.embeddings_cache = {}
            self._initialize_embeddings()
        except Exception as e:
            self.sentence_model = None

    def _load_graph(self) -> Dict[str, Any]:
        """Load the social graph from the JSON file."""
        try:
            with open(self.graph_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"people": {}, "places": [], "topics": []}

    def _initialize_embeddings(self):
        """Initialize embeddings for topics and phrases in the social graph."""
        if not self.sentence_model:
            return

        # Create embeddings for topics
        topics = self.graph.get("topics", [])
        for topic in topics:
            if topic not in self.embeddings_cache:
                self.embeddings_cache[topic] = self.sentence_model.encode(topic)

        # Create embeddings for common phrases
        for person_id, person_data in self.graph.get("people", {}).items():
            for phrase in person_data.get("common_phrases", []):
                if phrase not in self.embeddings_cache:
                    self.embeddings_cache[phrase] = self.sentence_model.encode(phrase)

        # Create embeddings for common utterances
        for category, utterances in self.graph.get("common_utterances", {}).items():
            for utterance in utterances:
                if utterance not in self.embeddings_cache:
                    self.embeddings_cache[utterance] = self.sentence_model.encode(
                        utterance
                    )

    def get_people_list(self) -> List[Dict[str, str]]:
        """Get a list of people from the social graph with their names and roles."""
        people = []
        for person_id, person_data in self.graph.get("people", {}).items():
            people.append(
                {
                    "id": person_id,
                    "name": person_data.get("name", person_id),
                    "role": person_data.get("role", ""),
                }
            )
        return people

    def get_person_context(self, person_id: str) -> Dict[str, Any]:
        """Get context information for a specific person."""
        # Check if the person_id contains a display name (e.g., "Emma (wife)")
        # and try to extract the actual ID
        if person_id not in self.graph.get("people", {}):
            # Try to find the person by name
            for pid, pdata in self.graph.get("people", {}).items():
                name = pdata.get("name", "")
                role = pdata.get("role", "")
                if f"{name} ({role})" == person_id:
                    person_id = pid
                    break

        # If still not found, return empty dict
        if person_id not in self.graph.get("people", {}):
            return {}

        person_data = self.graph["people"][person_id]
        return person_data

    def get_relevant_phrases(
        self, person_id: str, user_input: Optional[str] = None
    ) -> List[str]:
        """Get relevant phrases for a specific person based on user input."""
        if person_id not in self.graph.get("people", {}):
            return []

        person_data = self.graph["people"][person_id]
        phrases = person_data.get("common_phrases", [])

        # If no user input, return random phrases
        if not user_input or not self.sentence_model:
            return random.sample(phrases, min(3, len(phrases)))

        # Use semantic search to find relevant phrases
        user_embedding = self.sentence_model.encode(user_input)
        phrase_scores = []

        for phrase in phrases:
            if phrase in self.embeddings_cache:
                phrase_embedding = self.embeddings_cache[phrase]
            else:
                phrase_embedding = self.sentence_model.encode(phrase)
                self.embeddings_cache[phrase] = phrase_embedding

            similarity = np.dot(user_embedding, phrase_embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(phrase_embedding)
            )
            phrase_scores.append((phrase, similarity))

        # Sort by similarity score and return top phrases
        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in phrase_scores[:3]]

    def get_common_utterances(self, category: Optional[str] = None) -> List[str]:
        """Get common utterances from the social graph, optionally filtered by category."""
        utterances = []

        if "common_utterances" not in self.graph:
            return utterances

        if category and category in self.graph["common_utterances"]:
            return self.graph["common_utterances"][category]

        # If no category specified, return a sample from each category
        for category_utterances in self.graph["common_utterances"].values():
            utterances.extend(
                random.sample(category_utterances, min(2, len(category_utterances)))
            )

        return utterances


class SuggestionGenerator:
    """Generates contextual suggestions for the AAC system."""

    def __init__(self, model_name: str = "distilgpt2"):
        """Initialize the suggestion generator.

        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.model_name = model_name
        self.model_loaded = False
        self.generator = None
        self.aac_user_info = None
        self.loaded_models = {}  # Cache for loaded models

        # Load AAC user information from social graph
        try:
            with open("social_graph.json", "r") as f:
                social_graph = json.load(f)
                self.aac_user_info = social_graph.get("aac_user", {})
        except Exception as e:
            print(f"Error loading AAC user info from social graph: {e}")
            self.aac_user_info = {}

        # Try to load the model
        self.load_model(model_name)

        # Fallback responses if model fails to load or generate
        self.fallback_responses = [
            "I'm not sure how to respond to that.",
            "That's interesting. Tell me more.",
            "I'd like to talk about that further.",
            "I appreciate you sharing that with me.",
            "Could we talk about something else?",
            "I need some time to think about that.",
        ]

    def load_model(self, model_name: str) -> bool:
        """Load a model (either Hugging Face model or API-based model).

        Args:
            model_name: Name of the model to use (HuggingFace model name or API identifier)

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        self.model_name = model_name
        self.model_loaded = False

        # Check if model is already loaded in cache
        if model_name in self.loaded_models:
            print(f"Using cached model: {model_name}")
            self.generator = self.loaded_models[model_name]
            self.model_loaded = True
            return True

        # Check if this is a Gemini API model
        if model_name.startswith("gemini-api:"):
            try:
                import os
                import google.generativeai as genai

                # Get API key from environment
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    print("No GEMINI_API_KEY found in environment variables.")
                    print("Please set the GEMINI_API_KEY environment variable.")
                    return False

                # Configure the Gemini API
                genai.configure(api_key=api_key)

                # Extract the specific model name after the prefix
                gemini_model = model_name.split(":", 1)[1]
                print(f"Using Gemini API with model: {gemini_model}")

                # Store the model name and API client in the generator
                self.generator = {
                    "type": "gemini-api",
                    "model": gemini_model,
                    "client": genai,
                }

                # Cache the API client
                self.loaded_models[model_name] = self.generator

                self.model_loaded = True
                print(f"Gemini API configured successfully for model: {gemini_model}")
                return True

            except Exception as e:
                print(f"Error configuring Gemini API: {e}")
                self.model_loaded = False
                return False

        # Otherwise, try to load a Hugging Face model
        try:
            print(f"Loading Hugging Face model: {model_name}")

            # Check if this is a gated model that requires authentication
            is_gated_model = any(
                name in model_name.lower()
                for name in ["gemma", "llama", "mistral", "qwen", "phi"]
            )

            if is_gated_model:
                # Try to get token from environment
                import os
                import torch
                import time
                from transformers import BitsAndBytesConfig
                from requests.exceptions import ConnectionError, Timeout, HTTPError

                token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get(
                    "HF_TOKEN"
                )

                if token:
                    print(f"Using token for gated model: {model_name}")
                    from huggingface_hub import login

                    login(token=token, add_to_git_credential=False)

                    # Explicitly pass token to pipeline
                    from transformers import AutoTokenizer, AutoModelForCausalLM

                    # Implement retry mechanism for network issues
                    max_retries = 3
                    retry_delay = 2  # seconds

                    for attempt in range(max_retries):
                        try:
                            print(
                                f"Attempt {attempt+1}/{max_retries} to load model: {model_name}"
                            )

                            # First try to load just the tokenizer to check connectivity
                            print(f"Loading tokenizer for {model_name}...")
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_name,
                                token=token,
                                use_fast=True,
                                local_files_only=False,
                            )
                            print(f"Tokenizer loaded successfully for {model_name}")

                            # Configure 4-bit quantization to save memory
                            print("Configuring quantization settings...")
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                            )

                            # Load model with quantization
                            print(f"Loading model {model_name} with quantization...")
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                token=token,
                                quantization_config=quantization_config,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                            )
                            print(
                                f"Model {model_name} loaded successfully with quantization"
                            )

                            # Create pipeline
                            print("Creating text generation pipeline...")
                            self.generator = {
                                "type": "huggingface",
                                "pipeline": pipeline(
                                    "text-generation",
                                    model=model,
                                    tokenizer=tokenizer,
                                    torch_dtype=torch.float16,
                                ),
                            }
                            print("Pipeline created successfully")

                            # If we got here, loading succeeded
                            break

                        except (ConnectionError, Timeout, HTTPError) as network_error:
                            # Handle network-related errors with retries
                            print(
                                f"Network error loading model (attempt {attempt+1}/{max_retries}): {network_error}"
                            )
                            if attempt < max_retries - 1:
                                print(f"Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                print(
                                    "Maximum retries reached, falling back to alternative loading method"
                                )
                                raise network_error

                        except (RuntimeError, ValueError, OSError) as e:
                            # Handle memory errors or other issues
                            print(
                                f"Error loading gated model with token (attempt {attempt+1}/{max_retries}): {e}"
                            )
                            print(
                                "This may be due to memory limitations, network issues, or insufficient permissions."
                            )

                            if "CUDA out of memory" in str(
                                e
                            ) or "DefaultCPUAllocator" in str(e):
                                print(
                                    "Memory error detected. Trying with more aggressive memory optimization..."
                                )
                                break  # Skip to non-quantized version with CPU offloading

                            if attempt < max_retries - 1:
                                print(f"Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                print(
                                    "Maximum retries reached, falling back to alternative loading method"
                                )

                    # If the loop completed without success, try alternative loading methods
                    if not hasattr(self, "generator") or self.generator is None:
                        # Try loading without quantization as fallback
                        try:
                            print(
                                "Trying to load model without quantization (CPU only)..."
                            )
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_name, token=token, use_fast=True
                            )
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                token=token,
                                device_map="cpu",
                                low_cpu_mem_usage=True,
                            )
                            self.generator = {
                                "type": "huggingface",
                                "pipeline": pipeline(
                                    "text-generation", model=model, tokenizer=tokenizer
                                ),
                            }
                            print(
                                "Successfully loaded model on CPU without quantization"
                            )
                        except Exception as e2:
                            print(f"Fallback loading also failed: {e2}")
                            print(
                                "All loading attempts failed. Please try a different model or check your connection."
                            )
                            raise RuntimeError(
                                f"Failed to load model after multiple attempts: {str(e2)}"
                            )
                else:
                    print("No Hugging Face token found in environment variables.")
                    print(
                        "To use gated models like Gemma, you need to set up a token with the right permissions."
                    )
                    print("1. Create a token at https://huggingface.co/settings/tokens")
                    print(
                        "2. Make sure to enable 'Access to public gated repositories'"
                    )
                    print(
                        "3. Set it as an environment variable: export HUGGING_FACE_HUB_TOKEN=your_token_here"
                    )
                    raise ValueError("Authentication token required for gated model")
            else:
                # For non-gated models, use the standard pipeline
                from transformers import pipeline

                self.generator = {
                    "type": "huggingface",
                    "pipeline": pipeline("text-generation", model=model_name),
                }

            # Cache the loaded model
            self.loaded_models[model_name] = self.generator

            self.model_loaded = True
            print(f"Model loaded successfully: {model_name}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            return False

    def _clean_small_model_response(self, response: str) -> str:
        """Clean up responses from small models that often repeat instructions or generate nonsense.

        Args:
            response: The raw response from the model

        Returns:
            A cleaned response
        """
        # If response is too short, return as is
        if len(response) < 5:
            return response

        # Remove common instruction repetitions
        patterns_to_remove = [
            "I want to respond to what",
            "I'll use language appropriate for our relationship",
            "I should speak in first person",
            "I should use language appropriate",
            "I want to respond directly",
            "I'll speak as myself",
            "I want to initiate a conversation",
            "My response should be natural",
            "My response to",
            "Will's response to",
            "Will says to",
        ]

        # Check for and remove these patterns
        cleaned_response = response
        for pattern in patterns_to_remove:
            if pattern in cleaned_response:
                # Find the first occurrence and remove everything from there
                index = cleaned_response.find(pattern)
                if index > 10:  # Keep some beginning text if available
                    cleaned_response = cleaned_response[:index].strip()
                else:
                    # If pattern is at the beginning, remove just that pattern
                    parts = cleaned_response.split(pattern, 1)
                    if len(parts) > 1:
                        cleaned_response = parts[1].strip()

        # Remove any lines that are just the name repeated
        lines = cleaned_response.split("\n")
        cleaned_lines = []
        for line in lines:
            # Skip lines that are just a name repeated
            if line.strip() and not all(
                word == line.split()[0] for word in line.split()
            ):
                cleaned_lines.append(line)

        cleaned_response = "\n".join(cleaned_lines).strip()

        # If we've removed too much, use a fallback
        if len(cleaned_response) < 5:
            return "I'm not sure what to say about that."

        # Limit to first 2 sentences to avoid rambling
        sentences = cleaned_response.split(".")
        if len(sentences) > 2:
            cleaned_response = ".".join(sentences[:2]) + "."

        return cleaned_response

    def _get_mood_description(self, mood_value: int) -> str:
        """Convert mood value (1-5) to a descriptive string.

        Args:
            mood_value: Integer from 1-5 representing mood (1=sad, 5=happy)

        Returns:
            String description of the mood
        """
        mood_descriptions = {
            1: "I'm feeling quite down and sad today. My responses might be more subdued.",
            2: "I'm feeling a bit low today. I might be less enthusiastic than usual.",
            3: "I'm feeling okay today - neither particularly happy nor sad.",
            4: "I'm feeling pretty good today. I'm in a positive mood.",
            5: "I'm feeling really happy and upbeat today! I'm in a great mood.",
        }

        # Default to neutral if value is out of range
        return mood_descriptions.get(mood_value, mood_descriptions[3])

    def test_model(self) -> str:
        """Test if the model is working correctly."""
        if not self.model_loaded:
            return "Model not loaded"

        try:
            # Create a more explicit test prompt that clearly establishes Will's identity and role
            test_prompt = """I am Will, a 38-year-old with MND (Motor Neuron Disease).
I am talking to my 7-year-old son Billy.
Billy just asked me about football.
I want to respond to Billy in a natural, brief way.

My response to Billy:"""
            print(f"Testing model with prompt: {test_prompt}")

            # Check if we're using the Gemini API or a Hugging Face model
            if (
                isinstance(self.generator, dict)
                and self.generator.get("type") == "gemini-api"
            ):
                try:
                    # Use Gemini API
                    genai = self.generator["client"]
                    model_name = self.generator["model"]

                    # Create a generative model
                    model = genai.GenerativeModel(model_name)

                    # Generate content with timeout
                    print("Sending test request to Gemini API...")

                    # Set a timeout for the test
                    import threading
                    import time

                    result = ["No response received yet"]
                    generation_complete = [False]

                    def generate_with_timeout():
                        try:
                            print("Starting Gemini API test request...")
                            response = model.generate_content(test_prompt)
                            print(f"Received response from Gemini API: {response}")

                            if response and hasattr(response, "text"):
                                result[0] = response.text
                                print(f"Extracted text from response: {result[0]}")
                            else:
                                result[0] = "No text in Gemini API response"
                                print("Response object has no text attribute")

                            generation_complete[0] = True
                        except Exception as e:
                            print(f"Error in Gemini test generation: {e}")
                            result[0] = f"Error: {str(e)}"
                            generation_complete[0] = True

                    # Start generation in a separate thread
                    generation_thread = threading.Thread(target=generate_with_timeout)
                    generation_thread.daemon = True
                    generation_thread.start()

                    # Wait for up to 10 seconds
                    timeout = 10
                    start_time = time.time()
                    while (
                        not generation_complete[0]
                        and time.time() - start_time < timeout
                    ):
                        print(
                            f"Waiting for Gemini API response... ({int(time.time() - start_time)}s)"
                        )
                        time.sleep(1)

                    if not generation_complete[0]:
                        print("Gemini API test request timed out")
                        return "Gemini API test timed out after 10 seconds"

                    print(f"Test response from Gemini API: {result[0]}")
                    return f"Gemini API test successful: {result[0]}"
                except Exception as e:
                    print(f"Error testing Gemini API: {e}")
                    return f"Gemini API test failed: {str(e)}"

            elif (
                isinstance(self.generator, dict)
                and self.generator.get("type") == "huggingface"
            ):
                # Use Hugging Face pipeline
                pipeline = self.generator["pipeline"]
                response = pipeline(test_prompt, max_new_tokens=30, do_sample=True)
                full_text = response[0]["generated_text"]

                if len(test_prompt) < len(full_text):
                    result = full_text[len(test_prompt) :].strip()

                    # Check if this is a small model that needs cleaning
                    is_small_model = any(
                        name in self.model_name.lower()
                        for name in ["distilgpt2", "gpt2-small", "tiny"]
                    )
                    if is_small_model:
                        result = self._clean_small_model_response(result)
                else:
                    result = "No additional text generated"

                print(f"Test response from Hugging Face: {result}")
                return f"Hugging Face model test successful: {result}"

            else:
                # Legacy format (for backward compatibility)
                response = self.generator(
                    test_prompt, max_new_tokens=30, do_sample=True
                )
                full_text = response[0]["generated_text"]

                if len(test_prompt) < len(full_text):
                    result = full_text[len(test_prompt) :].strip()

                    # Check if this is a small model that needs cleaning
                    is_small_model = any(
                        name in self.model_name.lower()
                        for name in ["distilgpt2", "gpt2-small", "tiny"]
                    )
                    if is_small_model:
                        result = self._clean_small_model_response(result)
                else:
                    result = "No additional text generated"

                print(f"Test response: {result}")
                return f"Model test successful: {result}"

        except Exception as e:
            print(f"Error testing model: {e}")
            return f"Model test failed: {str(e)}"

    def generate_suggestion(
        self,
        person_context: Dict[str, Any],
        user_input: Optional[str] = None,
        max_length: int = 50,
        temperature: float = 0.7,
    ) -> str:
        """Generate a contextually appropriate suggestion.

        Args:
            person_context: Context information about the person
            user_input: Optional user input to consider
            max_length: Maximum length of the generated suggestion
            temperature: Controls randomness in generation (higher = more random)

        Returns:
            A generated suggestion string
        """
        if not self.model_loaded:
            # Use fallback responses if model isn't loaded
            import random

            print("Model not loaded, using fallback responses")
            return random.choice(self.fallback_responses)

        # Extract context information
        name = person_context.get("name", "")
        role = person_context.get("role", "")
        topics = person_context.get("topics", [])
        context = person_context.get("context", "")
        selected_topic = person_context.get("selected_topic", "")
        common_phrases = person_context.get("common_phrases", [])
        frequency = person_context.get("frequency", "")
        mood = person_context.get("mood", 3)  # Default to neutral mood (3)

        # Get AAC user information
        aac_user = self.aac_user_info

        # Build enhanced prompt
        prompt = f"""I am {aac_user.get('name', 'Will')}, a {aac_user.get('age', 38)}-year-old with MND (Motor Neuron Disease) from {aac_user.get('location', 'Manchester')}.
{aac_user.get('background', '')}

My communication needs: {aac_user.get('communication_needs', '')}

I am talking to {name}, who is my {role}.
About {name}: {context}
We typically talk about: {', '.join(topics)}
We communicate {frequency}.

My current mood: {self._get_mood_description(mood)}
"""

        # Add communication style based on relationship
        if role in ["wife", "son", "daughter", "mother", "father"]:
            prompt += "I communicate with my family in a warm, loving way, sometimes using inside jokes.\n"
        elif role in ["doctor", "therapist", "nurse"]:
            prompt += "I communicate with healthcare providers in a direct, informative way.\n"
        elif role in ["best mate", "friend"]:
            prompt += "I communicate with friends casually, often with humor and sometimes swearing.\n"
        elif role in ["work colleague", "boss"]:
            prompt += (
                "I communicate with colleagues professionally but still friendly.\n"
            )

        # Add topic information if provided
        if selected_topic:
            prompt += f"\nWe are currently discussing {selected_topic}.\n"

            # Add specific context about this topic with this person
            if selected_topic == "football" and "Manchester United" in context:
                prompt += "We both support Manchester United and often discuss recent matches.\n"
            elif selected_topic == "programming" and "software developer" in context:
                prompt += "We both work in software development and share technical interests.\n"
            elif selected_topic == "family plans" and role in ["wife", "husband"]:
                prompt += (
                    "We make family decisions together, considering my condition.\n"
                )
            elif selected_topic == "old scout adventures" and role == "best mate":
                prompt += "We often reminisce about our Scout camping trips in South East London.\n"
            elif selected_topic == "cycling" and "cycling" in context:
                prompt += "I miss being able to cycle but enjoy talking about past cycling adventures.\n"

        # Add the user's message if provided, or set up for conversation initiation
        if user_input:
            # If user input is provided, we're responding to something
            prompt += f'\n{name} just said to me: "{user_input}"\n'
            prompt += f"I want to respond directly to what {name} just said.\n"
        else:
            # No user input means we're initiating a conversation
            if selected_topic:
                # If a topic is selected, initiate conversation about that topic
                prompt += f"\nI'm about to start a conversation with {name} about {selected_topic}.\n"

                # Add specific context about initiating this topic with this person
                if selected_topic == "football" and "Manchester United" in context:
                    prompt += (
                        "We both support Manchester United and often discuss matches.\n"
                    )
                elif selected_topic == "family" and role in [
                    "wife",
                    "husband",
                    "son",
                    "daughter",
                ]:
                    prompt += (
                        "I want to check in about our family plans or activities.\n"
                    )
                elif selected_topic == "health" and role in [
                    "doctor",
                    "nurse",
                    "therapist",
                ]:
                    prompt += "I want to discuss my health condition or symptoms.\n"
                elif selected_topic == "work" and role in ["work colleague", "boss"]:
                    prompt += "I want to discuss a work-related matter.\n"

                prompt += f"I want to initiate a conversation about {selected_topic} in a natural way.\n"
            elif common_phrases:
                # Use context about our typical conversations if no specific topic
                prompt += f"\nI'm about to start a conversation with {name}.\n"
                default_message = common_phrases[0]
                prompt += f'{name} typically says things like: "{default_message}"\n'
                prompt += f"We typically talk about: {', '.join(topics)}\n"
                prompt += "I want to initiate a conversation in a natural way based on our relationship.\n"
            else:
                # Generic conversation starter
                prompt += f"\nI'm about to start a conversation with {name}.\n"
                prompt += "I want to initiate a conversation in a natural way based on our relationship.\n"

        # Add the response prompt with specific guidance
        # Check if this is an instruction-tuned model
        is_instruction_model = any(
            marker in self.model_name.lower()
            for marker in ["-it", "instruct", "chat", "phi-3", "phi-2"]
        )

        # Check if this is a very small model that needs simpler prompts
        is_small_model = any(
            name in self.model_name.lower()
            for name in ["distilgpt2", "gpt2-small", "tiny"]
        )

        if is_small_model:
            # Use a much simpler format for very small models
            if user_input:
                # Responding to something
                prompt += f"""
{name} said: "{user_input}"

Will's response:"""
            else:
                # Initiating a conversation
                if selected_topic:
                    prompt += f"""
Will starts a conversation with {name} about {selected_topic}.

Will says:"""
                else:
                    prompt += f"""
Will starts a conversation with {name}.

Will says:"""
        elif is_instruction_model:
            # Use instruction format for instruction-tuned models
            if user_input:
                # Responding to something
                prompt += f"""
<instruction>
I am Will, the person with MND. I need to respond to {name}'s message: "{user_input}"
My response should be natural, brief (1-2 sentences), and directly relevant to what {name} just said.
I should use language appropriate for our relationship.
I should speak in first person as myself (Will).
</instruction>

My response to {name}:"""
            else:
                # Initiating a conversation
                prompt += f"""
<instruction>
I am Will, the person with MND. I need to start a conversation with {name}.
My conversation starter should be natural, brief (1-2 sentences), and appropriate for our relationship.
If a topic was selected, I should focus on that topic.
I should speak in first person as myself (Will).
</instruction>

My conversation starter to {name}:"""
        else:
            # Use standard format for other models
            if user_input:
                # Responding to something
                prompt += f"""
I am Will, the person with MND. I want to respond to {name}'s message: "{user_input}"
My response should be natural, brief (1-2 sentences), and directly relevant to what {name} just said.
I'll use language appropriate for our relationship and speak as myself (Will).

My response to {name}:"""
            else:
                # Initiating a conversation
                prompt += f"""
I am Will, the person with MND. I want to start a conversation with {name}.
My conversation starter should be natural, brief (1-2 sentences), and appropriate for our relationship.
I'll speak in first person as myself (Will).

My conversation starter to {name}:"""

        # Generate suggestion
        try:
            print(f"Generating suggestion with prompt: {prompt}")

            # Check if we're using the Gemini API or a Hugging Face model
            if (
                isinstance(self.generator, dict)
                and self.generator.get("type") == "gemini-api"
            ):
                try:
                    # Use Gemini API
                    try:
                        genai = self.generator["client"]
                        model_name = self.generator["model"]

                        # Create a generative model
                        model = genai.GenerativeModel(model_name)

                        # Set generation config
                        generation_config = {
                            "temperature": temperature,
                            "top_p": 0.92,
                            "top_k": 50,
                            "max_output_tokens": 100,
                        }

                        # Generate content with timeout

                        result = [
                            "I'm thinking about what to say..."
                        ]  # Default response
                        generation_complete = [False]

                        def generate_with_gemini():
                            try:
                                response = model.generate_content(
                                    prompt, generation_config=generation_config
                                )

                                if response and hasattr(response, "text"):
                                    result[0] = response.text.strip()
                                    print(f"Gemini API response: {result[0]}")
                                else:
                                    print("No response from Gemini API")

                                generation_complete[0] = True
                            except Exception as e:
                                print(f"Error in Gemini generation thread: {e}")
                                generation_complete[0] = True

                        # Start generation in a separate thread
                        generation_thread = threading.Thread(
                            target=generate_with_gemini
                        )
                        generation_thread.daemon = True
                        generation_thread.start()

                        # Wait for up to 10 seconds
                        timeout = 10
                        start_time = time.time()
                        while (
                            not generation_complete[0]
                            and time.time() - start_time < timeout
                        ):
                            time.sleep(0.1)

                        if not generation_complete[0]:
                            print("Gemini API request timed out")
                            return "I'm thinking about what to say... (API timeout)"

                        return result[0]
                    except Exception as e:
                        print(f"Error setting up Gemini API: {e}")
                        return (
                            "I'm having trouble connecting to the Gemini API right now."
                        )

                except Exception as e:
                    print(f"Error generating with Gemini API: {e}")
                    return "Could not generate a suggestion with Gemini API. Please try again."

            elif (
                isinstance(self.generator, dict)
                and self.generator.get("type") == "huggingface"
            ):
                # Use Hugging Face pipeline
                pipeline = self.generator["pipeline"]

                # Generate with Hugging Face
                response = pipeline(
                    prompt,
                    max_new_tokens=100,  # Generate more tokens to ensure we get a response
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                    truncation=False,
                )

                # Extract only the generated part, not the prompt
                full_text = response[0]["generated_text"]
                print(f"Full generated text length: {len(full_text)}")
                print(f"Prompt length: {len(prompt)}")

                # Make sure we're not trying to slice beyond the text length
                if len(prompt) < len(full_text):
                    result = full_text[len(prompt) :].strip()

                    # Post-process the result for small models
                    if is_small_model:
                        result = self._clean_small_model_response(result)

                    print(f"Generated response: {result}")
                    return result
                else:
                    # If the model didn't generate anything beyond the prompt
                    print("Model didn't generate text beyond prompt")
                    return "I'm thinking about what to say..."

            else:
                # Legacy format (for backward compatibility)
                response = self.generator(
                    prompt,
                    max_new_tokens=100,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                    truncation=False,
                )

                # Extract only the generated part, not the prompt
                full_text = response[0]["generated_text"]
                print(f"Full generated text length: {len(full_text)}")
                print(f"Prompt length: {len(prompt)}")

                # Make sure we're not trying to slice beyond the text length
                if len(prompt) < len(full_text):
                    result = full_text[len(prompt) :].strip()

                    # Post-process the result for small models
                    if is_small_model:
                        result = self._clean_small_model_response(result)

                    print(f"Generated response: {result}")
                    return result
                else:
                    # If the model didn't generate anything beyond the prompt
                    print("Model didn't generate text beyond prompt")
                    return "I'm thinking about what to say..."

        except Exception as e:
            print(f"Error generating suggestion: {e}")
            return "Could not generate a suggestion. Please try again."
