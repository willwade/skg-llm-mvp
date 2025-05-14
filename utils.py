import json
import random
from typing import Dict, List, Any, Optional, Tuple
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
        """Load a Hugging Face model.

        Args:
            model_name: Name of the HuggingFace model to use

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

        try:
            print(f"Loading model: {model_name}")

            # Check if this is a gated model that requires authentication
            is_gated_model = any(
                name in model_name.lower()
                for name in ["gemma", "llama", "mistral", "qwen", "phi"]
            )

            if is_gated_model:
                # Try to get token from environment
                import os

                token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get(
                    "HF_TOKEN"
                )

                if token:
                    print(f"Using token for gated model: {model_name}")
                    from huggingface_hub import login

                    login(token=token, add_to_git_credential=False)

                    # Explicitly pass token to pipeline
                    from transformers import AutoTokenizer, AutoModelForCausalLM

                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_name, token=token
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name, token=token
                        )
                        self.generator = pipeline(
                            "text-generation", model=model, tokenizer=tokenizer
                        )
                    except Exception as e:
                        print(f"Error loading gated model with token: {e}")
                        print(
                            "This may be due to not having accepted the model license or insufficient permissions."
                        )
                        print(
                            "Please visit the model page on Hugging Face Hub and accept the license."
                        )
                        raise
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
                self.generator = pipeline("text-generation", model=model_name)

            # Cache the loaded model
            self.loaded_models[model_name] = self.generator

            self.model_loaded = True
            print(f"Model loaded successfully: {model_name}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            return False

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
            test_prompt = "I am Will. My son Billy asked about football. I respond:"
            print(f"Testing model with prompt: {test_prompt}")
            response = self.generator(test_prompt, max_new_tokens=30, do_sample=True)
            full_text = response[0]["generated_text"]
            if len(test_prompt) < len(full_text):
                result = full_text[len(test_prompt) :]
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

        if is_instruction_model:
            # Use instruction format for instruction-tuned models
            if user_input:
                # Responding to something
                prompt += f"""
<instruction>
Respond to {name} in a way that is natural, brief (1-2 sentences), and directly relevant to what they just said.
Use language appropriate for our relationship.
</instruction>

My response to {name}:"""
            else:
                # Initiating a conversation
                prompt += f"""
<instruction>
Start a conversation with {name} in a natural, brief (1-2 sentences) way.
Use language appropriate for our relationship.
If a topic was selected, focus on that topic.
</instruction>

My conversation starter to {name}:"""
        else:
            # Use standard format for non-instruction models
            if user_input:
                # Responding to something
                prompt += f"""
I want to respond to {name} in a way that is natural, brief (1-2 sentences), and directly relevant to what they just said. I'll use language appropriate for our relationship.

My response to {name}:"""
            else:
                # Initiating a conversation
                prompt += f"""
I want to start a conversation with {name} in a natural, brief (1-2 sentences) way. I'll use language appropriate for our relationship.

My conversation starter to {name}:"""

        # Generate suggestion
        try:
            print(f"Generating suggestion with prompt: {prompt}")
            # Use max_new_tokens instead of max_length to avoid the error
            response = self.generator(
                prompt,
                max_new_tokens=100,  # Generate more tokens to ensure we get a response
                temperature=temperature,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                # Only use truncation if we're providing a max_length
                truncation=False,
            )
            # Extract only the generated part, not the prompt
            full_text = response[0]["generated_text"]
            print(f"Full generated text length: {len(full_text)}")
            print(f"Prompt length: {len(prompt)}")

            # Make sure we're not trying to slice beyond the text length
            if len(prompt) < len(full_text):
                result = full_text[len(prompt) :]
                print(f"Generated response: {result}")
                return result.strip()
            else:
                # If the model didn't generate anything beyond the prompt
                print("Model didn't generate text beyond prompt")
                return "I'm thinking about what to say..."
        except Exception as e:
            print(f"Error generating suggestion: {e}")
            return "Could not generate a suggestion. Please try again."
