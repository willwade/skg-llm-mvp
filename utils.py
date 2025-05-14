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

        try:
            print(f"Loading model: {model_name}")
            # Use a simpler approach with a pre-built pipeline
            self.generator = pipeline("text-generation", model=model_name)
            self.model_loaded = True
            print(f"Model loaded successfully: {model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False

        # Fallback responses if model fails to load or generate
        self.fallback_responses = [
            "I'm not sure how to respond to that.",
            "That's interesting. Tell me more.",
            "I'd like to talk about that further.",
            "I appreciate you sharing that with me.",
        ]

    def test_model(self) -> str:
        """Test if the model is working correctly."""
        if not self.model_loaded:
            return "Model not loaded"

        try:
            test_prompt = "I am Will. My son Billy asked about football. I respond:"
            print(f"Testing model with prompt: {test_prompt}")
            response = self.generator(test_prompt, max_length=30, do_sample=True)
            result = response[0]["generated_text"][len(test_prompt) :]
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
        topics = ", ".join(person_context.get("topics", []))
        context = person_context.get("context", "")
        selected_topic = person_context.get("selected_topic", "")

        # Build prompt
        prompt = f"""I am Will, a person with MND (Motor Neuron Disease).
I'm talking to {name}, who is my {role}.
"""

        if context:
            prompt += f"Context: {context}\n"

        if topics:
            prompt += f"Topics of interest: {topics}\n"

        if selected_topic:
            prompt += f"We're currently talking about: {selected_topic}\n"

        if user_input:
            prompt += f'\n{name} just said to me: "{user_input}"\n'

        prompt += "\nMy response:"

        # Generate suggestion
        try:
            print(f"Generating suggestion with prompt: {prompt}")
            response = self.generator(
                prompt,
                max_length=len(prompt.split()) + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.92,
                top_k=50,
            )
            # Extract only the generated part, not the prompt
            result = response[0]["generated_text"][len(prompt) :]
            print(f"Generated response: {result}")
            return result.strip()
        except Exception as e:
            print(f"Error generating suggestion: {e}")
            return "Could not generate a suggestion. Please try again."
