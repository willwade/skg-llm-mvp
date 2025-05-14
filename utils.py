import json
import random
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

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
            self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            self.embeddings_cache = {}
            self._initialize_embeddings()
        except Exception as e:
            print(f"Warning: Could not load sentence transformer model: {e}")
            self.sentence_model = None
            
    def _load_graph(self) -> Dict[str, Any]:
        """Load the social graph from the JSON file."""
        try:
            with open(self.graph_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading social graph: {e}")
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
                    self.embeddings_cache[utterance] = self.sentence_model.encode(utterance)
    
    def get_people_list(self) -> List[Dict[str, str]]:
        """Get a list of people from the social graph with their names and roles."""
        people = []
        for person_id, person_data in self.graph.get("people", {}).items():
            people.append({
                "id": person_id,
                "name": person_data.get("name", person_id),
                "role": person_data.get("role", "")
            })
        return people
    
    def get_person_context(self, person_id: str) -> Dict[str, Any]:
        """Get context information for a specific person."""
        if person_id not in self.graph.get("people", {}):
            return {}
            
        return self.graph["people"][person_id]
    
    def get_relevant_phrases(self, person_id: str, user_input: Optional[str] = None) -> List[str]:
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
            utterances.extend(random.sample(category_utterances, 
                                           min(2, len(category_utterances))))
                                           
        return utterances

class SuggestionGenerator:
    """Generates contextual suggestions for the AAC system."""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initialize the suggestion generator.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.generator = pipeline("text2text-generation", 
                                     model=self.model, 
                                     tokenizer=self.tokenizer)
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            self.model_loaded = False
    
    def generate_suggestion(self, 
                           person_context: Dict[str, Any], 
                           user_input: Optional[str] = None,
                           max_length: int = 50) -> str:
        """Generate a contextually appropriate suggestion.
        
        Args:
            person_context: Context information about the person
            user_input: Optional user input to consider
            max_length: Maximum length of the generated suggestion
            
        Returns:
            A generated suggestion string
        """
        if not self.model_loaded:
            return "Model not loaded. Please check your installation."
            
        # Extract context information
        name = person_context.get("name", "")
        role = person_context.get("role", "")
        topics = ", ".join(person_context.get("topics", []))
        context = person_context.get("context", "")
        
        # Build prompt
        prompt = f"""Context: {context}
Person: {name} ({role})
Topics of interest: {topics}
"""
        
        if user_input:
            prompt += f"Current conversation: {user_input}\n"
            
        prompt += "Generate an appropriate phrase to say to this person:"
        
        # Generate suggestion
        try:
            response = self.generator(prompt, max_length=max_length)
            return response[0]["generated_text"]
        except Exception as e:
            print(f"Error generating suggestion: {e}")
            return "Could not generate a suggestion. Please try again."
