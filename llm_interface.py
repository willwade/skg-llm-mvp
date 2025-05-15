"""
LLM Interface for the AAC app using Simon Willison's LLM library.
"""

import subprocess
import time
from typing import List, Optional, Dict, Any


class LLMInterface:
    """Interface for Simon Willison's LLM tool."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        max_length: int = 150,
        temperature: float = 0.7,
    ):
        """Initialize the LLM interface.

        Args:
            model_name: Name of the model to use
            max_length: Maximum length of generated text
            temperature: Controls randomness (higher = more random)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.model_loaded = self._check_llm_installed()
        self.fallback_responses = [
            "I'm not sure how to respond to that.",
            "That's interesting. Tell me more.",
            "I'd like to talk about that further.",
            "I appreciate you sharing that with me.",
            "Could we talk about something else?",
            "I need some time to think about that.",
        ]

    def _check_llm_installed(self) -> bool:
        """Check if the LLM tool is installed and working."""
        try:
            result = subprocess.run(
                ["llm", "--version"],
                capture_output=True,
                text=True,
                timeout=5,  # Add a timeout to prevent hanging
            )
            if result.returncode == 0:
                print(f"LLM tool is installed: {result.stdout.strip()}")

                # Also check if the model exists
                try:
                    # Just check if the model is in the list of available models
                    model_check = subprocess.run(
                        ["llm", "models"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

                    if model_check.returncode == 0:
                        if self.model_name in model_check.stdout:
                            print(f"Model {self.model_name} is available")
                            return True
                        else:
                            print(
                                f"Model {self.model_name} not found in available models"
                            )
                            # Try to find similar models
                            if "gemini" in self.model_name.lower():
                                print("Available Gemini models:")
                                for line in model_check.stdout.splitlines():
                                    if "gemini" in line.lower():
                                        print(f"  {line}")
                            return False
                    else:
                        print("Error checking available models")
                        return False

                except Exception as model_error:
                    print(f"Error checking model availability: {model_error}")
                    return False
            else:
                print("LLM tool returned an error.")
                return False
        except subprocess.TimeoutExpired:
            print("Timeout checking LLM tool installation")
            return False
        except Exception as e:
            print(f"Error checking LLM tool: {e}")
            return False

    def _get_max_tokens_param(self) -> str:
        """Get the appropriate max tokens parameter name for the model."""
        if "gemini" in self.model_name.lower():
            return "max_output_tokens"
        else:
            return "max_tokens"

    def generate_suggestion(
        self,
        person_context: Dict[str, Any],
        user_input: Optional[str] = None,
        temperature: Optional[float] = None,
        progress_callback=None,
    ) -> str:
        """Generate a suggestion based on the person context and user input.

        Args:
            person_context: Context information about the person
            user_input: Optional user input to consider
            temperature: Controls randomness in generation (higher = more random)
            progress_callback: Optional callback function to report progress

        Returns:
            A generated suggestion string
        """
        if not self.model_loaded:
            import random

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

        # Get mood description
        mood_descriptions = {
            1: "I'm feeling quite down and sad today. My responses might be more subdued.",
            2: "I'm feeling a bit low today. I might be less enthusiastic than usual.",
            3: "I'm feeling okay today - neither particularly happy nor sad.",
            4: "I'm feeling pretty good today. I'm in a positive mood.",
            5: "I'm feeling really happy and upbeat today! I'm in a great mood.",
        }
        mood_description = mood_descriptions.get(mood, mood_descriptions[3])

        # Build enhanced prompt
        prompt = f"""I am Will, a 38-year-old with MND (Motor Neuron Disease) from Manchester.
I am talking to {name}, who is my {role}.
About {name}: {context}
We typically talk about: {', '.join(topics)}
We communicate {frequency}.

My current mood: {mood_description}
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
                prompt += f"I want to initiate a conversation about {selected_topic} in a natural way.\n"
            else:
                # Generic conversation starter
                prompt += f"\nI'm about to start a conversation with {name}.\n"
                prompt += "I want to initiate a conversation in a natural way based on our relationship.\n"

        # Add the response prompt with specific guidance
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

        # Use the provided temperature or default
        temp = temperature if temperature is not None else self.temperature

        # Update progress if callback provided
        if progress_callback:
            progress_callback(0.3, desc="Sending prompt to LLM...")

        try:
            # Get the appropriate max tokens parameter
            max_tokens_param = self._get_max_tokens_param()

            # Call the LLM tool
            result = subprocess.run(
                [
                    "llm",
                    "-m",
                    self.model_name,
                    "-s",
                    f"temperature={temp}",
                    "-s",
                    f"{max_tokens_param}={self.max_length}",
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=15,  # Add timeout to prevent hanging
            )

            if progress_callback:
                progress_callback(0.7, desc="Processing response...")

            if result.returncode == 0:
                # Get the generated text
                generated = result.stdout.strip()

                # Clean up the response if needed
                if not generated:
                    generated = "I'm not sure what to say about that."

                if progress_callback:
                    progress_callback(0.9, desc="Response generated successfully")

                return generated
            else:
                print(f"Error from LLM tool: {result.stderr}")
                if progress_callback:
                    progress_callback(0.9, desc="Error generating response")
                return "I'm having trouble responding to that right now."
        except subprocess.TimeoutExpired:
            print("LLM generation timed out")
            if progress_callback:
                progress_callback(0.9, desc="Generation timed out")
            return "I need more time to think about that."
        except Exception as e:
            print(f"Error generating with LLM tool: {e}")
            if progress_callback:
                progress_callback(0.9, desc="Error generating response")
            return "I'm having trouble responding to that."

    def generate_multiple_suggestions(
        self,
        person_context: Dict[str, Any],
        user_input: Optional[str] = None,
        num_suggestions: int = 3,
        temperature: Optional[float] = None,
        progress_callback=None,
    ) -> List[str]:
        """Generate multiple suggestions.

        Args:
            person_context: Context information about the person
            user_input: Optional user input to consider
            num_suggestions: Number of suggestions to generate
            temperature: Controls randomness in generation
            progress_callback: Optional callback function to report progress

        Returns:
            A list of generated suggestions
        """
        suggestions = []

        for i in range(num_suggestions):
            if progress_callback:
                progress_callback(
                    0.1 + (i * 0.3),
                    desc=f"Generating suggestion {i+1}/{num_suggestions}",
                )

            # Vary temperature slightly for each suggestion to increase diversity
            temp_variation = 0.05 * (i - 1)  # -0.05, 0, 0.05
            temp = (
                temperature if temperature is not None else self.temperature
            ) + temp_variation

            suggestion = self.generate_suggestion(
                person_context,
                user_input,
                temperature=temp,
                progress_callback=lambda p, desc: (
                    progress_callback(0.1 + (i * 0.3) + (p * 0.3), desc=desc)
                    if progress_callback
                    else None
                ),
            )

            suggestions.append(suggestion)

            # Small delay to ensure UI updates
            time.sleep(0.2)

        return suggestions

    def test_model(self) -> str:
        """Test if the model is working correctly."""
        if not self.model_loaded:
            return "LLM tool not available"

        try:
            # Create a simple test prompt
            test_prompt = "Say hello in one word."

            # Call the LLM tool
            result = subprocess.run(
                [
                    "llm",
                    "-m",
                    self.model_name,
                    "-s",
                    "temperature=0.7",
                    test_prompt,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                response = result.stdout.strip()
                return f"LLM test successful: {response}"
            else:
                return f"LLM test failed: {result.stderr}"
        except Exception as e:
            return f"LLM test error: {str(e)}"
