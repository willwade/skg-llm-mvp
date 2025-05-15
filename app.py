import gradio as gr
import whisper
import random
import time
from utils import SocialGraphManager
from llm_interface import LLMInterface

# Define available models
AVAILABLE_MODELS = {
    # Gemini models (online API)
    "gemini-1.5-flash-latest": "🌐 Gemini 1.5 Flash (Online API - Fast, Recommended)",
    "gemini-1.5-pro-latest": "🌐 Gemini 1.5 Pro (Online API - High quality)",
    # OpenAI models (if API key is set)
    "gpt-3.5-turbo": "🌐 ChatGPT 3.5 (Online API)",
    "gpt-4o-mini": "🌐 GPT-4o Mini (Online API - Fast)",
    # Ollama models (if installed locally)
    "ollama/gemma:7b": "💻 Gemma 7B (Offline - requires Ollama)",
    "ollama/llama3:8b": "💻 Llama 3 8B (Offline - requires Ollama)",
}

# Initialize the social graph manager
social_graph = SocialGraphManager("social_graph.json")

# Initialize the suggestion generator with a fast online model by default
print("Initializing with Gemini 1.5 Flash (online model)")
suggestion_generator = LLMInterface("gemini-1.5-flash-latest")

# Test the model to make sure it's working
print("Testing model connection...")
test_result = suggestion_generator.test_model()
print(f"Model test result: {test_result}")

# If the model didn't load, try Ollama as fallback
if not suggestion_generator.model_loaded:
    print("Online model not available, trying Ollama model...")
    suggestion_generator = LLMInterface("ollama/gemma:7b")
    test_result = suggestion_generator.test_model()
    print(f"Ollama model test result: {test_result}")

    # If Ollama also fails, try OpenAI as fallback
    if not suggestion_generator.model_loaded:
        print("Ollama not available, trying OpenAI model...")
        suggestion_generator = LLMInterface("gpt-3.5-turbo")
        test_result = suggestion_generator.test_model()
        print(f"OpenAI model test result: {test_result}")

# Test the model to make sure it's working
test_result = suggestion_generator.test_model()
print(f"Model test result: {test_result}")

# If the model didn't load, use the fallback responses
if not suggestion_generator.model_loaded:
    print("Model failed to load, using fallback responses...")
    # The SuggestionGenerator class has built-in fallback responses

# Initialize Whisper model (using the smallest model for speed)
try:
    whisper_model = whisper.load_model("tiny")
    whisper_loaded = True
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    whisper_loaded = False


def format_person_display(person):
    """Format person information for display in the dropdown."""
    return f"{person['name']} ({person['role']})"


def get_people_choices():
    """Get formatted choices for the people dropdown."""
    people = social_graph.get_people_list()
    choices = {}
    for person in people:
        display_name = format_person_display(person)
        person_id = person["id"]
        choices[display_name] = person_id
    return choices


def get_topics_for_person(person_id):
    """Get topics for a specific person."""
    if not person_id:
        return []

    person_context = social_graph.get_person_context(person_id)
    topics = person_context.get("topics", [])
    return topics


def get_suggestion_categories():
    """Get suggestion categories from the social graph with emoji prefixes."""
    if "common_utterances" in social_graph.graph:
        categories = list(social_graph.graph["common_utterances"].keys())
        emoji_map = {
            "greetings": "👋 greetings",
            "needs": "🆘 needs",
            "emotions": "😊 emotions",
            "questions": "❓ questions",
            "tech_talk": "💻 tech_talk",
            "reminiscing": "🔙 reminiscing",
            "organization": "📅 organization",
        }
        return [emoji_map.get(cat, cat) for cat in categories]
    return []


def on_person_change(person_id):
    """Handle person selection change."""
    if not person_id:
        return "", "", []

    person_context = social_graph.get_person_context(person_id)

    # Create a more user-friendly context display
    name = person_context.get("name", "")
    role = person_context.get("role", "")
    frequency = person_context.get("frequency", "")
    context_text = person_context.get("context", "")

    context_info = f"""### I'm talking to: {name}
**Relationship:** {role}
**How often we talk:** {frequency}

**Our relationship:** {context_text}
"""

    # Get common phrases for this person
    phrases = person_context.get("common_phrases", [])
    phrases_text = "\n\n".join(phrases)

    # Get topics for this person
    topics = person_context.get("topics", [])

    return context_info, phrases_text, topics


def change_model(model_name, progress=gr.Progress()):
    """Change the language model used for generation.

    Args:
        model_name: The name of the model to use
        progress: Gradio progress indicator

    Returns:
        A status message about the model change
    """
    global suggestion_generator

    print(f"Changing model to: {model_name}")

    # Check if we need to change the model
    if model_name == suggestion_generator.model_name:
        return f"Already using model: {model_name}"

    # Show progress indicator
    progress(0, desc=f"Loading model: {model_name}")

    # Create a new LLMInterface with the selected model
    try:
        progress(0.3, desc=f"Initializing {model_name}...")
        new_generator = LLMInterface(model_name)

        # Test if the model works
        progress(0.6, desc="Testing model connection...")
        test_result = new_generator.test_model()
        print(f"Model test result: {test_result}")

        if new_generator.model_loaded:
            # Replace the current generator with the new one
            suggestion_generator = new_generator
            progress(1.0, desc=f"Model loaded: {model_name}")
            return f"Successfully switched to model: {model_name}"
        else:
            progress(1.0, desc="Model loading failed")
            return f"Failed to load model: {model_name}. Using previous model instead."
    except Exception as e:
        print(f"Error changing model: {e}")
        progress(1.0, desc="Error loading model")
        return f"Error loading model: {model_name}. Using previous model instead."


def generate_suggestions(
    person_id,
    user_input,
    suggestion_type,
    selected_topic=None,
    model_name="gemini-1.5-flash",
    temperature=0.7,
    mood=3,
    progress=gr.Progress(),
):
    """Generate suggestions based on the selected person and user input."""
    print(
        f"Generating suggestions with: person_id={person_id}, user_input={user_input}, "
        f"suggestion_type={suggestion_type}, selected_topic={selected_topic}, "
        f"model={model_name}, temperature={temperature}, mood={mood}"
    )

    # Initialize progress
    progress(0, desc="Starting...")

    if not person_id:
        print("No person_id provided")
        return "Please select who you're talking to first."

    # Make sure we're using the right model
    if model_name != suggestion_generator.model_name:
        progress(0.1, desc=f"Switching to model: {model_name}")
        change_model(model_name, progress)

    person_context = social_graph.get_person_context(person_id)
    print(f"Person context: {person_context}")

    # Remove emoji prefix from suggestion_type if present
    clean_suggestion_type = suggestion_type
    if suggestion_type.startswith(
        ("🤖", "🔍", "💬", "👋", "🆘", "😊", "❓", "💻", "🔙", "📅")
    ):
        clean_suggestion_type = suggestion_type[2:].strip()  # Remove emoji and space

    # Try to infer conversation type if user input is provided
    inferred_category = None
    if user_input and clean_suggestion_type == "auto_detect":
        # Simple keyword matching for now - could be enhanced with ML
        user_input_lower = user_input.lower()
        if any(
            word in user_input_lower
            for word in ["hi", "hello", "morning", "afternoon", "evening"]
        ):
            inferred_category = "greetings"
        elif any(
            word in user_input_lower
            for word in ["feel", "tired", "happy", "sad", "frustrated"]
        ):
            inferred_category = "emotions"
        elif any(
            word in user_input_lower
            for word in ["need", "want", "help", "water", "toilet", "loo"]
        ):
            inferred_category = "needs"
        elif any(
            word in user_input_lower
            for word in ["what", "how", "when", "where", "why", "did"]
        ):
            inferred_category = "questions"
        elif any(
            word in user_input_lower
            for word in ["remember", "used to", "back then", "when we"]
        ):
            inferred_category = "reminiscing"
        elif any(
            word in user_input_lower
            for word in ["code", "program", "software", "app", "tech"]
        ):
            inferred_category = "tech_talk"
        elif any(
            word in user_input_lower
            for word in ["plan", "schedule", "appointment", "tomorrow", "later"]
        ):
            inferred_category = "organization"

    # Add topic to context if selected
    if selected_topic:
        person_context["selected_topic"] = selected_topic

    # Add mood to person context
    person_context["mood"] = mood

    # Format the output with multiple suggestions
    result = ""

    # If suggestion type is "model", use the language model for multiple suggestions
    if clean_suggestion_type == "model":
        print("Using model for suggestions")
        progress(0.2, desc="Preparing to generate suggestions...")

        # Generate suggestions using the LLM interface
        try:
            # Use the LLM interface to generate multiple suggestions
            suggestions = suggestion_generator.generate_multiple_suggestions(
                person_context=person_context,
                user_input=user_input,
                num_suggestions=3,
                temperature=temperature,
                progress_callback=lambda p, desc: progress(0.2 + (p * 0.7), desc=desc),
            )

            # Make sure we have at least one suggestion
            if not suggestions:
                suggestions = ["I'm not sure what to say about that."]

            # Make sure we have exactly 3 suggestions (pad with fallbacks if needed)
            while len(suggestions) < 3:
                suggestions.append("I'm not sure what else to say about that.")

            result = f"### AI-Generated Responses (using {suggestion_generator.model_name}):\n\n"
            for i, suggestion in enumerate(suggestions, 1):
                result += f"{i}. {suggestion}\n\n"

            print(f"Final result: {result[:100]}...")

        except Exception as e:
            print(f"Error generating suggestions: {e}")
            result = "### Error generating suggestions:\n\n"
            result += "1. I'm having trouble generating responses right now.\n\n"
            result += "2. Please try again or select a different model.\n\n"
            result += "3. You might want to check your internet connection if using an online model.\n\n"

        # Force a complete progress update before returning
        progress(0.9, desc="Finalizing suggestions...")

    # If suggestion type is "common_phrases", use the person's common phrases
    elif clean_suggestion_type == "common_phrases":
        phrases = social_graph.get_relevant_phrases(person_id, user_input)
        result = "### My Common Phrases with this Person:\n\n"
        for i, phrase in enumerate(phrases, 1):
            result += f"{i}. {phrase}\n\n"

    # If suggestion type is "auto_detect", use the inferred category or default to model
    elif clean_suggestion_type == "auto_detect":
        print(f"Auto-detect mode, inferred category: {inferred_category}")
        if inferred_category:
            utterances = social_graph.get_common_utterances(inferred_category)
            print(f"Got utterances for category {inferred_category}: {utterances}")
            result = f"### Auto-detected category: {inferred_category.replace('_', ' ').title()}\n\n"
            for i, utterance in enumerate(utterances, 1):
                result += f"{i}. {utterance}\n\n"
        else:
            print("No category inferred, falling back to model")
            # Fall back to model if we couldn't infer a category
            progress(0.3, desc="No category detected, using model instead...")
            try:
                suggestions = []
                # Set a timeout for each suggestion generation (10 seconds)
                timeout_per_suggestion = 10

                for i in range(3):
                    progress_value = 0.4 + (i * 0.15)  # Progress from 40% to 70%
                    progress(
                        progress_value, desc=f"Generating fallback suggestion {i+1}/3"
                    )
                    try:
                        # Add mood to person context
                        person_context["mood"] = mood

                        # Set a start time for timeout tracking
                        start_time = time.time()

                        # Try to generate a suggestion with timeout
                        suggestion = None

                        # If model isn't loaded, use fallback immediately
                        if not suggestion_generator.model_loaded:
                            print("Model not loaded, using fallback response")
                            suggestion = random.choice(
                                suggestion_generator.fallback_responses
                            )
                        else:
                            # Try to generate with the model
                            suggestion = suggestion_generator.generate_suggestion(
                                person_context, user_input, temperature=temperature
                            )

                        # Check if generation took too long
                        if time.time() - start_time > timeout_per_suggestion:
                            print(
                                f"Fallback suggestion {i+1} generation timed out, using fallback"
                            )
                            suggestion = (
                                "I'm not sure what to say about that right now."
                            )

                        # Only add non-empty suggestions
                        if suggestion and suggestion.strip():
                            suggestions.append(suggestion.strip())
                        else:
                            print("Empty fallback suggestion received, using default")
                            suggestions.append("I'm not sure what to say about that.")

                        # Force a progress update after each suggestion
                        progress(
                            0.4 + (i * 0.15) + 0.05,
                            desc=f"Completed fallback suggestion {i+1}/3",
                        )

                    except Exception as e:
                        print(f"Error generating fallback suggestion {i+1}: {e}")
                        suggestions.append("I'm having trouble responding to that.")
                        # Force a progress update even after error
                        progress(
                            0.4 + (i * 0.15) + 0.05,
                            desc=f"Error in fallback suggestion {i+1}/3",
                        )

                    # Small delay to ensure UI updates
                    time.sleep(0.2)

                # Make sure we have at least one suggestion
                if not suggestions:
                    suggestions = ["I'm not sure what to say about that."]

                # Make sure we have exactly 3 suggestions (pad with fallbacks if needed)
                while len(suggestions) < 3:
                    suggestions.append("I'm not sure what else to say about that.")

                # Force a progress update
                progress(0.85, desc="Finalizing fallback suggestions...")

                result = "### AI-Generated Responses (no category detected):\n\n"
                for i, suggestion in enumerate(suggestions, 1):
                    result += f"{i}. {suggestion}\n\n"
            except Exception as e:
                print(f"Error generating fallback suggestion: {e}")
                progress(0.9, desc="Error handling...")
                result = "### Could not generate a response:\n\n"
                result += "1. Sorry, I couldn't generate a suggestion at this time.\n\n"

    # If suggestion type is a category from common_utterances
    elif clean_suggestion_type in [
        "greetings",
        "needs",
        "emotions",
        "questions",
        "tech_talk",
        "reminiscing",
        "organization",
    ]:
        print(f"Using category: {clean_suggestion_type}")
        utterances = social_graph.get_common_utterances(clean_suggestion_type)
        print(f"Got utterances: {utterances}")
        result = f"### {clean_suggestion_type.replace('_', ' ').title()} Phrases:\n\n"
        for i, utterance in enumerate(utterances, 1):
            result += f"{i}. {utterance}\n\n"

    # Default fallback
    else:
        print(f"No handler for suggestion type: {clean_suggestion_type}")
        result = "No suggestions available. Please try a different option."

    print(f"Returning result: {result[:100]}...")
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result)}")

    # Make sure we're returning a non-empty string
    if not result or len(result.strip()) == 0:
        result = "No response was generated. Please try again with different settings."

    # Always complete the progress to 100% before returning
    progress(1.0, desc="Completed!")

    # Add a small delay to ensure UI updates properly
    time.sleep(0.5)

    # Print final status
    print("Generation completed successfully, returning result")

    return result


def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    if not whisper_loaded:
        return "Whisper model not loaded. Please check your installation."

    try:
        # Transcribe the audio
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception:
        return "Could not transcribe audio. Please try again."


# Create the Gradio interface
with gr.Blocks(title="Will's AAC Communication Aid", css="custom.css") as demo:
    gr.Markdown("# Will's AAC Communication Aid")
    gr.Markdown(
        """
    This demo simulates an AAC system from Will's perspective (a 38-year-old with MND). Its based on a social graph of people in Will's life and their common phrases. The idea is that this graph is generated on device securely. You can see this [here](https://github.com/willwade/skg-llm-mvp/blob/main/social_graph.json)

    **How to use this demo:**
    1. Select who you (Will) are talking to from the dropdown
    2. Optionally select a conversation topic
    3. Enter or record what the other person said to you
    4. Get suggested responses based on your relationship with that person
    """
    )

    # Display information about Will
    with gr.Accordion("About Me (Will)", open=False):
        gr.Markdown(
            """
        I'm Will, a 38-year-old computer programmer from Manchester with MND (diagnosed 5 months ago).
        I live with my wife Emma and two children (Mabel, 4 and Billy, 7).
        Originally from South East London, I enjoy technology, Manchester United, and have fond memories of cycling and hiking.
        I'm increasingly using this AAC system as my speech becomes more difficult.
        """
        )

    with gr.Row():
        with gr.Column(scale=1):
            # Person selection
            person_dropdown = gr.Dropdown(
                choices=get_people_choices(),
                label="I'm talking to:",
                info="Select who you (Will) are talking to",
            )

            # Get topics for the selected person
            def get_filtered_topics(person_id):
                if not person_id:
                    return []
                person_context = social_graph.get_person_context(person_id)
                return person_context.get("topics", [])

            # Topic selection dropdown
            topic_dropdown = gr.Dropdown(
                choices=[],  # Will be populated when a person is selected
                label="Topic (optional):",
                info="Select a topic to discuss or respond about",
                allow_custom_value=True,
            )

            # Context display
            context_display = gr.Markdown(label="Relationship Context")

            # User input section
            with gr.Row():
                user_input = gr.Textbox(
                    label="What they said to me: (leave empty to start a conversation)",
                    placeholder='Examples:\n"How was your physio session today?"\n"The kids are asking if you want to watch a movie tonight"\n"I\'ve been looking at that new AAC software you mentioned"',
                    lines=3,
                )

            # Audio input with auto-transcription
            with gr.Column(elem_classes="audio-recorder-container"):
                gr.Markdown("### 🎤 Or record what they said")
                audio_input = gr.Audio(
                    label="",
                    type="filepath",
                    sources=["microphone"],
                    elem_classes="audio-recorder",
                )
                gr.Markdown(
                    "*Recording will auto-transcribe when stopped*",
                    elem_classes="auto-transcribe-hint",
                )

            # Suggestion type selection with emojis
            suggestion_type = gr.Radio(
                choices=[
                    "🤖 model",
                    "🔍 auto_detect",
                    "💬 common_phrases",
                ]
                + get_suggestion_categories(),
                value="🤖 model",  # Default to model for better results
                label="How should I respond?",
                info="Choose response type",
                elem_classes="emoji-response-options",
            )

            # Add a mood slider with emoji indicators at the ends
            with gr.Column(elem_classes="mood-slider-container"):
                mood_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="How am I feeling today?",
                    info="This will influence the tone of your responses (😢 Sad → Happy 😄)",
                    elem_classes="mood-slider",
                )

            # Model selection
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="gemini-1.5-flash-latest",
                    label="Language Model",
                    info="Select which AI model to use (🌐 = online API, 💻 = offline model)",
                )

                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Controls randomness (higher = more creative, lower = more focused)",
                )

            # Generate button
            generate_btn = gr.Button(
                "Generate My Responses/Conversation Starters", variant="primary"
            )

            # Model status
            model_status = gr.Markdown(
                value=f"Current model: {suggestion_generator.model_name}",
                label="Model Status",
            )

        with gr.Column(scale=1):
            # Common phrases
            common_phrases = gr.Textbox(
                label="My Common Phrases",
                placeholder="Common phrases I often use with this person will appear here...",
                lines=5,
            )

            # Suggestions output
            suggestions_output = gr.Markdown(
                label="My Suggested Responses",
                value="Suggested responses will appear here...",
                elem_id="suggestions_output",  # Add an ID for easier debugging
            )

    # Set up event handlers
    def handle_person_change(person_id):
        """Handle person selection change and update UI elements."""
        context_info, phrases_text, _ = on_person_change(person_id)

        # Get topics for this person
        topics = get_filtered_topics(person_id)

        # Update the context, phrases, and topic dropdown
        return context_info, phrases_text, gr.update(choices=topics)

    def handle_model_change(model_name):
        """Handle model selection change."""
        status = change_model(model_name)
        return status

    # Set up the person change event
    person_dropdown.change(
        handle_person_change,
        inputs=[person_dropdown],
        outputs=[context_display, common_phrases, topic_dropdown],
    )

    # Set up the model change event
    model_dropdown.change(
        handle_model_change,
        inputs=[model_dropdown],
        outputs=[model_status],
    )

    # Set up the generate button click event
    generate_btn.click(
        generate_suggestions,
        inputs=[
            person_dropdown,
            user_input,
            suggestion_type,
            topic_dropdown,
            model_dropdown,
            temperature_slider,
            mood_slider,
        ],
        outputs=[suggestions_output],
    )

    # Auto-transcribe audio to text when recording stops
    audio_input.stop_recording(
        transcribe_audio,
        inputs=[audio_input],
        outputs=[user_input],
    )

# Launch the app
if __name__ == "__main__":
    print("Starting application...")
    try:
        demo.launch()
    except Exception as e:
        print(f"Error launching application: {e}")
