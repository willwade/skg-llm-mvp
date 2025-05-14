import gradio as gr
import whisper
import tempfile
import os
from utils import SocialGraphManager, SuggestionGenerator

# Define available models
AVAILABLE_MODELS = {
    "distilgpt2": "DistilGPT2 (Fast, smaller model)",
    "gpt2": "GPT-2 (Medium size, better quality)",
    "google/gemma-3-1b-it": "Gemma 3 1B-IT (Small, instruction-tuned)",
    "Qwen/Qwen1.5-0.5B": "Qwen 1.5 0.5B (Very small, efficient)",
    "Qwen/Qwen1.5-1.8B": "Qwen 1.5 1.8B (Small, good quality)",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama 1.1B (Small, chat-tuned)",
    "microsoft/phi-3-mini-4k-instruct": "Phi-3 Mini (Small, instruction-tuned)",
    "microsoft/phi-2": "Phi-2 (Small, high quality for size)",
}

# Initialize the social graph manager
social_graph = SocialGraphManager("social_graph.json")

# Initialize the suggestion generator with distilgpt2 (default)
suggestion_generator = SuggestionGenerator("distilgpt2")

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
    """Get suggestion categories from the social graph."""
    if "common_utterances" in social_graph.graph:
        return list(social_graph.graph["common_utterances"].keys())
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

    # Try to load the new model
    success = suggestion_generator.load_model(model_name)

    if success:
        progress(1.0, desc=f"Model loaded: {model_name}")
        return f"Successfully switched to model: {model_name}"
    else:
        progress(1.0, desc="Model loading failed")
        return f"Failed to load model: {model_name}. Using fallback responses instead."


def generate_suggestions(
    person_id,
    user_input,
    suggestion_type,
    selected_topic=None,
    model_name="distilgpt2",
    temperature=0.7,
    progress=gr.Progress(),
):
    """Generate suggestions based on the selected person and user input."""
    print(
        f"Generating suggestions with: person_id={person_id}, user_input={user_input}, "
        f"suggestion_type={suggestion_type}, selected_topic={selected_topic}, "
        f"model={model_name}, temperature={temperature}"
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

    # Try to infer conversation type if user input is provided
    inferred_category = None
    if user_input and suggestion_type == "auto_detect":
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

    # Format the output with multiple suggestions
    result = ""

    # If suggestion type is "model", use the language model for multiple suggestions
    if suggestion_type == "model":
        print("Using model for suggestions")
        progress(0.2, desc="Preparing to generate suggestions...")

        # Generate 3 different suggestions
        suggestions = []
        for i in range(3):
            progress_value = 0.3 + (i * 0.2)  # Progress from 30% to 70%
            progress(progress_value, desc=f"Generating suggestion {i+1}/3")
            print(f"Generating suggestion {i+1}/3")
            try:
                suggestion = suggestion_generator.generate_suggestion(
                    person_context, user_input, temperature=temperature
                )
                print(f"Generated suggestion: {suggestion}")
                suggestions.append(suggestion)
            except Exception as e:
                print(f"Error generating suggestion: {e}")
                suggestions.append("Error generating suggestion")

        result = (
            f"### AI-Generated Responses (using {suggestion_generator.model_name}):\n\n"
        )
        for i, suggestion in enumerate(suggestions, 1):
            result += f"{i}. {suggestion}\n\n"

        print(f"Final result: {result[:100]}...")

    # If suggestion type is "common_phrases", use the person's common phrases
    elif suggestion_type == "common_phrases":
        phrases = social_graph.get_relevant_phrases(person_id, user_input)
        result = "### My Common Phrases with this Person:\n\n"
        for i, phrase in enumerate(phrases, 1):
            result += f"{i}. {phrase}\n\n"

    # If suggestion type is "auto_detect", use the inferred category or default to model
    elif suggestion_type == "auto_detect":
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
                for i in range(3):
                    progress_value = 0.4 + (i * 0.15)  # Progress from 40% to 70%
                    progress(
                        progress_value, desc=f"Generating fallback suggestion {i+1}/3"
                    )
                    suggestion = suggestion_generator.generate_suggestion(
                        person_context, user_input, temperature=temperature
                    )
                    suggestions.append(suggestion)

                result = f"### AI-Generated Responses (no category detected, using {suggestion_generator.model_name}):\n\n"
                for i, suggestion in enumerate(suggestions, 1):
                    result += f"{i}. {suggestion}\n\n"
            except Exception as e:
                print(f"Error generating fallback suggestion: {e}")
                result = "### Could not generate a response:\n\n"
                result += "1. Sorry, I couldn't generate a suggestion at this time.\n\n"

    # If suggestion type is a category from common_utterances
    elif suggestion_type in get_suggestion_categories():
        print(f"Using category: {suggestion_type}")
        utterances = social_graph.get_common_utterances(suggestion_type)
        print(f"Got utterances: {utterances}")
        result = f"### {suggestion_type.replace('_', ' ').title()} Phrases:\n\n"
        for i, utterance in enumerate(utterances, 1):
            result += f"{i}. {utterance}\n\n"

    # Default fallback
    else:
        print(f"No handler for suggestion type: {suggestion_type}")
        result = "No suggestions available. Please try a different option."

    print(f"Returning result: {result[:100]}...")

    # Complete the progress
    progress(1.0, desc="Completed!")

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
with gr.Blocks(title="Will's AAC Communication Aid") as demo:
    gr.Markdown("# Will's AAC Communication Aid")
    gr.Markdown(
        """
    This demo simulates an AAC system from Will's perspective (a 38-year-old with MND).

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

            # Audio input
            with gr.Row():
                audio_input = gr.Audio(
                    label="Or record what they said:",
                    type="filepath",
                    sources=["microphone"],
                )
                transcribe_btn = gr.Button("Transcribe", variant="secondary")

            # Suggestion type selection
            suggestion_type = gr.Radio(
                choices=[
                    "model",
                    "auto_detect",
                    "common_phrases",
                ]
                + get_suggestion_categories(),
                value="model",  # Default to model for better results
                label="How should I respond?",
                info="Choose response type (model = AI-generated, auto_detect = automatic category detection)",
            )

            # Model selection
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="distilgpt2",
                    label="Language Model",
                    info="Select which AI model to use for generating responses",
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
        ],
        outputs=[suggestions_output],
    )

    # Transcribe audio to text
    transcribe_btn.click(
        transcribe_audio,
        inputs=[audio_input],
        outputs=[user_input],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
