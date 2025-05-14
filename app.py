import gradio as gr
import whisper
import tempfile
import os
from utils import SocialGraphManager, SuggestionGenerator

# Initialize the social graph manager and suggestion generator
social_graph = SocialGraphManager("social_graph.json")
suggestion_generator = SuggestionGenerator()

# Initialize Whisper model (using the smallest model for speed)
try:
    whisper_model = whisper.load_model("tiny")
    whisper_loaded = True
except Exception as e:
    print(f"Warning: Could not load Whisper model: {e}")
    whisper_loaded = False


def format_person_display(person):
    """Format person information for display in the dropdown."""
    return f"{person['name']} ({person['role']})"


def get_people_choices():
    """Get formatted choices for the people dropdown."""
    people = social_graph.get_people_list()
    return {format_person_display(person): person["id"] for person in people}


def get_suggestion_categories():
    """Get suggestion categories from the social graph."""
    if "common_utterances" in social_graph.graph:
        return list(social_graph.graph["common_utterances"].keys())
    return []


def on_person_change(person_id):
    """Handle person selection change."""
    if not person_id:
        return "", ""

    person_context = social_graph.get_person_context(person_id)
    context_info = (
        f"**{person_context.get('name', '')}** - {person_context.get('role', '')}\n\n"
    )
    context_info += f"**Topics:** {', '.join(person_context.get('topics', []))}\n\n"
    context_info += f"**Frequency:** {person_context.get('frequency', '')}\n\n"
    context_info += f"**Context:** {person_context.get('context', '')}"

    # Get common phrases for this person
    phrases = person_context.get("common_phrases", [])
    phrases_text = "\n\n".join(phrases)

    return context_info, phrases_text


def generate_suggestions(person_id, user_input, suggestion_type):
    """Generate suggestions based on the selected person and user input."""
    if not person_id:
        return "Please select a person first."

    person_context = social_graph.get_person_context(person_id)

    # If suggestion type is "model", use the language model
    if suggestion_type == "model":
        suggestion = suggestion_generator.generate_suggestion(
            person_context, user_input
        )
        return suggestion

    # If suggestion type is "common_phrases", use the person's common phrases
    elif suggestion_type == "common_phrases":
        phrases = social_graph.get_relevant_phrases(person_id, user_input)
        return "\n\n".join(phrases)

    # If suggestion type is a category from common_utterances
    elif suggestion_type in get_suggestion_categories():
        utterances = social_graph.get_common_utterances(suggestion_type)
        return "\n\n".join(utterances)

    # Default fallback
    return "No suggestions available."


def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    if not whisper_loaded:
        return "Whisper model not loaded. Please check your installation."

    try:
        # Transcribe the audio
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return "Could not transcribe audio. Please try again."


# Create the Gradio interface
with gr.Blocks(title="AAC Social Graph Assistant") as demo:
    gr.Markdown("# AAC Social Graph Assistant")
    gr.Markdown(
        "Select who you're talking to, and get contextually relevant suggestions."
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Person selection
            person_dropdown = gr.Dropdown(
                choices=get_people_choices(), label="Who are you talking to?"
            )

            # Context display
            context_display = gr.Markdown(label="Context Information")

            # User input section
            with gr.Row():
                user_input = gr.Textbox(
                    label="Your current conversation (optional)",
                    placeholder="Type or paste current conversation context here...",
                    lines=3,
                )

            # Audio input
            with gr.Row():
                audio_input = gr.Audio(
                    label="Or record your conversation",
                    type="filepath",
                    sources=["microphone"],
                )
                transcribe_btn = gr.Button("Transcribe", variant="secondary")

            # Suggestion type selection
            suggestion_type = gr.Radio(
                choices=["model", "common_phrases"] + get_suggestion_categories(),
                value="model",
                label="Suggestion Type",
            )

            # Generate button
            generate_btn = gr.Button("Generate Suggestions", variant="primary")

        with gr.Column(scale=1):
            # Common phrases
            common_phrases = gr.Textbox(
                label="Common Phrases",
                placeholder="Common phrases will appear here...",
                lines=5,
            )

            # Suggestions output
            suggestions_output = gr.Textbox(
                label="Suggested Phrases",
                placeholder="Suggestions will appear here...",
                lines=8,
            )

    # Set up event handlers
    person_dropdown.change(
        on_person_change,
        inputs=[person_dropdown],
        outputs=[context_display, common_phrases],
    )

    generate_btn.click(
        generate_suggestions,
        inputs=[person_dropdown, user_input, suggestion_type],
        outputs=[suggestions_output],
    )

    speak_btn.click(speak_text, inputs=[suggestions_output], outputs=[speech_output])

# Launch the app
if __name__ == "__main__":
    demo.launch()
