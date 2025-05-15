---
title: AAC Social Graph Assistant
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
short_description: AAC system using social graph for contextual suggestions
---

# AAC Social Graph Assistant

An Augmentative and Alternative Communication (AAC) system that uses a social graph to provide contextually relevant suggestions for users with Motor Neurone Disease (MND). This demo is designed to be hosted on Hugging Face Spaces using Gradio.

## Features

- **Person-Specific Suggestions**: Select who you're talking to and get suggestions tailored to that relationship
- **Context-Aware**: Uses a social graph to understand relationships and common topics
- **Multiple Suggestion Types**: Get suggestions from a language model, common phrases, or predefined utterance categories
- **British Context**: Designed with British English and NHS healthcare context in mind
- **MND-Specific**: Tailored for the needs of someone with Motor Neurone Disease
- **Expandable**: Easily improve the system by enhancing the social graph JSON file

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Open your browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860)

## How It Works

1. **Social Graph**: The system uses a JSON-based social graph (`social_graph.json`) that contains information about:
   - Will (the AAC user) - a 38-year-old with MND
   - People in Will's life (family, healthcare providers, friends, colleagues)
   - Relationships, common topics, and phrases

2. **Context Retrieval**: When you select who you are (as someone talking to Will), the system retrieves relevant context information from the social graph.

3. **Conversation Input**: You enter or record what you've said to Will in the conversation.

4. **Suggestion Generation**: Based on who you are and what you've said, the system generates appropriate responses for Will using:
   - A language model (Flan-T5)
   - Common phrases Will might say to you
   - General utterance categories (greetings, needs, emotions, questions)

5. **User Interface**: The Gradio interface provides an intuitive way to simulate conversations with Will and see what an AAC system might suggest for him to say.

## How to Use

1. Select who you (Will) are talking to from the dropdown menu
2. Optionally select a conversation topic
3. View the relationship context information
4. Enter what the other person said to you, or record audio
5. If you record audio, click "Transcribe" to convert it to text
6. Choose how you want to respond (auto-detect, AI-generated, common phrases, etc.)
7. Click "Generate My Responses" to get contextually relevant suggestions

## Customizing the Social Graph

You can customize the system by editing the `social_graph.json` file. The file has the following structure:

```json
{
  "people": {
    "person_id": {
      "name": "Person Name",
      "role": "Relationship",
      "topics": ["Topic 1", "Topic 2"],
      "frequency": "daily/weekly/monthly",
      "common_phrases": ["Phrase 1", "Phrase 2"],
      "context": "Detailed context about the relationship"
    }
  },
  "places": ["Place 1", "Place 2"],
  "topics": ["Topic 1", "Topic 2"],
  "common_utterances": {
    "category1": ["Utterance 1", "Utterance 2"],
    "category2": ["Utterance 3", "Utterance 4"]
  }
}
```

## Plan

Look at using a Structured Knowledge Format (SKF) – a compact, machine-optimized format designed for efficient AI parsing rather than human readability.

We should create a SKF<->JSON converter to convert the social graph JSON into a more compact format. This will help in reducing the size of the social graph and make it easier for AI models to parse.

See also https://github.com/marv1nnnnn/llm-min.txt

## Current Social Graph Context

The current social graph represents a British person with MND who:

- Lives in Manchester with their wife Emma and two children (Mabel, 4 and Billy, 7)
- Was diagnosed with MND 5 months ago
- Works as a computer programmer
- Has friends from Scout days growing up in South East London
- Enjoys cycling and hiking in the Peak District and Lake District
- Has a healthcare team including a neurologist, MND nurse specialist, physiotherapist, and speech therapist
- Is supported by work colleagues with flexible arrangements
- Has family in South East London

## Deployment to Hugging Face Spaces

To deploy this application to Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Select Gradio as the SDK
3. Upload the files from this repository
4. The Space will automatically build and deploy the application

## Future Improvements

- Add a visual representation of the social graph
- Support for multiple users with different social graphs
- Add emotion/sentiment detection for more contextually appropriate suggestions
- Implement text-to-speech for output
- Improve speech recognition with a larger Whisper model

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [Gradio](https://www.gradio.app/)
- Uses [Hugging Face Transformers](https://huggingface.co/transformers/) for language models
- Uses [Sentence Transformers](https://www.sbert.net/) for semantic matching
