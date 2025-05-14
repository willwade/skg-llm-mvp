# Will's AAC Communication Aid

An Augmentative and Alternative Communication (AAC) system that uses a social graph to provide contextually relevant suggestions for Will, a user with Motor Neurone Disease (MND).

## About

This demo simulates an AAC system from Will's perspective (a 38-year-old with MND). The system allows Will to select who he's talking to, optionally choose a conversation topic, and get appropriate responses based on what the other person has said. All suggestions are tailored to the relationship and conversation context, using British English and NHS healthcare terminology where appropriate.

## Features

- **Person-Specific Suggestions**: Select who you (Will) are talking to and get tailored responses
- **Topic Selection**: Choose conversation topics relevant to your relationship
- **Context-Aware**: Uses a social graph to understand relationships and common topics
- **Speech Recognition**: Record what others have said to you and have it transcribed
- **Auto-Detection**: Automatically detect conversation type from what others say
- **Multiple Response Types**: Get AI-generated responses, common phrases, or category-specific utterances
- **British Context**: Designed with British English and NHS healthcare context in mind
- **MND-Specific**: Tailored for the needs of someone with Motor Neurone Disease
- **Expandable**: Easily improve the system by enhancing the social graph JSON file

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
