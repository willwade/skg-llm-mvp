# AAC Social Graph Assistant for MND

An Augmentative and Alternative Communication (AAC) system that uses a social graph to provide contextually relevant suggestions for users with Motor Neurone Disease (MND).

## About

This demo showcases an AAC system that uses a social graph to provide contextually relevant suggestions for users with MND. The system allows users to select who they are talking to and provides suggestions based on the relationship and common topics of conversation, tailored to the British context with NHS healthcare terminology.

## Features

- **Person-Specific Suggestions**: Select who you're talking to and get suggestions tailored to that relationship
- **Context-Aware**: Uses a social graph to understand relationships and common topics
- **Multiple Suggestion Types**: Get suggestions from a language model, common phrases, or predefined utterance categories
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

1. Select a person from the dropdown menu
2. View their context information
3. Optionally enter current conversation context or record audio
4. If you record audio, click "Transcribe" to convert it to text
5. Choose a suggestion type
6. Click "Generate Suggestions" to get contextually relevant phrases

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
