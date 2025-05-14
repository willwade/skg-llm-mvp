# AAC Context-Aware Demo: To-Do Document

## Goal

Create a proof-of-concept offline-capable RAG (Retrieval-Augmented Generation) system for ALS AAC users that:

* Uses a lightweight knowledge graph (JSON)
* Supports utterance suggestion and correction
* Uses local/offline LLMs (e.g., Gemma, Flan-T5)
* Includes a semantic retriever to match context (e.g. conversation partner, topics)
* Provides a Gradio-based UI for deployment on HuggingFace

---

## Phase 1: Environment Setup

* [ ] Install Gradio, Transformers, Sentence-Transformers
* [ ] Choose and install inference backends:

  * [ ] `google/flan-t5-base` (via HuggingFace Transformers)
  * [ ] Gemma 2B via Ollama or Transformers (check support for offline use)
  * [ ] Sentence similarity model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` or similar)

---

## Phase 2: Knowledge Graph

* [ ] Create example `social_graph.json` (people, topics, relationships)
* [ ] Define function to extract relevant context given a selected person

  * Name, relationship, typical topics, frequency
* [ ] Format for prompt injection: inline context for LLM use

---

## Phase 3: Semantic Retriever

* [ ] Load sentence-transformer model
* [ ] Create index from the social graph topics/descriptions
* [ ] Match transcript to closest node(s) in the graph
* [ ] Retrieve context for prompt generation

---

## Phase 4: Gradio UI

* [ ] Simple interface:

  * Dropdown: Select "Who is speaking?" (Bob, Alice, etc.)
  * Record Button: Capture audio input
  * Text area: Show transcript
  * Toggle tabs:

    * [ ] "Suggest Utterance"
    * [ ] "Correct Message"
  * Output: Generated message
* [ ] Implement Whisper transcription (use `whisper`, `faster-whisper`, or `whisper.cpp`)
* [ ] Pass transcript + retrieved context to LLM model

---

## Phase 5: Model Comparison

* [ ] Test both Flan-T5 and Gemma:

  * [ ] Evaluate speed/quality tradeoffs
  * [ ] Compare correction accuracy and context-specific generation

---

## Optional Phase 6: HuggingFace Deployment

* [ ] Clean up UI and remove dependencies requiring GPU-only execution
* [ ] Upload Gradio demo to HuggingFace Spaces
* [ ] Add documentation and example graphs/transcripts

---

## Notes

* Keep user privacy and safety in mind (no cloud transcription if Whisper offline is available)
* Keep JSON editable for later expansion (add sessions, emotional tone, etc.)
* Option to cache LLM suggestions for fast recall

---

## Future Features (Post-Proof of Concept)

* Add visualisation of social graph (D3 or static SVG)
* Add editable profile page for caregivers
* Add chat history / rolling transcript viewer
* Add emotion/sentiment detection for tone-aware suggestions
