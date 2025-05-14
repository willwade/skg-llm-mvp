from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import json

# Load model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
rag_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Load KG
with open("social_graph.json", "r") as f:
    kg = json.load(f)

# Build context
person = kg["people"]["bob"]
context = f"Bob is the user's son. They talk about football weekly. Last conversation was about coaching changes."

# User input
query = "What should I say to Bob?"

# RAG-style prompt
prompt = f"""Context: {context}
User wants to say something appropriate to Bob. Suggest a phrase:"""

# Generate
response = rag_pipeline(prompt, max_length=50)
print(response[0]["generated_text"])
