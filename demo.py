from transformers import pipeline
import json

# Load model

# Use a simpler approach with a pre-built pipeline
rag_pipeline = pipeline("text-generation", model="distilgpt2")

# Load KG
with open("social_graph.json", "r") as f:
    kg = json.load(f)

# Build context
person = kg["people"]["billy"]  # Using Billy instead of Bob
context = person["context"]

# User input
query = "What should I say to Billy?"

# RAG-style prompt
prompt = """I am Will, a 38-year-old father with MND (Motor Neuron Disease). I have a 7-year-old son named Billy who loves Manchester United football.

Billy just asked me: "Dad, did you see the United match last night?"

My response to Billy:"""

# Generate
response = rag_pipeline(
    prompt,
    max_length=100,  # Longer output
    temperature=0.9,  # More creative
    do_sample=True,
    num_return_sequences=1,
    top_p=0.92,  # More focused sampling
    top_k=50,  # Limit vocabulary
)
print("Generated response:")
# For text-generation models, we need to extract just the generated part (not the prompt)
generated_text = response[0]["generated_text"][len(prompt) :]
print(generated_text)
