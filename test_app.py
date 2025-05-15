import sys
import os

print("Starting test...")

# Test importing the modules
try:
    import gradio as gr
    import whisper
    import random
    import time
    from utils import SocialGraphManager, SuggestionGenerator
    print("All modules imported successfully")
except Exception as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Test loading the social graph
try:
    social_graph = SocialGraphManager("social_graph.json")
    print("Social graph loaded successfully")
except Exception as e:
    print(f"Error loading social graph: {e}")
    sys.exit(1)

# Test initializing the suggestion generator
try:
    suggestion_generator = SuggestionGenerator("distilgpt2")  # Use a simpler model for testing
    print("Suggestion generator initialized successfully")
except Exception as e:
    print(f"Error initializing suggestion generator: {e}")
    sys.exit(1)

# Test getting people from the social graph
try:
    people = social_graph.get_people_list()
    print(f"Found {len(people)} people in the social graph")
    if people:
        print(f"First person: {people[0]['name']} ({people[0]['role']})")
except Exception as e:
    print(f"Error getting people from social graph: {e}")
    sys.exit(1)

# Test getting person context
try:
    if people:
        person_id = people[0]['id']
        person_context = social_graph.get_person_context(person_id)
        print(f"Got context for {person_context.get('name', 'unknown')}")
except Exception as e:
    print(f"Error getting person context: {e}")
    sys.exit(1)

print("All tests passed successfully!")
