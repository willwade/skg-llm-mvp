"""
Test script for AAC Social Graph Assistant components.
Run this script to verify that the components work correctly.
"""

import json
from utils import SocialGraphManager, SuggestionGenerator

def test_social_graph_manager():
    """Test the SocialGraphManager class."""
    print("\n=== Testing SocialGraphManager ===")
    
    # Initialize the social graph manager
    graph_manager = SocialGraphManager("social_graph.json")
    
    # Test loading the graph
    print(f"Loaded graph with {len(graph_manager.graph.get('people', {}))} people")
    
    # Test getting people list
    people = graph_manager.get_people_list()
    print(f"People in the graph: {', '.join([p['name'] for p in people])}")
    
    # Test getting person context
    if people:
        person_id = people[0]['id']
        person_context = graph_manager.get_person_context(person_id)
        print(f"\nContext for {person_context.get('name', person_id)}:")
        print(f"  Role: {person_context.get('role', '')}")
        print(f"  Topics: {', '.join(person_context.get('topics', []))}")
        print(f"  Frequency: {person_context.get('frequency', '')}")
        
        # Test getting relevant phrases
        phrases = graph_manager.get_relevant_phrases(person_id)
        print(f"\nCommon phrases for {person_context.get('name', person_id)}:")
        for phrase in phrases:
            print(f"  - {phrase}")
    
    # Test getting common utterances
    categories = list(graph_manager.graph.get("common_utterances", {}).keys())
    if categories:
        category = categories[0]
        utterances = graph_manager.get_common_utterances(category)
        print(f"\nCommon utterances in category '{category}':")
        for utterance in utterances:
            print(f"  - {utterance}")
    
    return graph_manager

def test_suggestion_generator(graph_manager):
    """Test the SuggestionGenerator class."""
    print("\n=== Testing SuggestionGenerator ===")
    
    # Initialize the suggestion generator
    try:
        generator = SuggestionGenerator()
        
        # Test generating a suggestion
        people = graph_manager.get_people_list()
        if people and generator.model_loaded:
            person_id = people[0]['id']
            person_context = graph_manager.get_person_context(person_id)
            
            print(f"\nGenerating suggestion for {person_context.get('name', person_id)}...")
            suggestion = generator.generate_suggestion(person_context)
            print(f"Suggestion: {suggestion}")
            
            # Test with user input
            user_input = "We were talking about the weather yesterday."
            print(f"\nGenerating suggestion with user input: '{user_input}'")
            suggestion = generator.generate_suggestion(person_context, user_input)
            print(f"Suggestion: {suggestion}")
        elif not generator.model_loaded:
            print("Model not loaded. Skipping suggestion generation test.")
        else:
            print("No people in the graph. Skipping suggestion generation test.")
    except Exception as e:
        print(f"Error testing suggestion generator: {e}")

if __name__ == "__main__":
    print("Testing AAC Social Graph Assistant components...")
    graph_manager = test_social_graph_manager()
    test_suggestion_generator(graph_manager)
    print("\nTests completed.")
