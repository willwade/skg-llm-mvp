"""
Test script to demonstrate the conversation history functionality.
"""

import json
import datetime
from utils import SocialGraphManager
from llm_interface import LLMInterface

def test_conversation_history():
    """Test the conversation history functionality."""
    print("\n=== Testing Conversation History ===")
    
    # Initialize the social graph manager
    graph_manager = SocialGraphManager("social_graph.json")
    
    # Get a person with conversation history
    person_id = "emma"  # Emma has conversation history
    person_context = graph_manager.get_person_context(person_id)
    
    # Print the person's conversation history
    print(f"\nConversation history for {person_context.get('name')}:")
    conversation_history = person_context.get("conversation_history", [])
    
    if not conversation_history:
        print("No conversation history found.")
    else:
        # Sort by timestamp (most recent first)
        sorted_history = sorted(
            conversation_history, 
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )
        
        for i, conversation in enumerate(sorted_history):
            # Format the timestamp
            timestamp = conversation.get("timestamp", "")
            try:
                dt = datetime.datetime.fromisoformat(timestamp)
                formatted_date = dt.strftime("%B %d, %Y at %I:%M %p")
            except (ValueError, TypeError):
                formatted_date = timestamp
                
            print(f"\nConversation {i+1} on {formatted_date}:")
            
            # Print the messages
            messages = conversation.get("messages", [])
            for message in messages:
                speaker = message.get("speaker", "Unknown")
                text = message.get("text", "")
                print(f"  {speaker}: \"{text}\"")
    
    # Test adding a new conversation
    print("\nAdding a new conversation...")
    new_messages = [
        {"speaker": "Emma", "text": "How are you feeling this afternoon?"},
        {"speaker": "Will", "text": "A bit tired, but the new medication seems to be helping with the muscle stiffness."},
        {"speaker": "Emma", "text": "That's good to hear. Do you want me to bring you anything?"},
        {"speaker": "Will", "text": "A cup of tea would be lovely, thanks."}
    ]
    
    success = graph_manager.add_conversation(person_id, new_messages)
    if success:
        print("New conversation added successfully.")
    else:
        print("Failed to add new conversation.")
    
    # Get the updated person context
    updated_person_context = graph_manager.get_person_context(person_id)
    updated_conversation_history = updated_person_context.get("conversation_history", [])
    
    # Print the updated conversation history
    print("\nUpdated conversation history:")
    if not updated_conversation_history:
        print("No conversation history found.")
    else:
        # Get the most recent conversation
        most_recent = sorted(
            updated_conversation_history, 
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )[0]
        
        # Format the timestamp
        timestamp = most_recent.get("timestamp", "")
        try:
            dt = datetime.datetime.fromisoformat(timestamp)
            formatted_date = dt.strftime("%B %d, %Y at %I:%M %p")
        except (ValueError, TypeError):
            formatted_date = timestamp
            
        print(f"\nMost recent conversation on {formatted_date}:")
        
        # Print the messages
        messages = most_recent.get("messages", [])
        for message in messages:
            speaker = message.get("speaker", "Unknown")
            text = message.get("text", "")
            print(f"  {speaker}: \"{text}\"")
    
    # Test generating a suggestion with conversation history
    print("\nGenerating a suggestion with conversation history...")
    llm_interface = LLMInterface()
    
    if llm_interface.model_loaded:
        # Store the original generate_suggestion method
        original_method = llm_interface.generate_suggestion
        
        # Create a mock method to print the prompt
        def mock_generate_suggestion(*args, **kwargs):
            """Mock method to print the prompt instead of sending it to the LLM."""
            # Call the original method up to the point where it builds the prompt
            person_context = args[0]
            user_input = args[1] if len(args) > 1 else kwargs.get("user_input")
            
            # Extract context information
            name = person_context.get("name", "")
            role = person_context.get("role", "")
            topics = person_context.get("topics", [])
            context = person_context.get("context", "")
            selected_topic = person_context.get("selected_topic", "")
            frequency = person_context.get("frequency", "")
            mood = person_context.get("mood", 3)
            
            # Get mood description
            mood_descriptions = {
                1: "I'm feeling quite down and sad today. My responses might be more subdued.",
                2: "I'm feeling a bit low today. I might be less enthusiastic than usual.",
                3: "I'm feeling okay today - neither particularly happy nor sad.",
                4: "I'm feeling pretty good today. I'm in a positive mood.",
                5: "I'm feeling really happy and upbeat today! I'm in a great mood.",
            }
            mood_description = mood_descriptions.get(mood, mood_descriptions[3])
            
            # Get current date and time
            current_datetime = datetime.datetime.now()
            current_time = current_datetime.strftime("%I:%M %p")
            current_day = current_datetime.strftime("%A")
            current_date = current_datetime.strftime("%B %d, %Y")
            
            # Build enhanced prompt
            prompt = f"""I am Will, a 38-year-old with MND (Motor Neuron Disease) from Manchester.
I am talking to {name}, who is my {role}.
About {name}: {context}
We typically talk about: {', '.join(topics)}
We communicate {frequency}.

Current time: {current_time}
Current day: {current_day}
Current date: {current_date}

My current mood: {mood_description}
"""
            
            # Add communication style based on relationship
            if role in ["wife", "son", "daughter", "mother", "father"]:
                prompt += "I communicate with my family in a warm, loving way, sometimes using inside jokes.\n"
            elif role in ["doctor", "therapist", "nurse"]:
                prompt += "I communicate with healthcare providers in a direct, informative way.\n"
            elif role in ["best mate", "friend"]:
                prompt += "I communicate with friends casually, often with humor and sometimes swearing.\n"
            elif role in ["work colleague", "boss"]:
                prompt += "I communicate with colleagues professionally but still friendly.\n"
            
            # Add topic information if provided
            if selected_topic:
                prompt += f"\nWe are currently discussing {selected_topic}.\n"
            
            # Add conversation history if available
            conversation_history = person_context.get("conversation_history", [])
            if conversation_history:
                # Get the two most recent conversations
                recent_conversations = sorted(
                    conversation_history, 
                    key=lambda x: x.get("timestamp", ""), 
                    reverse=True
                )[:2]
                
                if recent_conversations:
                    prompt += "\nOur recent conversations:\n"
                    
                    for conversation in recent_conversations:
                        # Format the timestamp
                        timestamp = conversation.get("timestamp", "")
                        try:
                            dt = datetime.datetime.fromisoformat(timestamp)
                            formatted_date = dt.strftime("%B %d at %I:%M %p")
                        except (ValueError, TypeError):
                            formatted_date = timestamp
                        
                        prompt += f"\nConversation on {formatted_date}:\n"
                        
                        # Add the messages
                        messages = conversation.get("messages", [])
                        for message in messages:
                            speaker = message.get("speaker", "Unknown")
                            text = message.get("text", "")
                            prompt += f'{speaker}: "{text}"\n'
            
            # Print the prompt
            print("\n=== PROMPT WITH CONVERSATION HISTORY ===")
            print(prompt)
            print("=======================================\n")
            
            # Return a mock response
            return "This is a mock response to test conversation history inclusion in the prompt."
        
        # Replace the original method with our mock method
        llm_interface.generate_suggestion = mock_generate_suggestion
        
        # Test with a user input
        user_input = "Do you think you'll be up for dinner with the kids tonight?"
        llm_interface.generate_suggestion(updated_person_context, user_input=user_input)
        
        # Restore the original method
        llm_interface.generate_suggestion = original_method
    else:
        print("LLM model not loaded, skipping prompt generation test.")
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_conversation_history()
