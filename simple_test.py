"""
Simple test script for conversation history.
"""

from utils import SocialGraphManager

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
    for i, conversation in enumerate(conversation_history):
        print(f"\nConversation {i+1}:")
        
        # Print the timestamp
        timestamp = conversation.get("timestamp", "")
        print(f"Timestamp: {timestamp}")
        
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
    # Count the conversations
    print(f"Found {len(updated_conversation_history)} conversations.")
    
    # Get the most recent conversation
    most_recent = sorted(
        updated_conversation_history, 
        key=lambda x: x.get("timestamp", ""), 
        reverse=True
    )[0]
    
    # Print the timestamp
    timestamp = most_recent.get("timestamp", "")
    print(f"Most recent timestamp: {timestamp}")
    
    # Print the messages
    messages = most_recent.get("messages", [])
    for message in messages:
        speaker = message.get("speaker", "Unknown")
        text = message.get("text", "")
        print(f"  {speaker}: \"{text}\"")

print("\nTest completed.")
