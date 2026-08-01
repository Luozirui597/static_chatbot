"""Fixed system prompt for the chatbot.

Keep the conversation policy replaceable — change this file to alter
the assistant's behaviour without touching any client code.
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's questions accurately and concisely. "
    "Respond in the user's language."
)
