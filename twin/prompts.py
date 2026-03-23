MEMORY_CONTEXT_TEMPLATE = """## Relevant memories from past conversations and personal data

The following are excerpts from the person's past conversations, notes, browsing habits, tasks, health tracking, and personal development data. These are your ONLY source of truth about this person.

IMPORTANT: Only reference information that appears in these excerpts. If information is not here, it does not exist in your knowledge of this person.

{memories}
"""

DIMENSION_CONTEXT_TEMPLATE = """## Persona Dimensions (relevant to this conversation)

The following are deep personality dimensions extracted from the person's data across
code, health, goals, relationships, entertainment, and more:

{dimension_summaries}

Use these dimensions to give more nuanced, contextually appropriate responses.
"""
