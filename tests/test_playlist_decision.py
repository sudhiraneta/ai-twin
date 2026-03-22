"""
Test: AI Twin decides between two songs from a playlist.

Seeds the twin with user music preferences and habits,
then runs the decision engine to see what Sudhira would pick vs what's IDEAL.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python tests/test_playlist_decision.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.vectorstore import VectorStore
from memory.chunker import Chunk
from twin.engine import TwinEngine
from persona.profile import PersonaProfile


# ── Step 1: Seed user music data points ───────────────────────────────────────

DATA_POINTS = [
    # Music taste
    "I love listening to chill, melodic tracks when I'm working — lo-fi beats, indie electronic, or ambient stuff.",
    "I tend to lean toward songs with a good groove and interesting production over raw vocals.",
    "When I'm coding, I prefer instrumental or low-vocal tracks so lyrics don't distract me.",
    "I'm really into artists like Bonobo, Tycho, Khruangbin, and Tame Impala.",
    "I appreciate when a song builds gradually — I'm not a fan of songs that are all energy from the start.",

    # Mood and context
    "In the morning I like something mellow and warm to ease into the day — nothing too intense.",
    "If I'm doing focused deep work, I need something that fades into the background but keeps me in flow.",
    "When I'm in a good mood or it's the weekend, I'll listen to something more upbeat and funky.",

    # Listening habits
    "I usually listen to music on Spotify. I have a mix of curated playlists and Discover Weekly.",
    "I tend to replay songs I like on loop for days before moving on.",
    "I value production quality a lot — a well-mixed track stands out to me more than lyrics.",
    "I generally skip songs within the first 30 seconds if they don't hook me with the vibe.",

    # Preferences
    "I don't really enjoy heavy metal, aggressive rap, or country music.",
    "I like songs in the 3-5 minute range. Anything over 7 minutes needs to be really good to keep my attention.",
    "I tend to discover new music through recommendations from friends or algorithm suggestions, not radio.",
    "I prefer songs that feel like a journey — with layers that unfold as you listen.",
]


def seed_data_points():
    """Seed the vector store with user music preference data points."""
    print("Seeding user music preference data points...")

    store = VectorStore()
    chunks = []

    for i, dp in enumerate(DATA_POINTS):
        chunks.append(Chunk(
            text=dp,
            metadata={
                "source": "self_reported",
                "conversation_id": f"test_playlist_seed_{i}",
                "title": "User music preference data point",
                "timestamp": "2026-03-21T00:00:00+00:00",
                "msg_timestamp": "2026-03-21T00:00:00+00:00",
                "role": "user",
                "type": "data_point",
            }
        ))

    count = store.ingest(chunks)
    print(f"   Seeded {count} data points into memory\n")
    return count


def seed_persona():
    """Create a persona profile with music-relevant traits."""
    print("Setting up test persona profile...")

    profile = PersonaProfile(
        communication_style={
            "tone": "casual and direct, sometimes uses tech jargon",
            "formality": "casual",
            "vocabulary_level": "technical",
            "sentence_patterns": ["uses short sentences", "asks follow-up questions"],
            "common_phrases": ["makes sense", "let me think about it", "that's interesting"],
        },
        knowledge_domains=["software engineering", "AI/ML", "music production basics"],
        decision_patterns=[
            "goes with gut feeling on music but rationalizes it after",
            "leans toward familiar vibes over completely new sounds",
            "values production quality and sonic texture highly",
            "prefers gradual builds over instant energy",
            "will replay a favorite track obsessively",
        ],
        values_and_priorities=[
            "focus and productivity",
            "aesthetic quality",
            "calm and immersive experiences",
            "discovering new music that fits existing taste",
        ],
        interests=["AI", "coffee", "music", "remote work", "minimalist design"],
        cognitive_biases=[
            "familiarity bias — gravitates toward sounds similar to what he already likes",
            "production quality bias — overweights mix/mastering quality over songwriting",
            "recency bias — tends to favor recently discovered tracks",
        ],
        risk_tolerance="moderate",
        time_preference="balanced",
        system_prompt="""You are a remote software engineer who loves chill, atmospheric music.
You listen to a lot of lo-fi, indie electronic, and ambient tracks while working. Your favorite
artists include Bonobo, Tycho, Khruangbin, and Tame Impala. You value production quality,
gradual builds, and songs that create a mood without being distracting. You lean toward
familiar vibes and tend to replay tracks you love. You communicate casually and directly.""",
    )
    profile.save()
    print("   Persona profile saved\n")
    return profile


def run_decision_test():
    """Run the playlist song decision through the twin engine."""
    print("=" * 70)
    print("PLAYLIST SONG DECISION TEST")
    print("=" * 70)

    question = """I'm about to start a deep work coding session and I want to pick the right
track to kick it off from my playlist. I'm deciding between these two songs:

**Option A: "Kiara" by Bonobo**
- Genre: Downtempo / Electronic
- Duration: 6:32
- Vibe: Slow, cinematic build with layered strings, gentle percussion, and a lush, immersive atmosphere
- Vocals: Minimal — mostly instrumental with ambient vocal samples
- Production: Beautifully mixed, rich textures, warm analog feel
- Energy: Starts mellow, builds to an emotional crescendo in the last 2 minutes
- Great for: Deep focus, getting into a flow state, contemplative mood
- From the album "Black Sands" — a classic in the downtempo scene

**Option B: "Breathe Deeper" by Tame Impala**
- Genre: Psychedelic Pop / Synth-Pop
- Duration: 6:12
- Vibe: Groovy, warm, hypnotic — feels like a sunny drive with the windows down
- Vocals: Kevin Parker's signature dreamy, layered vocals throughout
- Production: Immaculate studio production, synth-heavy, tight drums
- Energy: Upbeat from the start, danceable groove, maintains steady energy
- Great for: Energized focus, upbeat mood, when you want to feel good while working
- From the album "The Slow Rush" — widely acclaimed synth-pop record

Which one should I listen to?"""

    print(f"\nQUESTION:\n{question}\n")
    print("-" * 70)
    print("Analyzing decision through dual lens...\n")

    twin = TwinEngine()
    result = twin.decide(question)

    print("=" * 70)
    print("YOUR LIKELY DECISION")
    print("=" * 70)
    print(result.your_decision)

    print("\n" + "=" * 70)
    print("IDEAL DECISION")
    print("=" * 70)
    print(result.ideal_decision)

    print("\n" + "=" * 70)
    print("GAP ANALYSIS")
    print("=" * 70)
    print(result.reasoning_gap)

    print("\n" + "=" * 70)
    print("CONFIDENCE")
    print("=" * 70)
    print(result.confidence_score)

    print("\n" + "=" * 70)
    print("FOLLOW-UP QUESTIONS")
    print("=" * 70)
    for i, q in enumerate(result.follow_up_questions, 1):
        print(f"  {i}. {q}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)

    return result


if __name__ == "__main__":
    seed_data_points()
    seed_persona()
    result = run_decision_test()
