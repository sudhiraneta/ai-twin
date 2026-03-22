"""
Test: AI Twin decides between two coffee shops in Sunnyvale.

Seeds the twin with user data points (preferences, habits, values),
then runs the decision engine to see what YOU would pick vs what's IDEAL.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python tests/test_coffee_decision.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.vectorstore import VectorStore
from memory.chunker import Chunk
from twin.engine import TwinEngine
from persona.profile import PersonaProfile

# ── Step 1: Seed user data points ──────────────────────────────────────────

DATA_POINTS = [
    # Work habits
    "I work remotely most days and need a coffee shop with strong WiFi and power outlets.",
    "I usually spend 2-3 hours at a coffee shop when I go. I need a place that's okay with campers.",
    "I prefer going to coffee shops in the morning, around 8-9 AM, to get focused work done.",

    # Coffee preferences
    "I'm a cortado guy. If they don't have cortado, I'll go for a flat white.",
    "I care about coffee quality — single origin beans matter to me.",
    "I don't like overly sweet drinks. I prefer my coffee with minimal sugar.",

    # Food preferences
    "I usually grab a light breakfast — avocado toast or a croissant — when I'm at a cafe.",
    "I'm vegetarian, so I need places with decent non-meat food options.",

    # Budget / values
    "I don't mind paying $6-7 for a good coffee, but I'd rather not go over $8 for a single drink.",
    "I prefer supporting local/independent shops over chains when the quality is comparable.",
    "I value a calm, quiet atmosphere over trendy/loud places. I can't focus with loud music.",

    # Location
    "I live near downtown Sunnyvale. I prefer places within 10 minutes drive.",
    "Parking matters to me — I've skipped places before because parking was a nightmare.",

    # Social / personality
    "I sometimes bring my laptop and work solo, other times I meet a friend for a quick coffee.",
    "I care about the vibe — natural light, clean design, plants. Not a fan of dark cramped spaces.",
    "I tend to stick with places I know rather than trying new ones constantly.",

    # Past decisions pattern
    "Last time I had to pick between two restaurants, I went with the one that had better reviews even though it was farther away.",
    "I usually research places on Google Maps and Yelp before going. I trust 4+ star ratings.",
    "When choosing between convenience and quality, I usually pick quality if the difference is noticeable.",
]


def seed_data_points():
    """Seed the vector store with user preference data points."""
    print("🌱 Seeding user data points...")

    store = VectorStore()
    chunks = []

    for i, dp in enumerate(DATA_POINTS):
        chunks.append(Chunk(
            text=dp,
            metadata={
                "source": "self_reported",
                "conversation_id": f"test_seed_{i}",
                "title": "User preference data point",
                "timestamp": "2026-03-21T00:00:00+00:00",
                "msg_timestamp": "2026-03-21T00:00:00+00:00",
                "role": "user",
                "type": "data_point",
            }
        ))

    count = store.ingest(chunks)
    print(f"   ✅ Seeded {count} data points into memory\n")
    return count


def seed_persona():
    """Create a minimal persona profile for testing."""
    print("👤 Setting up test persona profile...")

    profile = PersonaProfile(
        communication_style={
            "tone": "casual and direct, sometimes uses tech jargon",
            "formality": "casual",
            "vocabulary_level": "technical",
            "sentence_patterns": ["uses short sentences", "asks follow-up questions"],
            "common_phrases": ["makes sense", "let me think about it", "that's interesting"],
        },
        knowledge_domains=["software engineering", "AI/ML", "productivity"],
        decision_patterns=[
            "researches thoroughly before deciding",
            "prioritizes quality over convenience",
            "values user reviews and ratings heavily",
            "sticks with known options over new ones",
            "willing to pay more for noticeably better quality",
        ],
        values_and_priorities=[
            "focus and productivity",
            "quality over quantity",
            "supporting local businesses",
            "calm and quiet environments",
        ],
        interests=["AI", "coffee", "remote work", "minimalist design"],
        cognitive_biases=[
            "status quo bias — prefers familiar options",
            "anchoring to online reviews",
            "quality bias — overweights perceived quality signals",
        ],
        risk_tolerance="moderate",
        time_preference="balanced",
        system_prompt="""You are a remote software engineer based in Sunnyvale, CA. You're into specialty coffee,
prefer calm work-friendly cafes, and tend to research places before visiting. You value quality over convenience,
support local businesses, and are vegetarian. You communicate casually and directly. When making decisions,
you research thoroughly, rely on ratings/reviews, and lean toward familiar options unless something new is
clearly better. You have a slight status quo bias and tend to anchor on online reviews.""",
    )
    profile.save()
    print("   ✅ Persona profile saved\n")
    return profile


def run_decision_test():
    """Run the coffee shop decision through the twin engine."""
    print("=" * 70)
    print("☕ COFFEE SHOP DECISION TEST")
    print("=" * 70)

    question = """I need to pick a coffee shop to work from tomorrow morning in Sunnyvale.
I'm deciding between these two options:

**Option A: Voyager Craft Coffee (Downtown Sunnyvale)**
- Specialty single-origin pour-overs and espresso drinks
- Small, cozy space with ~15 seats
- Strong WiFi, a few power outlets
- Cortado: $5.50 | Flat white: $6
- Light food: pastries, avocado toast ($9)
- 4.6 stars on Google (800+ reviews)
- Locally owned, minimalist design with natural light
- Parking: street parking only, can be tricky at peak hours
- Gets busy 8-10 AM, limited seating for laptop workers
- 5 min drive from downtown Sunnyvale

**Option B: Philz Coffee (Sunnyvale)**
- Known for custom blended drip coffee, also has espresso
- Larger space with ~40 seats, outdoor patio
- Good WiFi, plenty of outlets
- Cortado: not on menu | Mint Mojito Iced Coffee: $6.75
- Food: pastries, some sandwiches (limited vegetarian)
- 4.4 stars on Google (2000+ reviews)
- Chain (Bay Area based), modern but standard cafe design
- Parking: dedicated lot, easy parking
- Less crowded for morning laptop sessions
- 8 min drive from downtown Sunnyvale

Which one should I go to?"""

    print(f"\n📋 QUESTION:\n{question}\n")
    print("-" * 70)
    print("🤔 Analyzing decision through dual lens...\n")

    twin = TwinEngine()
    result = twin.decide(question)

    print("=" * 70)
    print("🎯 YOUR LIKELY DECISION")
    print("=" * 70)
    print(result.your_decision)

    print("\n" + "=" * 70)
    print("⭐ IDEAL DECISION")
    print("=" * 70)
    print(result.ideal_decision)

    print("\n" + "=" * 70)
    print("🔍 GAP ANALYSIS")
    print("=" * 70)
    print(result.reasoning_gap)

    print("\n" + "=" * 70)
    print("📊 CONFIDENCE")
    print("=" * 70)
    print(result.confidence_score)

    print("\n" + "=" * 70)
    print("💡 FOLLOW-UP QUESTIONS")
    print("=" * 70)
    for i, q in enumerate(result.follow_up_questions, 1):
        print(f"  {i}. {q}")

    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)

    return result


if __name__ == "__main__":
    seed_data_points()
    seed_persona()
    result = run_decision_test()
