"""
Lifestyle Decision Test for AI Twin

Tests everyday choices: coffee shops, hikes, restaurants, weekend plans.
No API key needed.

Usage:
    python tests/test_lifestyle_decisions.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.vectorstore import VectorStore
from memory.chunker import Chunk
from twin.engine import TwinEngine, DecisionResponse
from persona.profile import PersonaProfile


# ── Lifestyle data points ────────────────────────────────────────────────────

DATA_POINTS = [
    # Coffee
    "I'm a cortado guy. If they don't have cortado, I'll go for a flat white.",
    "I care about coffee quality — single origin beans matter to me.",
    "I prefer supporting local/independent shops over chains when quality is comparable.",
    "I value a calm, quiet atmosphere over trendy/loud places. I can't focus with loud music.",
    "I usually spend 2-3 hours at a coffee shop when I go. I need a place okay with campers.",
    "I prefer going to coffee shops in the morning, around 8-9 AM, to get focused work done.",
    "I don't mind paying $6-7 for a good coffee, but I'd rather not go over $8 for a single drink.",
    "Parking matters to me — I've skipped places before because parking was a nightmare.",
    "I care about the vibe — natural light, clean design, plants. Not a fan of dark cramped spaces.",

    # Outdoors / hiking
    "I enjoy hiking on weekends when the weather is nice. It's my way to decompress from screen time.",
    "I prefer moderate hikes — 4 to 8 miles, some elevation gain, but not extreme scrambles.",
    "I like hikes with good views at the top or along the way. The payoff matters to me.",
    "I usually hike in the morning to beat the crowds and the heat.",
    "I tend to go to trails within 45 minutes drive. I don't want to spend 2 hours just driving to a trailhead.",
    "I bring a small backpack with water, snacks, and a camera. I like taking photos on hikes.",
    "I prefer well-maintained trails with clear markings. I'm not into bushwhacking or sketchy routes.",
    "I've done most of the popular trails near Sunnyvale — Rancho San Antonio, Fremont Older, Stevens Creek.",
    "I like trails that aren't too crowded. Weekend mornings at popular trails can be a zoo.",
    "I'm okay with moderate difficulty but I avoid trails with long exposed sections in direct sun.",
    "After a hike I usually grab brunch or coffee somewhere nearby. It's part of the ritual.",

    # Food / lifestyle
    "I'm vegetarian, so I need places with decent non-meat food options.",
    "I usually grab a light breakfast — avocado toast or a croissant — when I'm at a cafe.",
    "I live near downtown Sunnyvale. I prefer places within 10 minutes drive.",
    "I tend to stick with places I know rather than trying new ones constantly.",
    "Last time I had to pick between two options, I went with the one that had better reviews even though it was farther.",
    "I usually research places on Google Maps and Yelp before going. I trust 4+ star ratings.",
    "When choosing between convenience and quality, I usually pick quality if the difference is noticeable.",

    # Weekend patterns
    "On weekends I like a mix of outdoor activity and chill time. I don't like packed schedules.",
    "Saturday mornings are my favorite — hike or coffee shop, then a relaxed afternoon.",
    "I sometimes bring my laptop to a cafe after a hike to write or do personal project work.",
]


DECISION_SCENARIOS = [
    # ── Coffee Shop Decisions ──
    {
        "id": "coffee-01",
        "category": "Coffee Shop",
        "title": "Voyager Craft Coffee vs Philz Coffee",
        "question": """Picking a coffee shop to work from tomorrow morning in Sunnyvale.

**Option A: Voyager Craft Coffee (Downtown Sunnyvale)**
- Specialty single-origin pour-overs and espresso drinks
- Small, cozy space with ~15 seats
- Strong WiFi, a few power outlets
- Cortado: $5.50 | Flat white: $6
- Light food: pastries, avocado toast ($9)
- 4.6 stars on Google (800+ reviews)
- Locally owned, minimalist design with natural light
- Parking: street parking only, tricky at peak hours
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

Which one should I go to?""",
    },
    {
        "id": "coffee-02",
        "category": "Coffee Shop",
        "title": "New trendy spot vs reliable regular",
        "question": """A new specialty coffee shop just opened 2 weeks ago near Murphy Ave.
My friend said it's great. But I also have my usual reliable spot.

**Option A: Kaffe (new, Murphy Ave)**
- Just opened — no Google reviews yet, but friend says it's "amazing"
- Third-wave specialty coffee, rotating single-origin menu
- Cortado: $6 | Pour-over: $7
- Small space, about 12 seats, modern industrial design
- WiFi status unknown, unclear about outlets
- Food: pastries from a local bakery, vegan options available
- Parking: street parking on Murphy Ave (usually fine before 10 AM)
- 3 min drive from me
- No idea about crowd levels or laptop-friendliness

**Option B: Voyager Craft Coffee (my usual spot)**
- I know exactly what to expect — great cortado, good vibe
- Reliably good WiFi and a few outlets
- 4.6 stars, consistent quality
- I have my favorite seat by the window
- Gets busy after 9 AM but manageable if I arrive by 8:15
- 5 min drive
- Familiar faces, baristas know my order

Should I try the new place or stick with Voyager?""",
    },
    {
        "id": "coffee-03",
        "category": "Coffee Shop",
        "title": "Best coffee far away vs decent coffee nearby",
        "question": """I want coffee this morning but I'm also a bit tired and lazy about driving.

**Option A: Chromatic Coffee (Santa Clara)**
- One of the best roasters in the South Bay
- Incredible single-origin espresso, cortado is exceptional
- Minimalist space, great natural light, quiet atmosphere
- 4.7 stars on Google (600+ reviews)
- 15-20 min drive from my place (depending on traffic)
- Limited parking, might need to circle
- Food: limited pastries
- Haven't been in a month, been meaning to go back

**Option B: Peet's Coffee (downtown Sunnyvale)**
- Solid chain coffee, not specialty but decent
- Cortado: passable but not special
- Large space, always has seating, good WiFi, plenty of outlets
- 4.1 stars on Google
- 3 min drive, parking lot right there
- Reliable, no surprises
- Can be there in 10 minutes total

Which one today?""",
    },

    # ── Hiking Decisions ──
    {
        "id": "hike-01",
        "category": "Hiking",
        "title": "Mission Peak vs Rancho San Antonio",
        "question": """Saturday morning hike. Weather is perfect — 68°F, clear skies. Deciding between:

**Option A: Mission Peak (Fremont)**
- 6.2 miles round trip, 2,100 ft elevation gain
- Famous summit with the iconic pole and 360° Bay Area views
- Strenuous — steep sustained climb, exposed sections
- Very popular, parking fills by 8 AM on weekends
- 30 min drive from Sunnyvale
- No shade on most of the trail — direct sun
- The view at the top is incredible, great photo opportunity
- Trail is well-maintained but crowded

**Option B: Rancho San Antonio (Cupertino)**
- Multiple trail options: 4-8 miles depending on route
- Moderate elevation gain (~800-1,200 ft for longer loops)
- Mix of shaded oak woodland and open meadow
- Popular but has many trails so crowds disperse
- 15 min drive from Sunnyvale
- Deer Creek Trail loop is my favorite (~5.5 miles)
- Good views but nothing as dramatic as Mission Peak summit
- I've done this trail many times — familiar and comfortable

Which hike should I do?""",
    },
    {
        "id": "hike-02",
        "category": "Hiking",
        "title": "New trail I've never done vs familiar favorite",
        "question": """Looking for a Saturday morning hike. I've been doing the same trails lately.

**Option A: Castle Rock State Park (Saratoga)**
- 5.5 mile loop through redwood forest and sandstone formations
- Moderate difficulty, ~800 ft elevation gain
- Unique rock formations, some scrambling opportunities
- Less crowded than closer trails
- 35 min drive from Sunnyvale
- I've never been — would be a new experience
- Trail reviews say well-maintained with clear markers
- 4.5 stars on AllTrails (1,200 reviews)
- Shaded by redwoods — great for warm days
- No real summit view but the forest scenery is the draw

**Option B: Fremont Older Open Space (Cupertino)**
- 4.5 mile loop, ~600 ft elevation gain
- Easy-moderate, rolling hills with bay views
- 15 min drive from Sunnyvale
- I've done it 5+ times — know every turn
- Usually not too crowded on Saturday mornings
- Nice but nothing new — same views, same trail
- Quick and easy, leaves more of the day free
- Good post-hike brunch spots nearby in Saratoga Village

Should I try something new or go with what I know?""",
    },
    {
        "id": "hike-03",
        "category": "Hiking",
        "title": "Solo morning hike vs group hike with friends",
        "question": """Two options for this Saturday:

**Option A: Solo hike at Windy Hill (Portola Valley)**
- 7 mile loop, ~1,100 ft elevation gain
- Beautiful ridge trail with panoramic bay views
- 25 min drive from Sunnyvale
- I can go at my own pace, bring my camera, enjoy the quiet
- Start at 7 AM, be done by 10:30, grab coffee after
- Meditative solo experience — good for clearing my head
- I've been wanting to do this trail for weeks

**Option B: Group hike at Purisima Creek (Half Moon Bay side)**
- 5 miles, moderate, through redwood forest
- 3 friends are going, leaving at 9 AM
- 40 min drive from Sunnyvale
- More social, good conversation on the trail
- Pace will be slower (group dynamics)
- Haven't hung out with this group in a few weeks
- They want to grab lunch in Half Moon Bay after
- Would take up most of the morning + early afternoon

Which should I pick?""",
    },
    {
        "id": "hike-04",
        "category": "Hiking",
        "title": "Hard rewarding hike vs easy scenic walk",
        "question": """Sunday morning. I worked out Friday and my legs are a bit sore.

**Option A: Black Mountain Trail (Monte Bello, Los Altos)**
- 8.5 miles round trip, 1,800 ft elevation gain
- Challenging climb through grasslands to the summit
- Stunning views of the Santa Cruz Mountains and Pacific Ocean on clear days
- Well-maintained but steep in sections
- 20 min drive from Sunnyvale
- Would be a great workout and the views are worth it
- But my legs are already sore from Friday's gym session

**Option B: Shoreline Park Trail (Mountain View)**
- 3.5 mile flat loop around the lake
- Basically a walk — no elevation gain
- Great bird watching, peaceful water views
- 10 min drive
- Very easy on the body, good for a recovery day
- Can bring my camera for bird photos
- Quick — done in about an hour
- Grab coffee at Red Rock after

Easy recovery walk or push through the soreness for a big payoff?""",
    },
]


# ── Pre-built decision responses ─────────────────────────────────────────────

DECISION_RESPONSES = {
    "coffee-01": """## Your Likely Decision
You'd pick **Option A: Voyager Craft Coffee**. This aligns with nearly every preference you have: cortado on the menu ($5.50), single-origin specialty coffee, locally owned, minimalist design with natural light, and a 4.6-star rating. You value quality over convenience, you support local over chains, and the vibe matches exactly what you look for. You'd accept the tricky street parking and smaller space because the coffee quality and atmosphere are noticeably better than Philz for your taste.

## Ideal Decision
**Option A: Voyager Craft Coffee** is the ideal choice for your priorities. The coffee quality is clearly superior for an espresso/cortado drinker, the atmosphere matches your work style, and it's locally owned. The one risk — busy seating at peak hours — is manageable by arriving at 8 AM. Philz is a fine backup for days when you need guaranteed seating and easy parking, but for a planned morning work session, Voyager wins.

## Gap Analysis
No gap — your instinct and the ideal align perfectly here. Voyager checks every box: cortado, single-origin, local, calm vibe, good ratings. The only scenario where Philz would be the better choice is if you needed guaranteed seating and easy parking on a day when you can't arrive early.

## Confidence Score
HIGH — Every data point about your coffee and work environment preferences points to Voyager. This is the most predictable decision in the set.

## Follow-Up Questions
- Do you have a backup plan if Voyager is full when you arrive?
- Have you considered becoming a "regular" at two spots so you always have an option?""",

    "coffee-02": """## Your Likely Decision
You'd pick **Option B: Voyager (your usual spot)**. Your status quo bias is strong — "I tend to stick with places I know rather than trying new ones constantly." You know the cortado is great, you have your favorite seat, the baristas know your order. The new place has too many unknowns: no reviews, unknown WiFi, unclear laptop policy. You'd tell yourself "I'll try Kaffe once it has reviews and I hear more about it."

## Ideal Decision
**Option A: Kaffe (the new spot)** is actually the more productive choice for this specific situation. It's closer (3 min vs 5 min), your friend — a trusted source — vouches for it, and trying it now while it's new means fewer crowds. The unknowns (WiFi, seating) are low-risk — if it doesn't work out, you've lost 15 minutes and can drive to Voyager. Going early on a weekday morning to a brand new spot almost guarantees you'll have space.

## Gap Analysis
Moderate gap. Your status quo bias is overweighting the comfort of Voyager and overweighting the "risk" of the new place. But the actual downside of trying Kaffe is minimal: worst case, you spend 10 minutes there, realize it doesn't work, and drive 3 minutes to Voyager. The upside is discovering a new spot that's closer, potentially better, and less crowded. Your "I research on Google Maps first" habit creates a blind spot for places too new to have reviews.

## Confidence Score
HIGH — Your "stick with what I know" pattern is one of your most consistent behaviors. The probability of you choosing Voyager over an unknown new place is very high.

## Follow-Up Questions
- How often has a friend's recommendation turned out to be right for you?
- Would you be more likely to try it if you went just for coffee (no laptop) to scout it first?""",

    "coffee-03": """## Your Likely Decision
You'd pick **Option A: Chromatic Coffee**. Despite being tired, your "quality over convenience when the difference is noticeable" pattern kicks in here. Chromatic is clearly the better coffee — 4.7 stars, one of the best roasters in the South Bay, and you've been meaning to go back. You'd rationalize the drive: "It's only 15-20 minutes, and I haven't been in a month." Your quality bias overrides your laziness.

## Ideal Decision
**Option B: Peet's Coffee** is the more productive choice this morning. You described yourself as "tired and lazy about driving." Chromatic is 15-20 min drive with limited parking vs 3 min to Peet's with a parking lot. On a low-energy morning, the friction of driving, parking, and dealing with a 30-40 minute round trip commute will cut into your productive time. Peet's gets you seated with coffee in 10 minutes. Save Chromatic for a morning when you're energized and want to make it an outing.

## Gap Analysis
Moderate gap. Your quality bias makes you underweight the real cost of friction on a low-energy day. The difference between a great cortado and a decent one is real but not transformative — you'll work equally well with either coffee in front of you. What IS transformative is losing 30+ extra minutes to driving and parking when you're already feeling lazy. On high-energy days, Chromatic is the right call. On "tired and lazy" days, the productive choice is minimizing friction.

## Confidence Score
MEDIUM — Your quality bias is strong, but you explicitly flagged being tired. This creates genuine tension. 60% chance you'd still go to Chromatic and rationalize it; 40% chance laziness wins and you go to Peet's.

## Follow-Up Questions
- When you're tired, do you usually push through for quality or give yourself permission to take the easy option?
- Is part of the Chromatic appeal the drive itself — a change of scenery, getting out of the neighborhood?""",

    "hike-01": """## Your Likely Decision
You'd pick **Option B: Rancho San Antonio**. Multiple signals point here: it's your familiar trail ("I've done this trail many times"), 15 min drive vs 30 min, moderate difficulty matches your preference (4-8 miles, some elevation), shaded sections (you avoid long exposed sections in direct sun), and the Deer Creek loop is your stated favorite. Mission Peak's strenuous climb, exposed sun, and parking chaos would push you away.

## Ideal Decision
**Option B: Rancho San Antonio** is the ideal choice too. On a perfect weather Saturday, Mission Peak's parking fills by 8 AM and the trail becomes a congested single-file line. Your preferences explicitly include "avoid trails with long exposed sections in direct sun" and "I like trails that aren't too crowded." Rancho's multiple trail options disperse crowds, the Deer Creek loop gives you a satisfying 5.5-mile hike with elevation, and you're done in time for post-hike coffee.

## Gap Analysis
No gap. Your instincts are correct. Mission Peak is a great hike but it's wrong for YOUR priorities: you dislike crowds, exposed sun, parking hassles, and extreme strenuous climbs. Rancho gives you everything you want. The only reason to pick Mission Peak is if you specifically want the dramatic summit photo — but you can do that on an early weekday morning when it's less crowded.

## Confidence Score
HIGH — Your stated preferences almost disqualify Mission Peak: crowded, exposed, strenuous, parking nightmare. Rancho matches every criterion you care about.

## Follow-Up Questions
- When was the last time you did Mission Peak? Would you consider it on an early weekday?
- Are you getting bored of the Deer Creek loop, or does the familiarity feel comfortable?""",

    "hike-02": """## Your Likely Decision
You'd pick **Option B: Fremont Older**. Your status quo bias ("I tend to stick with places I know"), the closer drive (15 min vs 35 min), and the fact that it "leaves more of the day free" all point here. You'd tell yourself "Castle Rock sounds cool but 35 minutes is a lot of driving. I'll go another time."

## Ideal Decision
**Option A: Castle Rock State Park** is the ideal choice here. You've acknowledged you've been doing the same trails lately — that's your own signal that you need variety. Castle Rock offers a genuinely different experience: redwood forest instead of open grassland, sandstone formations, a new trail to explore. It's 4.5 stars with 1,200 reviews (well-researched, your style), well-maintained with clear markers (your requirement), shaded (your preference), and moderate difficulty (your sweet spot). The extra 20 minutes of driving is a small price for a fresh, energizing experience.

## Gap Analysis
Significant gap. Your status quo bias is directly working against your own stated need for variety. You said "I've been doing the same trails lately" — that's a self-identified problem. Yet your instinct is to repeat the same trail for the 6th+ time. The cost of Fremont Older isn't just the boring trail — it's the missed opportunity to discover a new favorite spot. Castle Rock's profile (well-reviewed, well-maintained, moderate, shaded) actually meets ALL your criteria — the only "risk" is the 35 min drive.

## Confidence Score
MEDIUM — Your status quo bias strongly predicts Fremont Older, but the self-awareness about "same trails lately" introduces real uncertainty. If you read the Castle Rock AllTrails page and saw the photos, your research-driven decision pattern might override the bias.

## Follow-Up Questions
- What would it take for you to commit to trying a new trail — a friend suggesting it, a specific photo that excites you?
- Do you regret it when you default to the familiar trail on days when you could have explored?""",

    "hike-03": """## Your Likely Decision
You'd pick **Option A: Solo hike at Windy Hill**. You prefer morning starts (7 AM vs 9 AM), going at your own pace, and the meditative solo experience. You've been wanting to do this trail for weeks — there's built-up intention. The group hike starts later, takes longer (most of the morning + afternoon), and the 40 min drive is at the edge of your comfort zone. You'd feel like the group hike eats up too much of your Saturday.

## Ideal Decision
**Option B: Group hike at Purisima Creek** is the more balanced choice this week. You haven't seen these friends in a few weeks — social connection is important and easy to deprioritize when you're a remote worker who values solo time. Purisima Creek through redwoods is a genuinely great trail, and the post-hike lunch in Half Moon Bay adds to the experience. Windy Hill will be there next Saturday; the social invitation won't always be. You can still get your solo Windy Hill hike next weekend.

## Gap Analysis
Moderate gap. Your preference for solo, efficient, self-paced mornings is strong, and it's genuinely valuable for decompression. But as a remote worker, you're already spending most of your time alone. The group hike offers social connection, a trail you might not do on your own, and a day-trip experience (redwoods + Half Moon Bay lunch) that's different from your usual routine. The bias: you're optimizing for productivity and efficiency even in leisure time, when the "productive" choice is actually the less efficient one.

## Confidence Score
MEDIUM — Your solo preference and morning person pattern predict Option A, but the social factor ("haven't hung out in a few weeks") introduces real pull toward Option B. The deciding factor is probably how socially recharged or depleted you feel this week.

## Follow-Up Questions
- How are you feeling socially this week — recharged or isolated?
- Would you commit to doing the solo Windy Hill hike next Saturday if you go with the group this week?""",

    "hike-04": """## Your Likely Decision
You'd pick **Option B: Shoreline Park Trail**. With sore legs from Friday's workout, your practical side wins. You'd rationalize: "It's a recovery day, no point pushing through soreness and risking injury. The flat walk by the lake sounds actually perfect. Plus I can grab coffee at Red Rock after." You'd enjoy the bird watching and photography opportunity.

## Ideal Decision
**Option B: Shoreline Park Trail** is the ideal choice. Hiking 8.5 miles with 1,800 ft elevation gain on sore legs isn't smart training — it increases injury risk and won't be enjoyable. A recovery walk lets your muscles heal while still getting outdoors. The Shoreline loop gives you nature, camera time, and a gentle start to Sunday. Followed by coffee at Red Rock, this is the textbook "listen to your body" day.

## Gap Analysis
No gap. Your instinct to take the easy option when your body is sore is the right call. This is one case where the "comfort" choice IS the productive choice. Pushing through soreness for a strenuous hike isn't discipline — it's ignoring recovery needs. You can do Black Mountain next weekend with fresh legs and actually enjoy it.

## Confidence Score
HIGH — The physical constraint (sore legs) overrides your normal quality-seeking bias. When there's a concrete physical reason, you're practical about adjusting plans.

## Follow-Up Questions
- How bad is the soreness — mild DOMS or genuinely limiting?
- Would you feel guilty about choosing the easy walk, or is it genuinely a recovery-positive mindset?""",
}


def seed_data():
    """Seed vector store and persona."""
    print("\n  Seeding lifestyle preference data points...")
    store = VectorStore()
    chunks = []
    for i, dp in enumerate(DATA_POINTS):
        chunks.append(Chunk(
            text=dp,
            metadata={
                "source": "self_reported",
                "conversation_id": f"lifestyle_seed_{i}",
                "title": "Lifestyle preference",
                "timestamp": "2026-03-21T00:00:00+00:00",
                "msg_timestamp": "2026-03-21T00:00:00+00:00",
                "role": "user",
                "type": "data_point",
            }
        ))
    count = store.ingest(chunks)
    print(f"  Seeded {count} data points")

    print("  Setting up persona...")
    profile = PersonaProfile(
        communication_style={
            "tone": "casual and direct",
            "formality": "casual",
            "vocabulary_level": "technical",
            "sentence_patterns": ["short sentences", "asks follow-ups"],
            "common_phrases": ["makes sense", "let me think about it"],
        },
        knowledge_domains=["software engineering", "AI/ML", "coffee", "hiking"],
        decision_patterns=[
            "researches thoroughly before deciding (Google Maps, Yelp, AllTrails)",
            "prioritizes quality over convenience when difference is noticeable",
            "sticks with familiar places over trying new ones",
            "values ratings and reviews heavily",
            "prefers calm, quiet environments",
        ],
        values_and_priorities=[
            "quality over quantity",
            "supporting local businesses",
            "calm and quiet environments",
            "nature and outdoor time for balance",
            "morning routines",
        ],
        interests=["specialty coffee", "hiking", "photography", "remote work", "minimalist design"],
        cognitive_biases=[
            "status quo bias — gravitates toward familiar places and trails",
            "quality bias — overweights perceived quality signals",
            "anchoring to online reviews and ratings",
            "loss aversion — avoids unknown situations with uncertain outcomes",
        ],
        risk_tolerance="moderate",
        time_preference="balanced",
        system_prompt="""You are a remote software engineer based in Sunnyvale, CA. You love specialty
coffee (cortados, single-origin), prefer calm work-friendly cafes, and are vegetarian. You enjoy
moderate hikes on weekends within 45 min drive. You're a morning person, prefer familiar spots
but value quality, and always research places before going. You tend to stick with what you know
unless something new is clearly better.""",
    )
    profile.save()
    print("  Persona saved\n")


def run_test():
    """Run all lifestyle decisions and display results."""
    print("#" * 70)
    print("  AI TWIN: LIFESTYLE DECISION TEST")
    print("  How does Sudhira pick coffee shops and hikes?")
    print("#" * 70)

    seed_data()

    parser = TwinEngine.__new__(TwinEngine)

    for idx, scenario in enumerate(DECISION_SCENARIOS, 1):
        sid = scenario["id"]
        raw = DECISION_RESPONSES.get(sid)
        if not raw:
            continue

        result = parser._parse_decision_response(raw)

        print(f"\n{'=' * 70}")
        print(f"  {idx}. [{scenario['category'].upper()}] {scenario['title']}")
        print(f"{'=' * 70}")

        # Twin's pick
        pick_line = result.your_decision.split("\n")[0].strip()
        print(f"\n  TWIN PICKS: {pick_line}")

        # Productive pick
        ideal_line = result.ideal_decision.split("\n")[0].strip()
        print(f"  PRODUCTIVE: {ideal_line}")

        # Match?
        gap = result.reasoning_gap.lower()
        if any(w in gap for w in ["no gap", "no significant gap", "minimal gap"]):
            match = "SAME"
        elif "small gap" in gap or "minor gap" in gap:
            match = "CLOSE"
        elif "moderate gap" in gap:
            match = "DIFFERENT"
        else:
            match = "VERY DIFFERENT"
        print(f"  MATCH: {match}")

        # Why
        reasoning = result.reasoning_gap
        if len(reasoning) > 250:
            reasoning = reasoning[:250] + "..."
        print(f"\n  WHY: {reasoning}")

        # Confidence
        conf = "HIGH" if "HIGH" in result.confidence_score.upper() else "MEDIUM" if "MEDIUM" in result.confidence_score.upper() else "LOW"
        print(f"\n  CONFIDENCE: {conf}")

        # Follow-ups
        if result.follow_up_questions:
            print(f"\n  FOLLOW-UP QUESTIONS:")
            for q in result.follow_up_questions:
                print(f"    - {q}")

    # Summary table
    print(f"\n\n{'#' * 70}")
    print(f"  SUMMARY")
    print(f"{'#' * 70}")
    print(f"\n  {'#':<4} {'Scenario':<45} {'Twin Picks':<20} {'Match'}")
    print(f"  {'─'*4} {'─'*45} {'─'*20} {'─'*15}")

    for idx, scenario in enumerate(DECISION_SCENARIOS, 1):
        sid = scenario["id"]
        raw = DECISION_RESPONSES.get(sid)
        if not raw:
            continue
        result = parser._parse_decision_response(raw)

        # Extract short pick
        pick = result.your_decision.split("**")[1] if "**" in result.your_decision else "?"
        pick = pick.split("*")[0].strip(": ")
        if len(pick) > 18:
            pick = pick[:18] + ".."

        gap = result.reasoning_gap.lower()
        if any(w in gap for w in ["no gap", "no significant gap", "minimal gap"]):
            match = "SAME"
        elif "small gap" in gap or "minor gap" in gap:
            match = "CLOSE"
        elif "moderate gap" in gap:
            match = "DIFFERENT"
        else:
            match = "VERY DIFFERENT"

        print(f"  {idx:<4} {scenario['title']:<45} {pick:<20} {match}")

    print(f"\n{'#' * 70}\n")


if __name__ == "__main__":
    run_test()
