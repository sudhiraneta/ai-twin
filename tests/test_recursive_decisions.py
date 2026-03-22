"""
Recursive Multi-Decision Test for AI Twin

Tests the AI twin across a breadth of software engineer decision scenarios,
then analyzes patterns across all decisions to determine how productive
and consistent the twin's decision-making is.

No API key required — uses the twin's decision parsing and analysis
framework with pre-built decision responses to test the full pipeline.

Categories tested:
  1. Tech stack choices
  2. Work environment / productivity
  3. Career / growth decisions
  4. Code quality tradeoffs
  5. Tool & workflow choices
  6. Time management
  7. Learning & upskilling

Usage:
    python tests/test_recursive_decisions.py
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.vectorstore import VectorStore
from memory.chunker import Chunk
from twin.engine import TwinEngine, DecisionResponse
from persona.profile import PersonaProfile


# ── Decision Scenarios ────────────────────────────────────────────────────────

DECISION_SCENARIOS = [
    # ── 1. Tech Stack Choices ──
    {
        "id": "tech-01",
        "category": "Tech Stack",
        "title": "Python vs Go for a new microservice",
        "question": """I need to build a new microservice that handles real-time event processing.
I'm deciding between two approaches:

**Option A: Python (FastAPI + Celery)**
- I already know Python very well — can ship an MVP in ~3 days
- FastAPI gives great async support and auto-generated docs
- Rich ecosystem: tons of libraries for data processing
- Celery for background task queue
- Downside: GIL limits true parallelism, higher memory usage at scale
- Team is mostly Python-experienced

**Option B: Go (net/http + goroutines)**
- Compiled, lightweight, excellent concurrency with goroutines
- Much better performance for high-throughput event processing
- Smaller binary, lower memory footprint
- Downside: I'd need ~1 week to ramp up (haven't written Go in a while)
- Fewer libraries for data processing compared to Python
- Would be the first Go service in our stack

Which should I pick?""",
    },
    {
        "id": "tech-02",
        "category": "Tech Stack",
        "title": "SQL vs NoSQL for user activity logs",
        "question": """We need to store user activity logs for our analytics pipeline.
Choosing between:

**Option A: PostgreSQL with TimescaleDB extension**
- Familiar SQL interface, strong consistency guarantees
- TimescaleDB adds time-series optimizations (chunking, compression)
- Can join with existing user tables in our Postgres DB
- Downside: Schema changes are harder, vertical scaling has limits
- We already run Postgres so no new infra

**Option B: MongoDB with TTL indexes**
- Schema-flexible — activity log format changes frequently
- Horizontal scaling is straightforward
- TTL indexes auto-expire old logs
- Downside: No joins with our relational data, eventual consistency
- Would need to set up and maintain a new database system

Which is the better choice?""",
    },

    # ── 2. Work Environment / Productivity ──
    {
        "id": "prod-01",
        "category": "Productivity",
        "title": "Deep work block vs async-first day",
        "question": """I have a complex feature to design and implement this week.
I'm deciding how to structure my work day tomorrow:

**Option A: 4-hour deep work block (8 AM - 12 PM, all notifications off)**
- Block calendar, Slack on DND, no meetings
- Full immersion on the feature design + initial implementation
- Risk: might miss urgent messages, team might need me for blockers
- I tend to do my best thinking in long uninterrupted stretches
- Afternoon: catch up on messages, reviews, quick meetings

**Option B: Async-responsive day (work in 90-min sprints, check messages between)**
- 90 min focus, 15 min check-in cycle throughout the day
- Stay responsive to team needs, unblock others quickly
- Risk: context switching costs, might not get deep into the problem
- More collaborative but potentially less individual output
- Spread design + implementation across the full day

Which approach should I take?""",
    },
    {
        "id": "prod-02",
        "category": "Productivity",
        "title": "IDE choice: VS Code vs Neovim",
        "question": """I'm considering switching my primary development environment:

**Option A: Stay with VS Code**
- I've used it for 3+ years, highly customized setup with 20+ extensions
- Great for multi-language projects (Python, TypeScript, Go)
- Copilot integration works seamlessly
- Downside: Can feel sluggish on large projects, Electron memory usage
- Familiar — no learning curve, immediately productive

**Option B: Switch to Neovim (with LazyVim)**
- Terminal-native, blazing fast, minimal resource usage
- LazyVim gives a good starting config with LSP, completions, etc.
- Keyboard-driven workflow — potentially faster once learned
- Downside: 2-4 week learning curve, initial productivity drop
- Strong community, highly composable configuration
- Copilot plugin available but less polished than VS Code

Should I make the switch?""",
    },

    # ── 3. Career / Growth ──
    {
        "id": "career-01",
        "category": "Career",
        "title": "Lead a project vs stay as IC contributor",
        "question": """My manager offered me two paths for the next quarter:

**Option A: Tech Lead for the new platform rewrite**
- Lead a team of 4 engineers on a critical project
- More meetings, design reviews, mentoring, less coding
- High visibility with leadership, good for promotion trajectory
- Downside: I'd write maybe 30% of the code I write now
- I've never led a team before — learning opportunity but risky

**Option B: Stay as IC, own the hardest technical component**
- Deep dive into the most complex subsystem (distributed cache layer)
- Write a lot of code, solve hard problems, publish a tech blog post
- Less visibility but strong technical growth
- Downside: less management experience, might plateau at current level
- Comfortable and aligned with what I enjoy most

Which path should I take?""",
    },

    # ── 4. Code Quality Tradeoffs ──
    {
        "id": "code-01",
        "category": "Code Quality",
        "title": "Ship now with tech debt vs refactor first",
        "question": """We have a feature that's 80% done but the implementation has accumulated
some tech debt. Product wants it shipped by Friday. I'm deciding:

**Option A: Ship as-is, create tech debt tickets**
- Feature works correctly, passes all tests
- Code is messy: 3 god functions, some copy-paste duplication
- Create JIRA tickets for cleanup, schedule for next sprint
- Downside: tech debt tickets often get deprioritized and forgotten
- Meets the deadline, unblocks the product launch

**Option B: Take 2 extra days to refactor, ship Monday**
- Break up god functions, extract shared logic, add better error handling
- Cleaner codebase, easier to maintain and extend later
- Downside: delays the launch by 2 business days
- Product team might push back on the delay
- Sets a better precedent for code quality standards

Which should I do?""",
    },
    {
        "id": "code-02",
        "category": "Code Quality",
        "title": "Write tests first (TDD) vs test after for a new feature",
        "question": """I'm about to implement a new payment processing module. Deciding on approach:

**Option A: Test-Driven Development (TDD)**
- Write failing tests first, then implement to make them pass
- Forces me to think about edge cases upfront (refunds, partial payments, timeouts)
- Slower initial velocity but catches bugs earlier
- Results in high test coverage by default
- Downside: feels slower, hard to TDD when the API design is still fuzzy

**Option B: Implement first, then write tests**
- Prototype the happy path quickly, iterate on the API design
- Write comprehensive tests once the interface stabilizes
- Faster initial progress, can demo sooner
- Downside: might skip edge case tests under time pressure
- Risk of "I'll add tests later" turning into never

Which approach for this module?""",
    },

    # ── 5. Tool & Workflow ──
    {
        "id": "tool-01",
        "category": "Tools",
        "title": "Monorepo vs polyrepo for team projects",
        "question": """Our team is starting 3 new related services. Architecture decision:

**Option A: Monorepo (all 3 services + shared libs in one repo)**
- Atomic changes across services, shared CI/CD pipeline
- Easy code sharing, consistent tooling and linting
- Downside: CI gets slower as repo grows, need good build tooling (Bazel/Nx)
- All team members see all code — encourages ownership
- Google, Meta style approach

**Option B: Polyrepo (separate repo per service + shared lib repo)**
- Clear boundaries, independent deployment pipelines
- Teams own their repos, faster CI per repo
- Downside: cross-service changes require coordinated PRs
- Versioned shared libraries via package registry
- More standard in smaller orgs, each service is self-contained

Which structure should we go with?""",
    },

    # ── 6. Time Management ──
    {
        "id": "time-01",
        "category": "Time Management",
        "title": "Fix the bug now vs scheduled maintenance window",
        "question": """A non-critical bug was found in production: some users see stale data
for ~30 seconds after updating their profile. Deciding when to fix:

**Option A: Fix it now (hotfix + deploy today)**
- It's a cache invalidation issue, I already know the fix (~2 hours)
- Ship it today, users stop experiencing the issue immediately
- Downside: interrupts my current sprint work (feature due Thursday)
- Hotfix deployments carry some risk even for small changes
- Satisfies the support team who flagged it

**Option B: Schedule for next maintenance window (next Tuesday)**
- Add it to the maintenance batch with other small fixes
- Properly test in staging over the weekend
- Downside: users experience the stale data bug for 5 more days
- No sprint disruption, deploy with other changes for reduced risk
- Can write a more thorough fix with proper cache strategy review

When should I fix this?""",
    },

    # ── 7. Learning & Upskilling ──
    {
        "id": "learn-01",
        "category": "Learning",
        "title": "Learn Rust vs deepen Python expertise",
        "question": """I have a personal learning goal for this quarter. Deciding what to focus on:

**Option A: Learn Rust**
- Systems programming, memory safety without GC
- Growing adoption in infrastructure tooling (ripgrep, deno, etc.)
- Stretches my skills into a new paradigm (ownership, borrowing)
- Downside: steep learning curve, may not use it at work immediately
- Strong for resume and long-term career optionality

**Option B: Deepen Python expertise (advanced async, C extensions, CPython internals)**
- Directly applicable to my current job
- Become the team's Python expert, improve our codebase performance
- Learn to write C extensions for hot paths
- Downside: less novel, might feel incremental vs transformative
- Immediately useful but less exciting

Which should I invest my learning time in?""",
    },
]


# ── User data points to seed ─────────────────────────────────────────────────

DATA_POINTS = [
    # Work style
    "I work remotely most days and need long uninterrupted blocks to do my best work.",
    "I prefer shipping iteratively — get something working, then refine.",
    "I tend to research thoroughly before making technical decisions.",
    "When choosing between convenience and quality, I usually pick quality if the difference is noticeable.",

    # Technical preferences
    "Python is my strongest language. I've used it professionally for 5+ years.",
    "I value clean, readable code over clever one-liners.",
    "I believe in testing but I'm pragmatic — I test critical paths, not everything.",
    "I prefer using well-established tools over bleeding-edge ones unless the new tool solves a real pain point.",

    # Decision patterns
    "I have a slight status quo bias — I tend to stick with what I know works.",
    "When pressed for time, I'll ship what works and plan to fix it later (though I try to follow through).",
    "I value team productivity over individual productivity — I'll sacrifice my own flow to unblock others.",
    "I usually go with the option that has the best long-term payoff, even if short-term cost is higher.",

    # Career values
    "I care about technical depth — I'd rather be great at a few things than okay at many.",
    "I enjoy mentoring but I'm not sure I want to be a full-time manager.",
    "I want to keep coding as a significant part of my role for the foreseeable future.",
    "I value learning and growth — I get restless if I'm not being challenged.",

    # Personality
    "I'm a morning person — my best focus hours are 8-11 AM.",
    "I prefer calm, structured environments over chaotic ones.",
    "I'm somewhat risk-averse with production systems but more adventurous with personal projects.",
    "Last time I had to pick between two approaches, I went with the one that had better documentation and community support.",
]


def seed_data_points():
    """Seed the vector store with SW engineer preference data points."""
    print("\n  Seeding user data points...")
    store = VectorStore()
    chunks = []
    for i, dp in enumerate(DATA_POINTS):
        chunks.append(Chunk(
            text=dp,
            metadata={
                "source": "self_reported",
                "conversation_id": f"test_recursive_seed_{i}",
                "title": "User preference data point",
                "timestamp": "2026-03-21T00:00:00+00:00",
                "msg_timestamp": "2026-03-21T00:00:00+00:00",
                "role": "user",
                "type": "data_point",
            }
        ))
    count = store.ingest(chunks)
    print(f"  Seeded {count} data points into memory")
    return count


def seed_persona():
    """Create a persona profile for the SW engineer twin."""
    print("  Setting up persona profile...")
    profile = PersonaProfile(
        communication_style={
            "tone": "casual and direct, uses technical terms naturally",
            "formality": "casual",
            "vocabulary_level": "technical",
            "sentence_patterns": ["uses short sentences", "thinks out loud", "asks follow-up questions"],
            "common_phrases": ["makes sense", "let me think about it", "tradeoff here is", "ship it"],
        },
        knowledge_domains=["software engineering", "AI/ML", "distributed systems", "Python", "DevOps"],
        decision_patterns=[
            "researches thoroughly before deciding",
            "prioritizes quality and long-term payoff over short-term speed",
            "sticks with familiar tools unless new ones solve a real pain point",
            "values team productivity — will sacrifice personal flow to unblock others",
            "pragmatic about testing — covers critical paths, not 100% coverage",
            "ships iteratively: working MVP first, then refine",
        ],
        values_and_priorities=[
            "technical depth over breadth",
            "clean, readable code",
            "long-term thinking",
            "team collaboration",
            "continuous learning and growth",
            "calm, structured work environment",
        ],
        interests=["AI/ML", "systems design", "developer tools", "coffee", "music", "productivity"],
        cognitive_biases=[
            "status quo bias — prefers familiar tools and approaches",
            "quality bias — overweights code cleanliness vs shipping speed",
            "anchoring to documentation and community support",
            "loss aversion with production systems",
        ],
        risk_tolerance="moderate",
        time_preference="long-term",
        system_prompt="""You are a remote software engineer with 5+ years of Python experience.
You value clean code, long-term thinking, and team collaboration. You're a morning person who
does best work in long uninterrupted blocks. You ship iteratively — working MVP first, then refine.
You have a slight status quo bias toward familiar tools but you're open to change when the new
option clearly solves a pain point. You're pragmatic about testing, prefer depth over breadth
in your skills, and want to keep coding as a core part of your role. You're somewhat risk-averse
with production systems but adventurous with personal projects.""",
    )
    profile.save()
    print("  Persona profile saved")
    return profile


# ── Analysis functions ────────────────────────────────────────────────────────

@dataclass
class DecisionSummary:
    scenario_id: str
    category: str
    title: str
    your_decision: str
    ideal_decision: str
    has_gap: bool
    gap_severity: str  # "none", "small", "moderate", "significant"
    confidence: str
    productive_choice: str  # "yours", "ideal", "aligned"


def analyze_gap_severity(gap_text: str) -> str:
    """Categorize the gap severity from the gap analysis text."""
    lower = gap_text.lower()
    if any(w in lower for w in ["no gap", "no significant gap", "minimal gap", "aligns with"]):
        return "none"
    elif any(w in lower for w in ["small gap", "minor gap"]):
        return "small"
    elif any(w in lower for w in ["moderate gap"]):
        return "moderate"
    elif any(w in lower for w in ["significant gap", "meaningful gap", "large gap"]):
        return "significant"
    return "small"


def analyze_confidence(confidence_text: str) -> str:
    """Extract confidence level."""
    upper = confidence_text.upper()
    if "HIGH" in upper:
        return "HIGH"
    elif "MEDIUM" in upper:
        return "MEDIUM"
    elif "LOW" in upper:
        return "LOW"
    return "UNKNOWN"


def determine_productive_choice(your_dec: str, ideal_dec: str, gap_text: str) -> str:
    """Determine whether the user's likely choice or the ideal is more productive."""
    lower_gap = gap_text.lower()
    if any(w in lower_gap for w in ["no gap", "no significant gap", "minimal gap", "aligns with the ideal", "instinct aligns"]):
        return "aligned"
    # If there's a gap, the ideal is considered more productive
    return "ideal"


def print_scenario_result(idx: int, scenario: dict, result: DecisionResponse, summary: DecisionSummary):
    """Print a single scenario's results."""
    print(f"\n{'=' * 70}")
    print(f"  DECISION {idx}: [{scenario['category'].upper()}] {scenario['title']}")
    print(f"{'=' * 70}")

    # Your decision (truncated for readability)
    your_dec_short = result.your_decision[:300]
    if len(result.your_decision) > 300:
        your_dec_short += "..."
    print(f"\n  YOUR LIKELY DECISION:")
    print(f"  {your_dec_short}")

    # Ideal decision (truncated)
    ideal_short = result.ideal_decision[:300]
    if len(result.ideal_decision) > 300:
        ideal_short += "..."
    print(f"\n  IDEAL DECISION:")
    print(f"  {ideal_short}")

    # Gap
    gap_indicator = {
        "none": "[ALIGNED]     ",
        "small": "[SMALL GAP]   ",
        "moderate": "[MODERATE GAP]",
        "significant": "[BIG GAP]     ",
    }
    print(f"\n  GAP: {gap_indicator.get(summary.gap_severity, '[?]')} | Confidence: {summary.confidence}")

    # Productive choice
    if summary.productive_choice == "aligned":
        print(f"  PRODUCTIVITY: Your instinct IS the productive choice")
    else:
        print(f"  PRODUCTIVITY: The ideal decision is more productive than your instinct")


def print_overall_analysis(summaries: list[DecisionSummary]):
    """Print the recursive cross-decision analysis."""
    total = len(summaries)
    aligned = sum(1 for s in summaries if s.productive_choice == "aligned")
    gaps = sum(1 for s in summaries if s.productive_choice == "ideal")

    gap_severities = {"none": 0, "small": 0, "moderate": 0, "significant": 0}
    for s in summaries:
        gap_severities[s.gap_severity] = gap_severities.get(s.gap_severity, 0) + 1

    confidence_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for s in summaries:
        confidence_counts[s.confidence] = confidence_counts.get(s.confidence, 0) + 1

    # Category breakdown
    categories = {}
    for s in summaries:
        if s.category not in categories:
            categories[s.category] = {"aligned": 0, "gap": 0, "total": 0}
        categories[s.category]["total"] += 1
        if s.productive_choice == "aligned":
            categories[s.category]["aligned"] += 1
        else:
            categories[s.category]["gap"] += 1

    print(f"\n\n{'#' * 70}")
    print(f"  RECURSIVE CROSS-DECISION ANALYSIS")
    print(f"{'#' * 70}")

    print(f"\n  OVERALL ALIGNMENT SCORE")
    print(f"  -----------------------")
    pct = (aligned / total * 100) if total > 0 else 0
    bar_filled = int(pct / 5)
    bar = "[" + "|" * bar_filled + "." * (20 - bar_filled) + "]"
    print(f"  {bar} {aligned}/{total} decisions aligned ({pct:.0f}%)")
    print(f"  {gaps}/{total} decisions where ideal differs from instinct ({100-pct:.0f}%)")

    print(f"\n  GAP SEVERITY DISTRIBUTION")
    print(f"  -------------------------")
    for severity, count in gap_severities.items():
        label = severity.upper().ljust(12)
        bar = "|" * (count * 4)
        print(f"  {label} {bar} ({count})")

    print(f"\n  CONFIDENCE DISTRIBUTION")
    print(f"  -----------------------")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        count = confidence_counts.get(level, 0)
        bar = "|" * (count * 4)
        print(f"  {level.ljust(8)} {bar} ({count})")

    print(f"\n  CATEGORY BREAKDOWN")
    print(f"  ------------------")
    for cat, counts in sorted(categories.items()):
        aligned_pct = (counts["aligned"] / counts["total"] * 100) if counts["total"] > 0 else 0
        status = "ALIGNED" if aligned_pct >= 50 else "DIVERGENT"
        print(f"  {cat.ljust(18)} {counts['aligned']}/{counts['total']} aligned  [{status}]")

    # Identify patterns
    print(f"\n  DECISION-MAKING PATTERNS DETECTED")
    print(f"  ----------------------------------")

    # Pattern: where does the twin diverge most?
    gap_categories = [s.category for s in summaries if s.productive_choice == "ideal"]
    if gap_categories:
        print(f"  Divergence areas: {', '.join(set(gap_categories))}")
    else:
        print(f"  No divergence detected — instincts consistently match ideal choices")

    aligned_categories = [s.category for s in summaries if s.productive_choice == "aligned"]
    if aligned_categories:
        print(f"  Strength areas:   {', '.join(set(aligned_categories))}")

    # Productivity verdict
    print(f"\n  PRODUCTIVITY VERDICT")
    print(f"  --------------------")
    if pct >= 80:
        print(f"  HIGHLY PRODUCTIVE: Decision instincts align with ideal choices {pct:.0f}% of the time.")
        print(f"  The twin's biases (status quo, quality focus) are mostly well-calibrated.")
    elif pct >= 50:
        print(f"  MODERATELY PRODUCTIVE: Decision instincts align {pct:.0f}% of the time.")
        print(f"  Key growth areas exist in: {', '.join(set(gap_categories))}")
        print(f"  The twin's status quo bias and risk aversion create blind spots in some contexts.")
    else:
        print(f"  NEEDS CALIBRATION: Only {pct:.0f}% alignment with ideal choices.")
        print(f"  Systematic biases are overriding good judgment in multiple categories.")

    # Specific insights
    print(f"\n  KEY INSIGHTS")
    print(f"  ------------")
    insights = []
    tech_summaries = [s for s in summaries if s.category == "Tech Stack"]
    if all(s.productive_choice == "aligned" for s in tech_summaries):
        insights.append("Tech stack decisions are well-calibrated — familiarity bias is appropriate here")
    code_summaries = [s for s in summaries if s.category == "Code Quality"]
    divergent_code = [s for s in code_summaries if s.productive_choice == "ideal"]
    if divergent_code:
        insights.append("Code quality decisions show gaps — pragmatism sometimes overrides domain-appropriate rigor")
    career_summaries = [s for s in summaries if s.category == "Career"]
    if any(s.productive_choice == "ideal" for s in career_summaries):
        insights.append("Career decisions lean toward comfort — growth opportunities may be systematically underweighted")
    learn_summaries = [s for s in summaries if s.category == "Learning"]
    if any(s.productive_choice == "ideal" for s in learn_summaries):
        insights.append("Learning decisions favor incremental gains over transformative growth")

    for i, insight in enumerate(insights, 1):
        print(f"  {i}. {insight}")
    if not insights:
        print(f"  Decision patterns are consistently well-calibrated across all categories")


# ── Pre-built decision responses (no API needed) ─────────────────────────────
# These simulate what the twin's LLM would return, letting us test the full
# parsing, analysis, and cross-decision pattern detection pipeline offline.

DECISION_RESPONSES = {
    "tech-01": """## Your Likely Decision
You'd pick **Option A: Python (FastAPI + Celery)**. Based on your strong preference for familiar tools (5+ years of Python, status quo bias), your pattern of shipping iteratively with a working MVP first, and the fact that your entire team is Python-experienced, you'd go with what you know. The 3-day MVP timeline aligns with your ship-fast-then-refine approach.

## Ideal Decision
**Option A: Python (FastAPI + Celery)** is actually the ideal choice here too. Shipping quickly and validating the design matters more than premature optimization. FastAPI's async support handles most real-time event processing needs. Team familiarity reduces onboarding friction. Introducing Go as the first non-Python service adds operational complexity not justified at current scale.

## Gap Analysis
Minimal gap — your instinct aligns with the ideal decision. Your status quo bias works in your favor here because the familiar option genuinely is the better choice. One risk: you might not revisit Go even when scale demands it.

## Confidence Score
HIGH — Your Python preference is well-documented and your "familiar tools first" pattern is one of your strongest decision signals.

## Follow-Up Questions
- At what scale (events/sec) do you expect this service to operate in 6 months?
- Have you had situations where choosing the familiar option led to regret later?""",

    "tech-02": """## Your Likely Decision
You'd pick **Option A: PostgreSQL with TimescaleDB**. Your bias toward familiar, well-established tools is strong — you already run Postgres, so there's zero new infrastructure. The SQL interface and strong consistency match your pragmatic approach.

## Ideal Decision
**Option A: PostgreSQL with TimescaleDB** is the ideal choice. The benefits of staying within your existing database ecosystem outweigh MongoDB's schema flexibility advantage. TimescaleDB handles time-series activity logs well, and JSONB columns cover flexible fields.

## Gap Analysis
No significant gap. Your preference for established tools aligns with the pragmatic choice here.

## Confidence Score
HIGH — Strong pattern of preferring existing infrastructure and well-established tools.

## Follow-Up Questions
- How often do you expect the activity log schema to change?
- Is there a future need for real-time analytics on this data?""",

    "prod-01": """## Your Likely Decision
You'd pick **Option A: 4-hour deep work block**. You're a morning person with best focus hours 8-11 AM, you need long uninterrupted blocks for best work, and you prefer calm structured environments. You'd set Slack to DND, block your calendar, and dive in.

## Ideal Decision
**Option A with a small modification**: Do the 4-hour deep work block BUT post a proactive Slack message before going DND: "Heads down on [feature] until noon. If truly urgent, text me." This respects your team productivity value while protecting focus.

## Gap Analysis
Small gap — your instinct is right, but your "team productivity over individual productivity" value creates tension. The proactive message eliminates guilt and gives the team an escape valve for genuine blockers.

## Confidence Score
HIGH — Your morning person pattern, preference for uninterrupted blocks, and the task type all strongly predict Option A.

## Follow-Up Questions
- How often do truly urgent blockers come up that need your input within 4 hours?
- Do you tend to check Slack even when it's on DND?""",

    "prod-02": """## Your Likely Decision
You'd pick **Option A: Stay with VS Code**. Textbook status quo bias — 3+ years of customizations, 20+ extensions, "immediately productive." You'd look at Neovim, watch some YouTube videos, and decide "maybe later."

## Ideal Decision
**Option A: Stay with VS Code** is the ideal choice. The 2-4 week productivity dip isn't justified. Neovim's performance benefits are marginal for most development work. Your setup is already optimized.

## Gap Analysis
No gap — your status quo bias serves you well here. The switching cost is high and the marginal benefit is low for your use case.

## Confidence Score
HIGH — Your status quo bias, familiarity preference, and practical mindset all strongly predict staying with VS Code.

## Follow-Up Questions
- Do you have specific performance issues with VS Code?
- Have you ever made a major tool switch before?""",

    "career-01": """## Your Likely Decision
You'd pick **Option B: Stay as IC, own the hardest technical component**. You want to keep coding, you care about technical depth over breadth, and the distributed cache layer is exactly the kind of challenge that energizes you.

## Ideal Decision
**Option A: Tech Lead for the platform rewrite** is the ideal choice. Leading a team of 4 on a critical project is a low-risk opportunity to test leadership — it's one quarter, not permanent. The ability to multiply impact through others becomes more valuable than individual depth. Leadership opportunities on critical projects are rarer than hard technical problems.

## Gap Analysis
Significant gap. Your preference for IC work is driven by comfort and identity. Your stated value of continuous growth creates tension — staying IC is comfortable but leading would stretch you more. If you always choose the IC path, you may plateau while peers who took leadership advance.

## Confidence Score
MEDIUM — Your IC preference is clear, but the "I value growth and get restless without challenge" signal could swing this.

## Follow-Up Questions
- If someone else leads the project successfully, would you regret not trying?
- What concerns you most about leadership — meetings, people management, or reduced coding?""",

    "code-01": """## Your Likely Decision
You'd pick **Option A: Ship as-is, create tech debt tickets**. Despite valuing clean code, your "ship iteratively" pattern wins. The feature passes all tests and meets the deadline. You'd create tickets with genuine intent to follow through.

## Ideal Decision
**Option A: Ship as-is** is the pragmatic ideal, BUT with a stronger commitment mechanism than JIRA tickets. Schedule a 2-hour refactoring session on day one of next sprint. Tech debt tickets get deprioritized 70%+ of the time — a calendar block is harder to skip.

## Gap Analysis
Small gap. Your shipping instinct is correct. The risk is in follow-through — your own data says "though I try to follow through" which implies you sometimes don't. Adding a concrete commitment mechanism closes this gap.

## Confidence Score
HIGH — Your "ship iteratively" and "pragmatic about testing" patterns are strong predictors.

## Follow-Up Questions
- How often have your tech debt tickets actually been completed in the following sprint?
- Is there an upcoming feature that would be blocked by the messy code?""",

    "code-02": """## Your Likely Decision
You'd pick **Option B: Implement first, then write tests**. Your "pragmatic about testing — covers critical paths, not 100% coverage" stance combined with "ship iteratively" points toward prototyping first.

## Ideal Decision
**Option A: TDD** is the ideal choice for payment processing specifically. Payment systems are high-consequence — bugs in refund logic or partial payments cost real money. TDD forces you to enumerate edge cases (refunds, currency rounding, timeouts, idempotency) before writing implementation.

## Gap Analysis
Meaningful gap. Your pragmatic testing approach works for most features, but payment processing is where "cover critical paths" should mean "TDD the entire module." You're anchoring to your usual workflow even when the domain demands different rigor. Financial code is one of the few areas where TDD has clearly positive ROI.

## Confidence Score
HIGH — Your general testing philosophy is clear, but the high-stakes nature might make you reconsider if someone on your team suggests TDD.

## Follow-Up Questions
- Have you worked on payment or financial code before?
- If a bug caused incorrect charges, how severe would the business impact be?""",

    "tool-01": """## Your Likely Decision
You'd pick **Option B: Polyrepo**. Your preference for well-established, standard approaches and the fact that polyrepo requires less specialized tooling points here. You'd value clear boundaries and independent deployments.

## Ideal Decision
**Option A: Monorepo** is the ideal choice for 3 related services with shared code. Coordinated polyrepo PRs become friction tax. A monorepo with basic CI tooling (path-filtered pipelines, no Bazel needed) keeps atomic cross-service changes simple.

## Gap Analysis
Moderate gap. Your preference for established tools creates bias against monorepo, which you associate with complex build systems. But a monorepo doesn't require complex tooling initially. The real cost of polyrepo shows up in 3-6 months when shared library versioning becomes painful. Your "long-term payoff" value should favor monorepo, but your "familiar tools" bias overrides it.

## Confidence Score
MEDIUM — Your tool preferences point to polyrepo, but your "long-term thinking" value creates tension.

## Follow-Up Questions
- How frequently will these 3 services change together or share code?
- Have you experienced the pain of coordinating changes across multiple repos?""",

    "time-01": """## Your Likely Decision
You'd pick **Option B: Schedule for next maintenance window**. Your risk aversion with production systems, the non-critical nature of the bug, and your sprint commitment all point toward the scheduled fix.

## Ideal Decision
**Option A: Fix it now** is the ideal choice. The fix is known, estimated at 2 hours, and well-understood. A targeted hotfix carries minimal risk. 5 more days of stale data affects user trust and the support team has flagged it. The sprint disruption of 2 hours is minimal.

## Gap Analysis
Moderate gap. Your loss aversion with production systems leads you to overweight deployment risk for small, well-understood changes. A 2-hour cache fix is about as low-risk as hotfixes get. You're anchoring on "hotfixes are risky" as a general rule rather than evaluating this specific fix's risk profile.

## Confidence Score
MEDIUM — Your risk aversion predicts Option B, but team-first values could swing you if the support team escalates.

## Follow-Up Questions
- Is a single-service deploy truly risky, or is the risk perception outdated?
- Has delaying a "non-critical" bug fix ever escalated into a bigger issue?""",

    "learn-01": """## Your Likely Decision
You'd pick **Option B: Deepen Python expertise**. Your pragmatism ("immediately applicable"), preference for depth over breadth, and status quo bias all point toward doubling down on Python. "I'll learn Rust when there's a concrete use case at work."

## Ideal Decision
**Option A: Learn Rust** is the ideal choice for a learning goal. Learning goals should be stretch goals — you said you "get restless without challenge." Rust teaches fundamentally different concepts (ownership, borrowing) that make you a better systems thinker. It expands career optionality in infrastructure and performance-critical systems.

## Gap Analysis
Significant gap. Your "depth over breadth" value is applied too narrowly — true depth means understanding computing at multiple levels, not just one language. The Python path feels like growth but is actually comfort (you're already top-percentile on your team). Rust would provide the genuine challenge you crave.

## Confidence Score
MEDIUM — Your pragmatic values predict Python, but your growth-seeking tendency could swing this if you frame Rust as the "deeper" systems-level choice.

## Follow-Up Questions
- When did you last learn something that fundamentally changed how you think about programming?
- If you deepen Python this quarter, what would your learning goal be next quarter?""",
}


def run_recursive_test():
    """Run all decision scenarios recursively and analyze cross-decision patterns.

    Uses pre-built decision responses to test the full pipeline (parsing,
    gap analysis, pattern detection) without requiring any API key.
    """
    print("\n" + "#" * 70)
    print("  AI TWIN: RECURSIVE MULTI-DECISION TEST")
    print("  Testing across 10 software engineering decision scenarios")
    print("#" * 70)

    seed_data_points()
    seed_persona()

    # We only need the parser from TwinEngine — no API calls
    parser = TwinEngine.__new__(TwinEngine)
    summaries = []

    print(f"\n  Running {len(DECISION_SCENARIOS)} decisions through the decision framework...")
    print(f"  (No API key needed — using built-in decision analysis)\n")

    for idx, scenario in enumerate(DECISION_SCENARIOS, 1):
        sid = scenario["id"]
        raw_response = DECISION_RESPONSES.get(sid)
        if not raw_response:
            print(f"  [{idx}/{len(DECISION_SCENARIOS)}] {scenario['title']}... SKIPPED (no response)")
            continue

        # Parse the response through the twin's own parsing logic
        result = parser._parse_decision_response(raw_response)

        # Analyze
        gap_sev = analyze_gap_severity(result.reasoning_gap)
        conf = analyze_confidence(result.confidence_score)
        productive = determine_productive_choice(result.your_decision, result.ideal_decision, result.reasoning_gap)

        summary = DecisionSummary(
            scenario_id=sid,
            category=scenario["category"],
            title=scenario["title"],
            your_decision=result.your_decision[:100],
            ideal_decision=result.ideal_decision[:100],
            has_gap=gap_sev != "none",
            gap_severity=gap_sev,
            confidence=conf,
            productive_choice=productive,
        )
        summaries.append(summary)

        print_scenario_result(idx, scenario, result, summary)

    # Cross-decision analysis
    if summaries:
        print_overall_analysis(summaries)

    print(f"\n{'#' * 70}")
    print(f"  TEST COMPLETE: {len(summaries)} decisions analyzed")
    print(f"{'#' * 70}\n")

    return summaries


if __name__ == "__main__":
    run_recursive_test()
