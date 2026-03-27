"""
Skill Files — per-dimension markdown files that represent who you are.

Each skill file has two sections:
  1. **Traits** — what the twin knows about you (auto-extracted, editable)
  2. **Sources** — where to pull data for this dimension (retrieval orchestration)

The Sources section tells the twin which data sources and chunk types to query
when a question maps to this dimension. This is the workflow orchestration layer.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from config import PERSONA_DIR
from .dimensions import DIMENSIONS, PersonaDimension

SKILLS_DIR = PERSONA_DIR / "skills"

# ---------------------------------------------------------------------------
# Retrieval orchestration per dimension — SKILL FILE BLUEPRINT
#
# This is the single source of truth for how the twin answers questions.
# Each dimension defines a 3-tier retrieval strategy:
#
#   Tier 1 (CORE):  Skill traits (static) + Singularity DB entries + clustered
#                   Apple Notes — these are the user's own words and structured data.
#   Tier 2 (MEMORY): Imported LLM memory (ChatGPT/Claude/Gemini conversation
#                    history) — user_message and conversation_pair chunks that were
#                    classified into this dimension.
#   Tier 3 (SUPPLEMENTARY): Browser activity, weekly reviews, photos, and other
#                           observational data that adds behavioral evidence.
#
# Additionally:
#   - do:         what the LLM MUST do when answering questions in this dimension
#   - dont:       guardrails — what to avoid (edge cases, privacy, bias)
#   - habits:     known user habits/patterns to look for in data
#   - preferences: known user preferences to prioritize
#   - data_goals: what data the twin should try to learn over time
#   - edge_cases: tricky scenarios and how to handle them
#   - sql_tables: structured metric tables to query for quantitative answers
# ---------------------------------------------------------------------------

DIMENSION_SOURCES: dict[str, dict] = {
    "code": {
        "description": "Programming languages, frameworks, coding patterns, engineering practices",

        # Tier 1: Core — Singularity career entries + clustered Apple Notes
        "primary_types": ["singularity_entry", "pillar_journal", "note", "data_point"],
        "primary_queries": [
            "python rust code programming framework deploy architecture",
            "engineering backend infrastructure API design system",
            "debugging refactoring code review pull request git",
        ],

        # Tier 2: LLM Memory — past conversations about coding
        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "code programming help debug fix implement build",
            "python javascript react fastapi technical discussion",
        ],

        # Tier 3: Supplementary — browser visits to coding sites
        "secondary_types": ["browser_domain", "browser_daily", "task"],
        "secondary_queries": ["github stackoverflow coding sites developer tools"],

        # SQL tables for quantitative queries
        "sql_tables": ["entries"],
        "sql_filters": {"pillar": "MIND"},

        "do": [
            "Reference specific languages, frameworks, and tools mentioned in Singularity entries",
            "Cross-reference Apple Notes for coding project plans or architecture decisions",
            "Check LLM conversation history for what the user has asked AI to help build",
            "Mention skill level (beginner/intermediate/expert) only when evidence supports it",
            "Include recent projects, repos, or side projects from journal entries",
            "Cite the data source: [Singularity/MIND], [Apple Note], [ChatGPT conversation]",
            "Look for cluster_label matches like 'code_architecture', 'python_projects'",
        ],
        "dont": [
            "Guess programming skills not supported by any data source",
            "Assume proficiency from a single mention — look for repeated evidence",
            "Conflate professional work with side projects unless data links them",
            "List technologies from AI assistant responses as user skills — only user messages count",
            "Infer the user taught themselves something just because they asked an AI about it",
        ],
        "habits": [
            "Coding language preferences and frequency of use",
            "Preferred IDE, tools, and development workflow",
            "Code review style and collaboration patterns",
            "Common debugging approaches from conversation history",
        ],
        "preferences": [
            "Language and framework choices for new projects",
            "Architecture style (monolith vs microservices, sync vs async)",
            "Testing philosophy and CI/CD preferences",
        ],
        "data_goals": [
            "Build a complete tech stack profile from all sources",
            "Track skill evolution over time via Singularity entry timestamps",
            "Identify gaps between what user knows and what they're learning",
        ],
        "edge_cases": [
            "User asking AI to write code ≠ user knowing that language — check if they edited/used it",
            "Browser visits to docs don't confirm proficiency — could be learning",
            "Old Singularity entries may reference abandoned technologies",
        ],
    },

    "professional": {
        "description": "Career, current role, company, leadership, influence, work style",

        "primary_types": ["singularity_entry", "pillar_journal", "note", "data_point"],
        "primary_queries": [
            "career work engineer leadership influence articulation role",
            "promotion manager team product strategy company",
            "work style collaboration cross-functional impact visibility",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "career advice job work manager promotion strategy",
            "resume interview professional development leadership",
        ],

        "secondary_types": ["browser_daily", "weekly_review", "task", "week_carry"],
        "secondary_queries": [
            "work sites meetings project deadlines jira slack",
            "weekly wins work accomplishments shipped delivered",
        ],

        "sql_tables": ["entries", "tasks", "communications"],
        "sql_filters": {"pillar": "MIND"},

        "do": [
            "Ground answers in Singularity MIND/career pillar entries (score, label, tags)",
            "Reference pillar_journal entries for career reflections and leadership growth",
            "Check Apple Notes for career planning, 1:1 notes, or strategy documents",
            "Use LLM conversations to find career questions, resume drafts, or job discussions",
            "Include weekly_review data for current work focus and accomplishments",
            "Reference communications table for networking and interaction patterns",
            "Show career trajectory from timestamped entries — not just current state",
        ],
        "dont": [
            "Make up job titles, company names, or responsibilities not in the data",
            "Speculate about salary, compensation, or performance reviews",
            "Assume career satisfaction — report what the data shows (frustration is valid)",
            "Share details from 1:1 notes or private manager conversations broadly",
            "Conflate aspirations from AI conversations with actual career moves",
        ],
        "habits": [
            "Work hours and productivity patterns from browser/task data",
            "Meeting frequency and communication style with different stakeholders",
            "How often user reflects on career via journals and notes",
        ],
        "preferences": [
            "Leadership style (hands-on vs delegating)",
            "Preferred working mode (async, deep work, collaborative)",
            "Career priority (IC track vs management track)",
        ],
        "data_goals": [
            "Map full career history from all sources",
            "Track influence and visibility growth via communications metrics",
            "Identify career themes (e.g., always gravitates toward infrastructure)",
        ],
        "edge_cases": [
            "User venting about work in AI chats ≠ objective career assessment",
            "Old job entries may reference previous companies — check timestamps",
            "Task completion data may not distinguish work tasks from personal tasks",
        ],
    },

    "learning": {
        "description": "Courses, books, tutorials, certifications, growth areas, self-education",

        "primary_types": ["singularity_entry", "plan_note", "note", "data_point"],
        "primary_queries": [
            "learning course tutorial book education study research",
            "certification bootcamp workshop conference growth skill",
            "30 day plan learning goals knowledge development",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "learn explain teach me how does understand concept",
            "tutorial guide course recommendation what should I study",
        ],

        "secondary_types": ["browser_domain", "browser_daily"],
        "secondary_queries": [
            "udemy coursera youtube deeplearning.ai learning platforms",
            "documentation API docs research papers articles read",
        ],

        "sql_tables": ["notes_index"],

        "do": [
            "List specific courses, books, or tutorials from Apple Notes and plan_notes",
            "Reference 30-day plan entries for current learning priorities",
            "Check LLM conversation history — topics the user asked AI to explain",
            "Use browser_domain data to confirm actual learning platform usage",
            "Cross-reference notes_index for Apple Notes categorized under learning",
            "Show learning trajectory — what topics evolved over time via timestamps",
            "Distinguish self-directed learning from structured courses",
        ],
        "dont": [
            "Assume completion — 'enrolled' vs 'completed' vs 'abandoned' based on data",
            "Recommend courses unless explicitly asked — focus on what user has done",
            "Count AI conversations as completed learning — they're exploration",
            "Treat one-time browser visits to docs as sustained learning interest",
        ],
        "habits": [
            "Preferred learning format (video, text, hands-on, AI-assisted)",
            "Learning schedule and consistency from browser timestamps",
            "Topics revisited frequently vs one-time explorations",
        ],
        "preferences": [
            "Depth vs breadth in learning (specialist vs generalist)",
            "Free vs paid learning resources",
            "Solo learning vs cohort-based",
        ],
        "data_goals": [
            "Build a complete learning timeline from all sources",
            "Identify current active learning topics vs historical interests",
            "Track which learning methods lead to actual skill application",
        ],
        "edge_cases": [
            "Asking AI to explain X ≠ user learning X systematically",
            "Browser history may show research for work, not personal learning",
            "Plan_note goals may be aspirational — check if followed through",
        ],
    },

    "wellness": {
        "description": "Gym frequency, workouts, meditation, sleep, mental health, health goals",

        "primary_types": ["body_gym", "body_wellness", "soul_checkin", "singularity_entry", "pillar_journal"],
        "primary_queries": [
            "gym workout exercise fitness strength training cardio yoga",
            "meditation mindfulness sleep recovery mental health therapy",
            "body pillar health journal reflection wellness habits",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "workout routine exercise plan fitness goal health",
            "meditation stress sleep recovery mental health wellness",
        ],

        "secondary_types": ["weekly_review", "data_point", "note"],
        "secondary_queries": [
            "health habits weekly wins body progress streak",
            "wellness goals morning routine self-care",
        ],

        "sql_tables": ["gym", "wellness"],

        "do": [
            "Use body_gym table for workout frequency, types (workout_type), and intensity",
            "Use body_wellness table for habit tracking (days_done out of 7)",
            "Reference soul_checkin data for meditation streaks and daily signals",
            "Include Singularity BODY pillar entries for health reflections (score, label)",
            "Check LLM conversations for workout plans or health questions asked",
            "Cite specific week_ids and dates from structured data for precision",
            "Show streaks AND gaps — both are informative",
            "Look for cluster_label matches like 'fitness_tracking', 'meditation_practice'",
        ],
        "dont": [
            "Give medical advice or diagnose health conditions",
            "Assume consistency — report actual streaks and gaps from gym/wellness tables",
            "Mix up gym data (workout_type, intensity) with nutrition data (separate dimension)",
            "Treat soul_checkin mood data as clinical mental health assessment",
            "Infer workout quality from frequency alone — check intensity and notes",
        ],
        "habits": [
            "Weekly gym frequency and preferred workout days",
            "Meditation consistency from soul_checkin",
            "Sleep patterns and morning routine habits",
            "Which wellness habits stick vs which get dropped",
        ],
        "preferences": [
            "Workout types preferred (lifting, cardio, yoga, mixed)",
            "Gym vs home workout preference",
            "Meditation style and duration",
        ],
        "data_goals": [
            "Track gym consistency trends across weeks (gym table by week_id)",
            "Correlate wellness habits with soul_checkin mood signals",
            "Identify which wellness habits have the longest active streaks",
        ],
        "edge_cases": [
            "Missing weeks in gym data may mean rest, not quitting",
            "Soul_checkin data is self-reported — may not capture skipped days",
            "Wellness notes field contains JSON — parse for veggies, protein, carbs",
        ],
    },

    "nutrition": {
        "description": "Dietary preferences, cuisines, cooking, restaurants, meal patterns",

        "primary_types": ["body_nutrition", "data_point", "singularity_entry", "note"],
        "primary_queries": [
            "food diet meal cooking restaurant cuisine protein calories",
            "vegetarian vegan thai indian breakfast lunch dinner",
            "recipe meal prep grocery nutrition eating habits",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "food recipe diet meal restaurant recommendation cooking",
            "nutrition protein calories healthy eating plan",
        ],

        "secondary_types": ["browser_daily", "photo_daily"],
        "secondary_queries": [
            "food delivery restaurant cafe dining ubereats doordash",
            "food photos restaurant meals dining out",
        ],

        "sql_tables": ["nutrition"],

        "do": [
            "Use nutrition table for weekly data: meal_source (home vs out), calorie_deficit (on/off)",
            "Parse nutrition.notes JSON for detailed breakdown: veggies, protein, carbs, cheats",
            "Reference self-reported data_points for dietary identity and preferences",
            "Check Apple Notes for recipes, food discoveries, or diet plans",
            "Use LLM conversations for food-related questions and recipe requests",
            "Cite photo_daily chunks tagged with food/restaurant category",
            "Show patterns: home cooking frequency, eating out patterns, cuisine preferences",
        ],
        "dont": [
            "Prescribe diets or calorie targets — only report observed patterns",
            "Assume dietary restrictions without explicit self-reported evidence",
            "Confuse one restaurant visit with a regular preference",
            "Treat calorie_deficit='off' as failure — user may have intentional flex days",
            "Make moral judgments about food choices (cheat days, etc.)",
        ],
        "habits": [
            "Home cooking vs eating out ratio (from meal_source)",
            "Calorie tracking consistency (calorie_deficit on/off patterns)",
            "Cuisine rotation and favorite restaurants",
            "Meal prep frequency and cooking patterns",
        ],
        "preferences": [
            "Dietary identity (vegetarian, pescatarian, no restrictions, etc.)",
            "Cuisine preferences ranked by frequency",
            "Cooking complexity preference (simple vs elaborate)",
            "Eating schedule and meal frequency",
        ],
        "data_goals": [
            "Build cuisine preference profile from all sources",
            "Track dietary consistency trends across weeks",
            "Identify favorite restaurants from photos + notes + browser",
        ],
        "edge_cases": [
            "Nutrition notes are JSON strings — must parse veggies/protein/carbs/cheats fields",
            "Photo food tags are AI-inferred from descriptions — may be inaccurate",
            "LLM recipe requests don't mean user actually cooked the recipe",
            "Browser food delivery visits show ordering, not actual consumption",
        ],
    },

    "creative": {
        "description": "Content creation, music, writing, design, creative outlets and process",

        "primary_types": ["singularity_entry", "pillar_journal", "note", "data_point"],
        "primary_queries": [
            "creative content creation music writing design art",
            "youtube video substack blog post produce create",
            "creative process flow state inspiration aesthetic",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "creative writing design music produce content idea",
            "video editing blog post social media content strategy",
        ],

        "secondary_types": ["browser_domain", "browser_daily"],
        "secondary_queries": ["youtube figma canva creative tools design platforms"],

        "sql_tables": ["entries"],
        "sql_filters": {"pillar": "SOUL"},

        "do": [
            "Reference Singularity SOUL/create pillar entries for creative projects",
            "Check pillar_journal entries for creative reflections and process notes",
            "Include Apple Notes with creative ideas, drafts, or project plans",
            "Use LLM conversations to find content drafts, writing help, design discussions",
            "Mention specific creative outputs: videos, posts, music, designs",
            "Include tools and platforms used (from browser + notes)",
            "Distinguish between creating and consuming creative content",
        ],
        "dont": [
            "Conflate consuming content (watching YouTube) with creating it",
            "Assume artistic skill level without evidence of actual output",
            "Treat brainstorming in AI chats as completed creative work",
            "Ignore creative blocks or struggles — they're part of the process",
        ],
        "habits": [
            "Creative output frequency and preferred time for creative work",
            "Tools and platforms used for different creative formats",
            "Inspiration sources and creative influences",
        ],
        "preferences": [
            "Preferred creative medium (video, writing, music, design)",
            "Solo creation vs collaborative",
            "Polished output vs rapid experimentation",
        ],
        "data_goals": [
            "Map all creative outputs across time from all sources",
            "Track creative consistency — is output regular or in bursts?",
            "Identify which creative pursuits have sustained vs faded",
        ],
        "edge_cases": [
            "Browser visits to YouTube could be consumption or research for creation",
            "AI conversation about 'writing a blog post' may never have been published",
            "Singularity SOUL entries may mix creative with emotional/spiritual topics",
        ],
    },

    "vibe": {
        "description": "Energy patterns, mood, atmosphere, aesthetic, music taste, daily rhythm",

        "primary_types": ["soul_checkin", "data_point", "singularity_entry", "note"],
        "primary_queries": [
            "energy mood vibe atmosphere aesthetic calm focus",
            "morning evening routine rhythm daily energy pattern",
            "music playlist ambient lofi chill focus concentration",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "mood energy feeling vibe atmosphere music playlist",
            "morning routine evening wind down focus flow state",
        ],

        "secondary_types": ["weekly_review", "body_wellness"],
        "secondary_queries": ["mood energy weekly reflection wellness habits"],

        "sql_tables": ["wellness"],

        "do": [
            "Use soul_checkin signals for daily energy and mood patterns (daily_signals field)",
            "Reference self-reported data_points for vibe and aesthetic preferences",
            "Check Apple Notes for atmosphere preferences, music lists, or vibe descriptions",
            "Use LLM conversations about mood, playlists, or environment preferences",
            "Show patterns across time — morning vs evening energy, weekly rhythms",
            "Cross-reference wellness table for habit completion affecting mood",
            "Look for cluster_label matches like 'mindset_confidence', 'morning_energy'",
        ],
        "dont": [
            "Diagnose mental health conditions from mood data",
            "Generalize one bad day or week into a personality pattern",
            "Assume vibe preferences are permanent — show evolution over time",
            "Conflate stated preferences with actual behavior (says they're a morning person but data shows otherwise)",
        ],
        "habits": [
            "Time of day with highest energy from soul_checkin patterns",
            "Music listening patterns tied to moods or activities",
            "Environment preferences for focus vs relaxation",
        ],
        "preferences": [
            "Morning person vs night owl (from data, not self-report)",
            "Preferred atmosphere for deep work",
            "Music genres/artists for different moods",
            "Aesthetic style (minimalist, cozy, energetic, etc.)",
        ],
        "data_goals": [
            "Build a mood/energy profile across the week",
            "Correlate habit completion (wellness table) with mood signals",
            "Map music and atmosphere preferences to productivity patterns",
        ],
        "edge_cases": [
            "Soul_checkin is self-reported at a point in time — mood may shift within the day",
            "Vibe preferences stated in AI chats may be aspirational, not actual",
            "Music taste from browser may differ from what user actually listens to offline",
        ],
    },

    "entertainment": {
        "description": "Movies, shows, music, podcasts, gaming, books, streaming habits",

        "primary_types": ["data_point", "note", "singularity_entry"],
        "primary_queries": [
            "movie film show series netflix anime music podcast book",
            "watching listening gaming streaming entertainment favorite",
            "sci-fi thriller comedy drama genre artist album band",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "movie show recommend watch listen music podcast game",
            "favorite show book series artist song album review",
        ],

        "secondary_types": ["browser_domain", "browser_daily"],
        "secondary_queries": [
            "netflix spotify youtube twitch streaming entertainment sites",
            "imdb goodreads letterboxd gaming platforms",
        ],

        "do": [
            "List specific titles, artists, genres, and platforms from all data sources",
            "Use browser_domain data for actual streaming platform usage frequency",
            "Check LLM conversations for entertainment recommendations asked and opinions shared",
            "Reference Apple Notes for watchlists, reading lists, or reviews",
            "Distinguish between active interests and one-time mentions",
            "Show preference evolution over time from timestamped data",
            "Cross-reference data sources: user mentioned X in chat AND has browser visits to X",
        ],
        "dont": [
            "Recommend entertainment unless explicitly asked — focus on observed patterns",
            "Assume taste from a single data point — look for repeated engagement",
            "Treat AI-generated recommendations in conversations as user preferences",
            "Conflate what user asked about with what user actually watched/played",
        ],
        "habits": [
            "Streaming platform of choice and usage frequency",
            "Binge patterns vs regular viewing habits",
            "Music listening frequency and context (work, commute, exercise)",
            "Reading habits and book completion rate",
        ],
        "preferences": [
            "Favorite genres across media types",
            "Preferred platforms and services",
            "Solo vs social entertainment (watching alone vs with others)",
            "Active (gaming, reading) vs passive (streaming) entertainment ratio",
        ],
        "data_goals": [
            "Build a comprehensive taste profile across all media types",
            "Track which genres/artists come up across multiple sources",
            "Identify seasonal or mood-based entertainment patterns",
        ],
        "edge_cases": [
            "Browser visits to Netflix don't confirm what was watched",
            "Asking AI 'what should I watch?' doesn't mean they watched the suggestion",
            "Music browser visits may be for work background, not active listening",
        ],
    },

    "relationships": {
        "description": "Communication style, social preferences, networking, conflict, team dynamics",

        "primary_types": ["singularity_entry", "pillar_journal", "data_point", "note"],
        "primary_queries": [
            "communication networking social relationship people team",
            "friend family partner collaboration mentor dating",
            "conflict resolution feedback giving receiving boundary",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "relationship advice communication social interaction team",
            "networking friend family partner conflict boundary",
        ],

        "secondary_types": ["card_counters", "weekly_review"],
        "secondary_queries": [
            "networking reps social interactions connection events",
            "communication wins team collaboration weekly reflection",
        ],

        "sql_tables": ["communications"],

        "do": [
            "Reference Singularity SOCIAL/voice pillar for communication patterns",
            "Use communications table for quantitative interaction data (manager, co_worker, networking)",
            "Use card_counters for networking rep milestones and consistency",
            "Check pillar_journal for relationship reflections and social growth",
            "Include Apple Notes about relationship plans, social goals, or reflections",
            "Use LLM conversations about relationship advice or social situations",
            "Show interaction patterns from communications table across weeks",
        ],
        "dont": [
            "Share specific names or private relationship details from any source",
            "Judge communication style — describe it neutrally with evidence",
            "Assume social preferences are fixed — show evolution from timestamps",
            "Expose sensitive relationship discussions from LLM conversations",
            "Conflate professional networking with personal relationships",
        ],
        "habits": [
            "Networking frequency from card_counters and communications table",
            "Manager interaction patterns (skip_level, co_worker columns)",
            "Social event attendance and new person meeting frequency",
        ],
        "preferences": [
            "Communication style (direct, diplomatic, context-heavy)",
            "Networking approach (event-based, 1:1, organic)",
            "Team collaboration style (lead, contribute, support)",
            "Conflict resolution preference (confront, avoid, mediate)",
        ],
        "data_goals": [
            "Track networking consistency from card_counters over time",
            "Map communication patterns by stakeholder type",
            "Identify social growth trajectory from pillar journals",
        ],
        "edge_cases": [
            "Communications table counts interactions, not quality — don't infer relationship depth",
            "Relationship advice sought in AI chats may be hypothetical or for someone else",
            "Card counter reps reset periodically — compare within periods, not across",
        ],
    },

    "language_style": {
        "description": "Writing tone, vocabulary, phrasing, formality, communication patterns",

        "primary_types": ["user_message", "conversation_pair", "data_point", "note"],
        "primary_queries": [
            "writing style tone vocabulary casual direct technical",
            "communication phrasing formality expression language",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "writing message communication expression style tone",
        ],

        "secondary_types": ["pillar_journal", "singularity_entry"],
        "secondary_queries": ["journal entries writing samples reflections"],

        "do": [
            "Analyze actual user_message chunks for tone, vocabulary, and phrasing patterns",
            "Compare language across platforms: ChatGPT vs Claude vs Gemini (source field)",
            "Reference conversation_pair chunks to see how user's style adapts to AI responses",
            "Check Apple Notes and pillar_journals for non-AI writing style",
            "Note differences in style across contexts (technical vs casual vs reflective)",
            "Look for common phrases, sentence starters, and vocabulary preferences",
            "Use data_point self-reports for how user describes their own style",
        ],
        "dont": [
            "Fabricate example phrases the user never wrote — quote actual chunks",
            "Over-index on AI chat style — it may differ from real-world communication",
            "Assume formality level is uniform — likely varies by context",
            "Count assistant responses as user language — only user messages matter",
            "Treat typos or shorthand in quick AI prompts as actual writing style",
        ],
        "habits": [
            "Typical message length across platforms",
            "Formality gradient (notes > journals > AI chats)",
            "Common interjections, filler words, or signature phrases",
        ],
        "preferences": [
            "Preferred communication length (concise vs detailed)",
            "Technical jargon usage level",
            "Humor style and frequency",
            "Emoji and punctuation usage patterns",
        ],
        "data_goals": [
            "Build vocabulary and tone profile from actual user messages",
            "Compare language style across platforms and time periods",
            "Identify if user adapts language to different AI assistants",
        ],
        "edge_cases": [
            "AI prompt style ('fix this code') is not the same as natural writing",
            "Very short prompts may not reveal style — focus on longer messages",
            "Multilingual users may switch languages — detect and note all",
            "Old conversations may show evolved language patterns",
        ],
    },

    "goals": {
        "description": "Short/long-term goals, professional targets, personal aspirations, planning",

        "primary_types": ["plan_note", "goals_completed", "singularity_entry", "task", "note"],
        "primary_queries": [
            "goal plan target objective milestone aspiration ship build",
            "30 day plan quarter roadmap priority deadline commitment",
            "life goal career goal health goal creative goal project",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "goal plan want to achieve build ship create accomplish",
            "resolution priority focus what should I work on next",
        ],

        "secondary_types": ["weekly_review", "week_carry"],
        "secondary_queries": [
            "weekly wins progress goal tracking what I accomplished",
            "carried over unfinished pending next week",
        ],

        "sql_tables": ["tasks", "entries"],

        "do": [
            "Use plan_note chunks for current 30-day plan — these are the active priorities",
            "Reference goals_completed for achieved milestones with timestamps",
            "Include task chunks for in-flight work items (by pillar and date)",
            "Check Apple Notes for goal-setting documents, life plans, or wish lists",
            "Use LLM conversations to find aspirational goals discussed with AI",
            "Use weekly_review data to validate which goals got actual work",
            "Use week_carry data to identify goals that repeatedly roll over (stuck goals)",
            "Show goal status explicitly: active, completed, abandoned, or stalled",
            "Reference tasks table for task completion rates by pillar",
        ],
        "dont": [
            "Assume goals are still active if data is old — check recency of evidence",
            "Conflate aspirations (I want to...) with commitments (I will...)",
            "Ignore abandoned goals — they show decision-making patterns and pivots",
            "Treat AI brainstorming sessions as committed goals",
            "Mix up someone else's goals mentioned in conversation with user's own goals",
        ],
        "habits": [
            "Goal setting cadence (monthly plans, weekly reviews)",
            "Completion rate from goals_completed vs goals set",
            "Which pillars get the most goal attention from tasks data",
            "How often goals carry over in week_carry (stickiness)",
        ],
        "preferences": [
            "Planning horizon (30-day, quarterly, yearly)",
            "Goal granularity (big picture vs specific deliverables)",
            "Accountability method (self-tracking, public, shared)",
        ],
        "data_goals": [
            "Build complete goal timeline: set → worked on → completed/abandoned",
            "Track goal themes across pillars — where does user focus energy?",
            "Identify patterns in what goals succeed vs fail from historical data",
        ],
        "edge_cases": [
            "Plan_notes may be aspirational at creation — check weekly_review for follow-through",
            "Goals mentioned in AI chats may be exploratory, not committed",
            "Task completion in Singularity may double-count across sources",
            "week_carry items persisting 3+ weeks likely need re-evaluation",
        ],
    },

    "life": {
        "description": "Daily routines, shopping, living situation, habits, lifestyle choices",

        "primary_types": ["soul_checkin", "data_point", "note", "singularity_entry"],
        "primary_queries": [
            "morning routine evening routine daily life habit schedule",
            "shopping apartment home living situation lifestyle",
            "cafe commute time management productivity daily",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "routine schedule daily life habit morning evening plan",
            "apartment living shopping travel commute lifestyle",
        ],

        "secondary_types": ["browser_daily", "browser_domain", "photo_daily"],
        "secondary_queries": [
            "shopping amazon zara retail lifestyle sites",
            "daily activity photos places visited events",
        ],

        "sql_tables": ["wellness", "browser"],

        "do": [
            "Use soul_checkin for daily habit patterns (which habits, days_done consistency)",
            "Reference Apple Notes for life plans, moving plans, or routine designs",
            "Use LLM conversations about routine optimization, life decisions, or shopping",
            "Include browser data for shopping patterns and lifestyle site visits",
            "Cite photo_daily data for places visited, activities, and daily snapshots",
            "Use wellness table for habit tracking data (task_id, days_done)",
            "Use browser table for category-level site usage patterns",
            "Show lifestyle patterns from multiple sources, not just one",
        ],
        "dont": [
            "Share specific financial details or spending amounts",
            "Assume routines are consistent — show actual patterns and gaps",
            "Treat browser shopping visits as purchases",
            "Infer living situation from indirect data without confirmation",
            "Share location data from photos without user's awareness",
        ],
        "habits": [
            "Morning and evening routine consistency from soul_checkin",
            "Shopping frequency and preferred platforms from browser",
            "Cooking vs eating out patterns (overlaps with nutrition)",
            "Commute patterns and daily schedule rhythm",
        ],
        "preferences": [
            "Lifestyle priorities (minimalism, comfort, experience-oriented)",
            "Preferred shopping channels (online vs in-store, specific brands)",
            "Daily structure preference (rigid routine vs flexible)",
        ],
        "data_goals": [
            "Build complete daily routine profile from all sources",
            "Track lifestyle evolution via timestamped data",
            "Identify which habits are sustained vs dropped",
        ],
        "edge_cases": [
            "Soul_checkin may miss days — gaps don't mean habits stopped",
            "Browser shopping visits may be browsing, not buying",
            "Photo location data may be imprecise or private",
            "Routine discussions in AI chats may be aspirational, not actual",
        ],
    },

    "progress": {
        "description": "Weekly wins, milestones, streaks, active projects, trajectory",

        "primary_types": ["weekly_review", "week_carry", "card_counters", "goals_completed", "singularity_entry"],
        "primary_queries": [
            "progress wins milestone streak achievement shipped launched",
            "weekly review retrospective accomplishment delivered",
            "score trend improvement growth trajectory momentum",
        ],

        "memory_types": ["user_message", "conversation_pair"],
        "memory_queries": [
            "progress update what I did accomplished shipped launched",
            "milestone achieved completed built deployed",
        ],

        "secondary_types": ["task", "plan_note"],
        "secondary_queries": [
            "project status task completion active work in-flight",
            "30 day plan check-in progress update",
        ],

        "sql_tables": ["tasks", "entries", "weekly_review"],

        "do": [
            "Lead with weekly_review data for most recent progress (weekly_wins from notes JSON)",
            "Use card_counters for rep milestones and streak data",
            "Reference week_carry for what rolled over — shows both progress and blockers",
            "Use goals_completed for achieved milestones with dates",
            "Include Singularity entries with high scores as evidence of meaningful work",
            "Use tasks table for task completion rates (completed_tasks, total_entries)",
            "Parse weekly_review.notes JSON for weekly_wins array and feedback array",
            "Show trajectory — improving, plateauing, or declining across weeks",
            "Use LLM conversations where user discussed their progress or celebrated wins",
        ],
        "dont": [
            "Cherry-pick only wins — include setbacks and blockers for honest assessment",
            "Report progress without time context (when did it happen?)",
            "Assume lack of data means lack of progress — user may work outside tracked systems",
            "Ignore stalled items in week_carry — persistent carryover signals blockers",
            "Compare across people — only compare user against their own history",
        ],
        "habits": [
            "Weekly review completion consistency",
            "Which pillars show most progress from tasks data",
            "Streak maintenance patterns from card_counters",
            "Momentum patterns (bursts vs steady progress)",
        ],
        "preferences": [
            "Progress tracking style (quantitative metrics vs qualitative reflection)",
            "Celebration style (public wins vs private satisfaction)",
            "Review cadence (weekly, monthly, quarterly)",
        ],
        "data_goals": [
            "Build multi-week progress timeline from weekly_review data",
            "Track which pillars receive consistent effort vs sporadic attention",
            "Identify what causes momentum loss from week_carry patterns",
        ],
        "edge_cases": [
            "Weekly review notes are JSON — parse weekly_wins and feedback arrays",
            "Card counter reps may reset — compare within periods not across",
            "High Singularity scores don't always mean important work — check labels",
            "Tasks table counts entries, not effort — one big task ≠ one small task",
        ],
    },
}


def generate_skill_file(dim: PersonaDimension) -> str:
    """Generate a markdown skill file with traits + retrieval sources."""
    lines = []
    lines.append(f"# {dim.display_name}")
    lines.append(f"<!-- pillar: {dim.pillar} | dimension: {dim.name} -->")
    lines.append(f"<!-- confidence: {dim.confidence:.0%} | evidence: {dim.evidence_count} chunks -->")
    if dim.last_updated:
        lines.append(f"<!-- last updated: {dim.last_updated[:10]} -->")
    lines.append("")

    # Traits section
    lines.append("## Traits")
    lines.append("")

    if not dim.traits:
        lines.append("*No data yet. This dimension will populate as more data flows in.*")
    else:
        for key, value in dim.traits.items():
            label = key.replace("_", " ").title()

            if isinstance(value, list) and value:
                lines.append(f"### {label}")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"- {json.dumps(item)}")
                    else:
                        lines.append(f"- {item}")
                lines.append("")

            elif isinstance(value, dict) and value:
                lines.append(f"### {label}")
                for k, v in value.items():
                    lines.append(f"- **{k}**: {v}")
                lines.append("")

            elif isinstance(value, str) and value:
                lines.append(f"### {label}")
                lines.append(value)
                lines.append("")

    sources = DIMENSION_SOURCES.get(dim.name, {})

    # Instructions section — what to do and not do
    lines.append("")
    lines.append("## Instructions")
    lines.append("")

    if sources:
        if sources.get("do"):
            lines.append("### Do")
            for item in sources["do"]:
                lines.append(f"- {item}")
            lines.append("")

        if sources.get("dont"):
            lines.append("### Don't")
            for item in sources["dont"]:
                lines.append(f"- {item}")
            lines.append("")

        if sources.get("edge_cases"):
            lines.append("### Edge Cases")
            for item in sources["edge_cases"]:
                lines.append(f"- {item}")
            lines.append("")

    # Habits & Preferences
    if sources.get("habits") or sources.get("preferences"):
        lines.append("")
        lines.append("## Habits & Preferences")
        lines.append("")

        if sources.get("habits"):
            lines.append("### Known Habits to Look For")
            for item in sources["habits"]:
                lines.append(f"- {item}")
            lines.append("")

        if sources.get("preferences"):
            lines.append("### Known Preferences to Prioritize")
            for item in sources["preferences"]:
                lines.append(f"- {item}")
            lines.append("")

    # Data Collection Goals
    if sources.get("data_goals"):
        lines.append("")
        lines.append("## Data Collection Goals")
        lines.append("")
        for item in sources["data_goals"]:
            lines.append(f"- {item}")
        lines.append("")

    # Sources section — 3-tier retrieval orchestration
    lines.append("")
    lines.append("## Sources — Retrieval Strategy")
    lines.append("")

    if sources:
        lines.append(f"**What this covers:** {sources.get('description', '')}")
        lines.append("")

        lines.append("### Tier 1: Core (always searched)")
        lines.append(f"Types: `{', '.join(sources.get('primary_types', []))}`")
        lines.append("Queries:")
        for q in sources.get("primary_queries", []):
            lines.append(f"- `{q}`")
        lines.append("")

        if sources.get("memory_types"):
            lines.append("### Tier 2: LLM Memory (ChatGPT/Claude/Gemini conversations)")
            lines.append(f"Types: `{', '.join(sources.get('memory_types', []))}`")
            lines.append("Queries:")
            for q in sources.get("memory_queries", []):
                lines.append(f"- `{q}`")
            lines.append("")

        lines.append("### Tier 3: Supplementary (when Tier 1+2 are thin)")
        lines.append(f"Types: `{', '.join(sources.get('secondary_types', []))}`")
        lines.append("Queries:")
        for q in sources.get("secondary_queries", []):
            lines.append(f"- `{q}`")
        lines.append("")

        if sources.get("sql_tables"):
            lines.append("### SQL Tables (for quantitative questions)")
            lines.append(f"Tables: `{', '.join(sources.get('sql_tables', []))}`")
            if sources.get("sql_filters"):
                filters = ", ".join(f"{k}={v}" for k, v in sources["sql_filters"].items())
                lines.append(f"Filters: `{filters}`")
            lines.append("")
    else:
        lines.append("*No source configuration yet.*")

    lines.append("")
    return "\n".join(lines)


def write_all_skill_files(dimensions: dict[str, PersonaDimension]) -> list[Path]:
    """Write all dimension skill files to disk. Returns list of written paths."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, dim in dimensions.items():
        path = SKILLS_DIR / f"{name}.md"
        # Only overwrite the Traits section if file exists with user edits
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            # Check if user added custom content (not auto-generated)
            if "<!-- user-edited -->" in existing:
                # Only update the Traits section, preserve Sources and custom content
                new_content = generate_skill_file(dim)
                # Keep user's Sources section if they edited it
                if "## Sources" in existing:
                    user_sources = existing[existing.index("## Sources"):]
                    new_traits = new_content[:new_content.index("## Sources")]
                    content = new_traits + user_sources
                else:
                    content = new_content
            else:
                content = generate_skill_file(dim)
        else:
            content = generate_skill_file(dim)

        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def read_skill_file(dimension_name: str) -> str | None:
    """Read a single skill file. Returns content or None."""
    path = SKILLS_DIR / f"{dimension_name}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def write_skill_file(dimension_name: str, content: str) -> bool:
    """Write a single skill file (user edit). Marks as user-edited."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    path = SKILLS_DIR / f"{dimension_name}.md"

    # Add user-edited marker if not present
    if "<!-- user-edited -->" not in content:
        # Insert after the first HTML comment block
        lines = content.split("\n")
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("<!--"):
                insert_at = i + 1
        lines.insert(insert_at, "<!-- user-edited -->")
        content = "\n".join(lines)

    path.write_text(content, encoding="utf-8")
    return True


def read_all_skill_files() -> dict[str, str]:
    """Read all skill files from disk. Returns {dimension_name: content}."""
    if not SKILLS_DIR.exists():
        return {}
    skills = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        skills[path.stem] = path.read_text(encoding="utf-8")
    return skills


def build_persona_from_skills() -> str:
    """Compose a system prompt from all populated skill files."""
    skills = read_all_skill_files()
    if not skills:
        return ""

    parts = []
    parts.append("You are a digital twin of a real person. Everything below describes who they are,")
    parts.append("extracted from their real data — notes, browsing, tasks, habits, health, goals.")
    parts.append("This evolves daily as new data arrives. Use it to respond exactly as they would.")
    parts.append("Each dimension also lists WHERE to find supporting data — use the Sources section")
    parts.append("to know which memory types to cite in your responses.\n")

    pillar_groups: dict[str, list[tuple[str, str]]] = {}
    for dim_name, content in skills.items():
        if "*No data yet" in content and "## Sources" not in content:
            continue
        meta = DIMENSIONS.get(dim_name, {})
        pillar = meta.get("pillar", "OTHER")
        pillar_groups.setdefault(pillar, []).append((dim_name, content))

    pillar_order = ["MIND", "BODY", "SOUL", "SOCIAL", "PURPOSE"]
    pillar_labels = {
        "MIND": "Mind & Career",
        "BODY": "Body & Health",
        "SOUL": "Soul & Creativity",
        "SOCIAL": "Social & Communication",
        "PURPOSE": "Purpose & Goals",
    }

    for pillar in pillar_order:
        group = pillar_groups.get(pillar)
        if not group:
            continue
        parts.append(f"\n---\n## {pillar_labels.get(pillar, pillar)}\n")
        for dim_name, content in group:
            lines = content.split("\n")
            clean_lines = [l for l in lines if not l.startswith("<!--")]
            parts.append("\n".join(clean_lines))

    return "\n".join(parts)
