# LLM Social Media Simulation: Staged Implementation Guide for Claude Code

## Project Overview

This project builds an LLM-agent-based social media simulation for youth co-design research. Participants describe their social network; the system populates a simulated platform with agents representing those people. Participants post content, observe agent reactions, and reason about design choices (connection caps vs. no caps) by inspecting emergent dynamics.

**Core research question:** Does interacting with an LLM simulation help youth surface insights about platform design that hypothetical reasoning alone does not?

---

## Technical Specification Summary

### Agent Architecture (Four Layers)

1. **Traits (stable):** Node characteristics (about the person) + edge characteristics (about relationship to participant)
2. **Beliefs (learned):** Norm estimates, attitude toward posting—inferred from memory for MVP
3. **State (transient):** Mood, recent experiences
4. **Memory:** Evidence like "last post got 2 likes," "got teased once"

### Agent Types

**Key ties (5-8):** Participant-described with rich detail
- Name, relationship type, 1-2 sentence description, contexts
- System infers node/edge characteristics from description

**Filler agents (30-60):** Auto-generated from templates
- Fill out contexts to realistic sizes
- Default characteristics based on relationship template

### Node Characteristics (1-5 scale)

| Dimension | 1 | 3 | 5 |
|-----------|---|---|---|
| Initiation scaffold need | Almost never posts | Prompts reliably work | Posts whenever they want |
| Reciprocation need | Doesn't care about reactions | Wants acknowledgment | Highly feedback-dependent |
| Norm sensitivity | Ignores the vibe | Usually follows norms | Won't post until others do |
| Privacy vigilance | Posts freely | Selective | Maximum caution |
| Norm enforcement | Never polices others | Occasionally signals disapproval | Strong enforcer |
| Leakiness | Never screenshots/forwards | Occasionally shares | Very leaky |
| Response expressiveness | Rarely reacts | Standard | Highly expressive |

### Edge Characteristics (1-5 scale)

| Dimension | 1 | 3 | 5 |
|-----------|---|---|---|
| Tie strength | Acquaintance | Friendly | Best friend |
| Judgment safety | Always evaluating | Generally supportive | Complete safety |
| Boundary respect | Would share anything | Generally respects | Complete trust |
| Reciprocity obligation | No pressure | Some expectation | Strong obligation |

### Connection Tiers & Design Conditions

**Tiers:** Close Friends → Friends → Followers

**Condition A (With Caps):**
- Close Friends: max 8
- Friends: max 30
- Followers: max 50

**Condition B (No Caps):** Unlimited

**Audience rules:**
- Close Friends post → only Close Friends see
- Friends post → Friends + Close Friends see
- Public post → everyone sees

### Simulation Loop

1. Participant posts (selects audience tier)
2. System determines who sees it (based on tier + condition)
3. For each agent who sees it: generate reaction via LLM
4. Time skip ("1 hour later...")
5. Display what happened
6. Optionally: agents post, participant sees feed
7. Repeat

### Violation Events

- **Screenshot:** Agent screenshots post, sends to other context
- **Unexpected viewer:** Family member/boss/ex sees and reacts

---

## Implementation Stages

The stages below are ordered by dependency. Each stage is self-contained and testable before moving to the next.

---

# STAGE 0: Project Cleanup & Schema Definition

## Goal
Clean slate with proper schema definitions. Clear out stale code, define data structures.

## Claude Code Instruction

```
CONTEXT:
I'm building an LLM social media simulation for research. The project has some existing files that need to be reorganized. I want a clean foundation.

TASK:
1. Read the current project structure to understand what exists
2. Create a new schema definition file (schema.py or models.py) with dataclasses/Pydantic models for:
   
   Agent:
   - id: str
   - name: str
   - type: "key_tie" | "filler"
   - relationship_template: str (e.g., "close friend", "classmate you don't talk to")
   - description: str (for key ties only, participant's description)
   - contexts: list[str] (e.g., ["school", "close_friends"])
   - tier: "close_friends" | "friends" | "followers"
   
   Node characteristics (all 1-5 int):
   - initiation_scaffold_need
   - reciprocation_need  
   - norm_sensitivity
   - privacy_vigilance
   - norm_enforcement
   - leakiness
   - response_expressiveness
   
   Edge characteristics (all 1-5 int, relationship TO participant):
   - tie_strength
   - judgment_safety
   - boundary_respect
   - reciprocity_obligation
   
   Memory (simplified for MVP):
   - recent_events: list[str] (short summaries like "last post got 2 likes")
   
   Post:
   - id: str
   - author_id: str (participant or agent)
   - content: str
   - audience_tier: "close_friends" | "friends" | "public"
   - timestamp: datetime
   - reactions: list[Reaction]
   
   Reaction:
   - agent_id: str
   - post_id: str
   - action: "like" | "comment_supportive" | "comment_neutral" | "comment_teasing" | "screenshot" | "ignore"
   - comment_text: str | None
   - reason: str (one-sentence explanation)
   - timestamp: datetime
   
   SimulationState:
   - condition: "caps" | "no_caps"
   - current_tick: int
   - agents: list[Agent]
   - posts: list[Post]
   - events: list[Event] (for violations)

3. Create a config.py with:
   - LLM API settings (placeholder)
   - Tier caps: {"close_friends": 8, "friends": 30, "followers": 50}
   - Context templates for filler generation

4. Update database.py to work with these schemas (or replace with SQLite via SQLAlchemy if that's cleaner)

OUTPUT:
- schema.py (or models.py)
- config.py
- updated database.py
- Brief test that models instantiate correctly

SUCCESS CRITERIA:
- Can instantiate Agent, Post, Reaction, SimulationState
- Schema matches the spec above
- No stale/conflicting code
```

---

# STAGE 1: Agent Generation System

## Goal
Generate both key tie agents (from participant input) and filler agents (from templates).

## Claude Code Instruction

```
CONTEXT:
Building on Stage 0. We have schema definitions. Now we need to generate agents.

The system needs two types of agents:
1. Key ties: Participant describes 5-8 people. We infer characteristics from their description.
2. Filler agents: Auto-generated to fill out contexts to realistic sizes.

TASK:
1. Create agent_generator.py with:

   A. Filler agent templates (dict mapping template name to default characteristics):
   
   FILLER_TEMPLATES = {
       "classmate_distant": {
           "relationship_template": "classmate you recognize but don't talk to",
           "initiation_scaffold_need": 1,
           "reciprocation_need": 2,
           "norm_sensitivity": 3,
           "privacy_vigilance": 3,
           "norm_enforcement": 1,
           "leakiness": 1,
           "response_expressiveness": 2,
           "tie_strength": 1,
           "judgment_safety": 3,
           "boundary_respect": 3,
           "reciprocity_obligation": 1
       },
       "classmate_acquaintance": {
           "relationship_template": "acquaintance from class",
           # ... values from MVP spec
       },
       "popular_person": {
           "relationship_template": "person who's popular",
           # norm_sensitivity: 4, leakiness: 3, norm_enforcement: 3, etc.
       },
       "coworker_friendly": {...},
       "manager": {...},
       "extended_family": {...},
       "parent_friend": {...}
   }

   B. Function: generate_filler_agents(context_name: str, count: int, template_distribution: dict) -> list[Agent]
      - Creates 'count' filler agents for a context
      - Uses template_distribution to vary agent types (e.g., 60% distant classmates, 20% acquaintances, 20% popular)
      - Names: "{Context}_{template}_{n}" (e.g., "School_classmate_7")

   C. Function: infer_characteristics_from_description(description: str, relationship_type: str) -> dict
      - Uses LLM to infer node/edge characteristics from participant's natural language description
      - Prompt should extract: Does this person post a lot? Are they supportive? Do they gossip? etc.
      - Returns dict of characteristic values (1-5)
      - Include a mock/fallback that returns sensible defaults without LLM for testing

   D. Function: create_key_tie_agent(name: str, relationship: str, description: str, contexts: list[str]) -> Agent
      - Creates a key tie agent using inferred characteristics
      - Falls back to relationship-based defaults if inference fails

   E. Function: populate_network(key_ties: list[dict], context_sizes: dict[str, int]) -> list[Agent]
      - Takes participant's key tie definitions and desired context sizes
      - Creates all key tie agents
      - Fills remaining slots in each context with filler agents
      - Example: context_sizes = {"school": 30, "close_friends": 6, "family": 10}
        If 3 key ties are in "school", generate 27 filler agents for school

2. Write to agents.csv and/or database with full agent data

3. Create a simple test script that:
   - Defines sample key ties and contexts
   - Runs populate_network
   - Prints agent counts per context and tier
   - Verifies no duplicate IDs

LLM PROMPT for characteristic inference (include in agent_generator.py):

CHARACTERISTIC_INFERENCE_PROMPT = '''
You are analyzing a participant's description of someone in their social network to infer behavioral characteristics.

Description: "{description}"
Relationship type: "{relationship}"

Rate each dimension 1-5 based on the description. If the description doesn't mention something, use the default for this relationship type.

Dimensions to rate:
- initiation_scaffold_need: How much prompting do they need to post? (1=never posts, 5=posts freely)
- reciprocation_need: How much do they need feedback? (1=doesn't care, 5=highly dependent)
- norm_sensitivity: How much do they follow norms? (1=ignores vibe, 5=extremely norm-driven)
- privacy_vigilance: How cautious about audience? (1=posts freely, 5=maximum caution)
- norm_enforcement: Do they police others? (1=never, 5=strong enforcer)
- leakiness: Do they share/screenshot things? (1=never, 5=very leaky)
- response_expressiveness: How do they react to others' posts? (1=minimal, 5=highly expressive)
- tie_strength: How close to participant? (1=acquaintance, 5=best friend)
- judgment_safety: Does participant feel judged? (1=always evaluating, 5=complete safety)
- boundary_respect: Would they share participant's content? (1=would share anything, 5=complete trust)
- reciprocity_obligation: Pressure to respond? (1=none, 5=strong)

Respond with JSON only:
{
  "initiation_scaffold_need": <int>,
  "reciprocation_need": <int>,
  ...
}
'''

SUCCESS CRITERIA:
- Can generate a full network of 40-70 agents from minimal input
- Key ties have inferred characteristics
- Filler agents have template-based characteristics
- Each agent has correct context membership
- agents.csv contains all agents with all fields
```

---

# STAGE 2: Tier Assignment & Audience Logic

## Goal
Implement connection tier assignment and audience visibility rules for both conditions (caps vs. no caps).

## Claude Code Instruction

```
CONTEXT:
We have agents generated. Now we need to assign them to tiers (Close Friends / Friends / Followers) and implement the audience rules that determine who sees which posts.

DESIGN CONDITIONS:
- Condition A (caps): Close Friends max 8, Friends max 30, Followers max 50
- Condition B (no caps): Unlimited

AUDIENCE RULES (same in both conditions):
- Close Friends post → only Close Friends tier sees
- Friends post → Friends + Close Friends see
- Public post → everyone sees

TASK:
1. Create tier_manager.py with:

   A. Function: assign_tiers_with_caps(agents: list[Agent], caps: dict) -> list[Agent]
      - Assigns agents to tiers respecting caps
      - Priority: key ties first, then by tie_strength
      - Returns agents with updated 'tier' field
      - Logic:
        1. Sort agents by (is_key_tie desc, tie_strength desc)
        2. Fill close_friends up to cap
        3. Fill friends up to cap
        4. Rest go to followers
   
   B. Function: assign_tiers_no_caps(agents: list[Agent]) -> list[Agent]
      - Assigns based on tie_strength thresholds:
        - tie_strength >= 4 → close_friends
        - tie_strength >= 2 → friends
        - else → followers
      - No caps enforced

   C. Function: get_audience(post_tier: str, agents: list[Agent]) -> list[Agent]
      - Returns list of agents who would see a post at given tier
      - "close_friends" → agents where tier == "close_friends"
      - "friends" → agents where tier in ["close_friends", "friends"]
      - "public" → all agents

   D. Function: switch_condition(agents: list[Agent], new_condition: str, caps: dict) -> list[Agent]
      - Re-assigns tiers when condition changes
      - Used for A/B comparison in study

2. Create audience_display.py with:

   A. Function: format_audience_preview(agents: list[Agent]) -> str
      - "This will be visible to: Alex, Jordan, Sam, and 12 others"
      - Show key ties by name, aggregate filler agents

   B. Function: format_audience_by_context(agents: list[Agent]) -> dict[str, list[str]]
      - Groups audience by context for display
      - {"school": ["Alex", "Jordan", "3 classmates"], "family": ["Mom", "2 others"]}

3. Test script that:
   - Creates a sample network
   - Applies caps condition, prints tier assignments
   - Applies no-caps condition, prints tier assignments
   - Shows audience for each post type
   - Verifies caps are respected

SUCCESS CRITERIA:
- Caps condition respects limits exactly
- No-caps condition assigns by tie_strength
- get_audience returns correct subsets
- Key ties prioritized in caps assignment
- Audience preview shows meaningful info
```

---

# STAGE 3: Behavior Generation (LLM Integration)

## Goal
Generate agent reactions to posts via LLM. Different prompts for key ties (rich) vs. filler agents (simple).

## Claude Code Instruction

```
CONTEXT:
We have agents and audience logic. Now we need agents to react to posts. This is the core LLM integration.

KEY DESIGN DECISIONS:
- Key ties: Full prompt with description, traits, relationship context
- Filler agents: Simpler prompt with just traits (faster, cheaper)
- Actions: like, comment_supportive, comment_neutral, comment_teasing, screenshot, ignore
- Each reaction includes a one-sentence reason (for explanation panel)

TASK:
1. Create behavior_generator.py with:

   A. KEY_TIE_REACTION_PROMPT template:
   
   '''
   You are simulating {name}, who is {participant_name}'s {relationship}.
   Description: "{description}"
   
   Their traits:
   - Reciprocation need: {reciprocation_need}/5 ({reciprocation_anchor})
   - Norm sensitivity: {norm_sensitivity}/5 ({norm_anchor})
   - Leakiness: {leakiness}/5 ({leakiness_anchor})
   - Response expressiveness: {response_expressiveness}/5 ({expressiveness_anchor})
   - Norm enforcement: {norm_enforcement}/5 ({enforcement_anchor})
   
   Their relationship to {participant_name}:
   - Tie strength: {tie_strength}/5 ({tie_anchor})
   - Judgment safety: {judgment_safety}/5 ({judgment_anchor})
   - Boundary respect: {boundary_respect}/5 ({boundary_anchor})
   
   Recent memory: {memory}
   
   {participant_name} posted: "{post_content}"
   Visible to: {audience_description}
   Post type: {post_type} (casual/vulnerable/controversial/etc.)
   
   Given this context, what does {name} do? Choose ONE action:
   - ignore (sees but doesn't react)
   - like
   - comment_supportive (write the comment)
   - comment_neutral (write the comment)
   - comment_teasing (write the comment)
   - screenshot (sends to someone outside this audience)
   
   Respond with JSON:
   {
     "action": "<action>",
     "comment_text": "<text or null>",
     "reason": "<one sentence explaining why, grounded in their traits/relationship>"
   }
   '''

   B. FILLER_AGENT_REACTION_PROMPT template (simpler):
   
   '''
   You are simulating a {relationship_template}.
   
   Traits: initiation={initiation}/5, norm_sensitivity={norm_sensitivity}/5, leakiness={leakiness}/5
   Relationship to poster: tie_strength={tie_strength}/5, judgment_safety={judgment_safety}/5
   
   They saw this post: "{post_content}"
   
   What do they do? Most fillers ignore posts from weak ties.
   
   Respond with JSON:
   {
     "action": "ignore|like|comment_neutral",
     "comment_text": "<text or null>",
     "reason": "<brief reason>"
   }
   '''

   C. Function: generate_reaction(agent: Agent, post: Post, audience: list[Agent]) -> Reaction
      - Selects appropriate prompt based on agent.type
      - Calls LLM API
      - Parses response into Reaction object
      - Handles errors gracefully (default to "ignore" if parse fails)

   D. Function: generate_reactions_batch(agents: list[Agent], post: Post) -> list[Reaction]
      - Generates reactions for all agents who see the post
      - Parallelizes LLM calls for speed (asyncio or threading)
      - Key ties: full prompt
      - Filler agents: simple prompt OR probabilistic shortcut:
        - If tie_strength <= 2 and response_expressiveness <= 2: 80% chance just return "ignore" without LLM call
      - Returns list of Reactions

   E. Anchor text mappings (for prompt readability):
   
   ANCHORS = {
       "reciprocation_need": {
           1: "doesn't care about reactions",
           2: "barely notices low reactions", 
           3: "wants some acknowledgment",
           4: "strongly wants feedback",
           5: "highly feedback-dependent"
       },
       # ... etc for all dimensions
   }

2. Create llm_client.py with:
   - Wrapper for your LLM API (Claude/GPT-4)
   - Handles rate limiting, retries, error handling
   - Mock mode for testing without API calls

3. Test script:
   - Create sample post
   - Generate reactions from 5 key ties + 10 filler agents
   - Print each reaction with reason
   - Time the batch generation
   - Verify JSON parsing works

SUCCESS CRITERIA:
- Key tie reactions reference their description/traits in the reason
- Filler agents mostly ignore weak-tie posts
- Reactions are contextually appropriate
- Batch generation completes in <30 seconds for 50 agents
- JSON parsing handles malformed responses gracefully
```

---

# STAGE 4: Simulation Loop

## Goal
Implement the turn-based simulation loop: post → reactions → time skip → display → repeat.

## Claude Code Instruction

```
CONTEXT:
We have agents, tiers, and behavior generation. Now we wire it into a simulation loop that advances time and tracks state.

SIMULATION FLOW:
1. Participant posts (content + audience tier)
2. System determines audience
3. Generate reactions from audience
4. Time skip (1 hour, next day, etc.)
5. Display "what happened"
6. (Optional) Agents post content
7. Repeat

TASK:
1. Create simulation.py (or heavily modify existing) with:

   A. Class: Simulation
      - __init__(self, agents: list[Agent], condition: str)
      - current_tick: int
      - posts: list[Post]
      - events: list[Event]
   
   B. Method: participant_posts(self, content: str, audience_tier: str) -> Post
      - Creates Post object
      - Stores in self.posts
      - Returns the post

   C. Method: process_tick(self) -> TickResult
      - For each new post since last tick:
        - Get audience
        - Generate reactions (batch)
        - Store reactions on post
      - Increment tick
      - Return TickResult with summary

   D. Method: get_what_happened(self, post: Post) -> WhatHappened
      - Returns structured summary:
        - who_saw: list of agent names (key ties named, fillers aggregated)
        - reactions: list of {agent_name, action, comment_text, reason}
        - audience_explanation: "Because you posted to Close Friends, only your 6 close friends could see it."

   E. Method: time_skip(self, amount: str) -> None
      - Updates internal clock
      - amount: "1_hour", "next_day", etc.
      - Could trigger state changes (agents' moods shift, etc.) - for MVP just advances tick

   F. Method: generate_agent_posts(self, count: int = 2) -> list[Post]
      - Randomly select 'count' agents to post
      - Generate mundane content via LLM
      - Used to make feed feel alive
      - Simple prompt: "Generate a brief, realistic social media post for a {relationship_template} who {description}"

2. Create simulation_runner.py with:
   
   A. Interactive loop (for testing):
      ```
      while True:
          content = input("Post content (or 'quit'): ")
          if content == 'quit': break
          tier = input("Audience (close_friends/friends/public): ")
          post = sim.participant_posts(content, tier)
          result = sim.process_tick()
          print(format_what_happened(sim.get_what_happened(post)))
          
          if input("Generate agent posts? (y/n): ") == 'y':
              agent_posts = sim.generate_agent_posts(2)
              for ap in agent_posts:
                  print(f"[{ap.author.name}]: {ap.content}")
      ```

   B. Session runner (for actual study):
      - Loads participant's network
      - Runs through structured tasks
      - Logs everything

3. Create formatters.py with:
   
   A. format_what_happened(wh: WhatHappened) -> str
      - Pretty prints the "what happened" panel
      - Example output:
        ```
        Your post was seen by: Alex, Jordan, Sam, Taylor (4 people)
        Reactions:
          - Alex: ❤️ "Aw I'm here for you!" (because: Alex is highly expressive and supportive)
          - Jordan: 👍
          - Sam: [saw but didn't react]
          - Taylor: [saw but didn't react]
        
        Because you posted to Close Friends, only your 6 close friends could see it.
        4 of them checked the app.
        ```

   B. format_feed(posts: list[Post], for_participant: bool) -> str
      - Formats a feed view

4. Test script that:
   - Initializes simulation with sample network
   - Runs 5 participant posts with different tiers
   - Shows what happened after each
   - Generates some agent posts
   - Prints final state summary

SUCCESS CRITERIA:
- Full loop works: post → reactions → display
- Reactions are correctly attributed to agents
- What-happened summary is readable and informative
- Agent posts are contextually plausible
- All state is logged for later analysis
```

---

# STAGE 5: Violation Events

## Goal
Implement privacy violation events: screenshot/forward and unexpected viewer.

## Claude Code Instruction

```
CONTEXT:
We have a working simulation loop. Now we add violation events that raise privacy concerns for participant reflection.

VIOLATION TYPES:
1. Screenshot: An agent screenshots a post and sends it to someone outside the intended audience
2. Unexpected viewer: Someone the participant didn't expect (family, boss, ex) sees and reacts

DESIGN:
- Violations are probabilistic based on agent characteristics
- Can also be triggered deterministically at specific points
- Violations create events that participants can inspect

TASK:
1. Create violations.py with:

   A. Function: check_screenshot_risk(agent: Agent, post: Post) -> float
      - Returns probability 0-1 that agent screenshots
      - Based on: agent.leakiness, agent.boundary_respect (inverse), post sensitivity
      - Example: leakiness=5, boundary_respect=1 → high probability

   B. Function: execute_screenshot(agent: Agent, post: Post, all_agents: list[Agent]) -> ScreenshotEvent | None
      - Determines if screenshot happens (roll against probability)
      - If yes: picks a recipient outside original audience
        - Prefer agents in different context from post's audience
        - Prefer agents the participant might be awkward about
      - Returns ScreenshotEvent:
        - source_agent: who screenshotted
        - post: the post
        - recipient_agent: who received it
        - recipient_contexts: what contexts they're in
        - consequence: did recipient react? spread further?

   C. Function: check_unexpected_viewer_risk(post: Post, all_agents: list[Agent]) -> list[Agent]
      - Returns agents who might unexpectedly see the post
      - Logic: agents NOT in intended audience but who are in overlapping contexts
      - Weighted by: network position, activity level
      - For no-caps condition: higher risk due to larger audiences

   D. Function: execute_unexpected_viewer(unexpected_agents: list[Agent], post: Post) -> UnexpectedViewerEvent | None
      - Probabilistically picks one unexpected agent
      - Generates their reaction (might be awkward)
      - Returns UnexpectedViewerEvent:
        - viewer: the unexpected agent
        - post: the post
        - how_they_saw: "saw in their feed" / "someone showed them"
        - reaction: their response

   E. Function: trigger_violation(simulation: Simulation, violation_type: str, post: Post) -> Event
      - Deterministic trigger for study protocol
      - Forces a violation to happen for reflection
      - violation_type: "screenshot" or "unexpected_viewer"

2. Modify simulation.py to:
   - Check for violation risks after each tick
   - Log violation events
   - Include violations in get_what_happened

3. Create violation_display.py with:

   A. format_screenshot_event(event: ScreenshotEvent) -> str
      ```
      ⚠️ Something happened
      Alex screenshotted your post and sent it to your coworker Jamie.
      
      Jamie is in your "work" context but wasn't in your Close Friends audience.
      Jamie saw your post about feeling overwhelmed.
      
      [Show Jamie's reaction if any]
      ```

   B. format_unexpected_viewer_event(event: UnexpectedViewerEvent) -> str
      ```
      ⚠️ Something unexpected happened
      Your mom saw your post and reacted: 😟
      
      Even though you posted to Friends, she follows you and was checking the app.
      ```

4. Test script:
   - Create post with high-leakiness agent in audience
   - Run multiple ticks, observe screenshot events
   - Force an unexpected viewer event
   - Print formatted outputs

SUCCESS CRITERIA:
- Screenshot events triggered by high-leakiness agents
- Unexpected viewer events more common in no-caps condition
- Events are clearly explained to participant
- Deterministic triggers work for study protocol
- All events logged with full context
```

---

# STAGE 6: Interface & Output Formatting

## Goal
Create the output layer: feed display, post composer prompts, what-happened panel, audience preview.

## Claude Code Instruction

```
CONTEXT:
Backend is complete. Now we create the output layer that presents information to participants. For MVP, this can be terminal/text-based or simple HTML. The key is clear, readable output.

TASK:
1. Create interface.py with:

   A. Class: SimulationInterface
      - Takes a Simulation instance
      - Provides all display methods

   B. Method: display_feed(self, limit: int = 10) -> str
      - Shows recent posts (participant + agents)
      - Format:
        ```
        ═══════════════════════════════════════
        📱 YOUR FEED
        ═══════════════════════════════════════
        
        [You] • 2 hours ago • Close Friends
        "Feeling kind of overwhelmed today"
        ❤️ Alex  👍 Jordan
        💬 Alex: "Aw I'm here for you!"
        
        ───────────────────────────────────────
        
        [Alex] • 4 hours ago • Friends
        "Finally finished my project! 🎉"
        👍 You  ❤️ Sam  ❤️ Taylor
        
        ═══════════════════════════════════════
        ```

   C. Method: display_audience_preview(self, tier: str) -> str
      - Shows who will see a post before posting
      - Format:
        ```
        📤 POSTING TO: Close Friends
        
        This will be visible to:
        • Alex (close friend, school)
        • Jordan (close friend, school)
        • Sam (close friend, school)
        • Taylor (close friend, work)
        + 2 others
        
        Total: 6 people
        Contexts represented: school, work
        
        ⚠️ Note: Taylor is in a different context than most of your close friends.
        ```

   D. Method: display_what_happened(self, post: Post) -> str
      - Full "what happened" panel after posting
      - Include: who saw, reactions, explanation, any warnings

   E. Method: display_condition_comparison(self) -> str
      - Side-by-side comparison of same post under both conditions
      - Used for A/B reflection

   F. Method: display_violation_alert(self, event: Event) -> str
      - Formatted violation event

2. Create session_protocol.py with:

   A. Function: run_setup_phase() -> tuple[list[Agent], dict]
      - Prompts participant to describe key ties
      - Prompts for context sizes
      - Returns (agents, context_sizes)
      
   B. Function: run_initial_reasoning(post_scenarios: list[str]) -> list[str]
      - Presents vignettes before simulation
      - Captures hypothetical reasoning
      - Returns participant responses

   C. Function: run_simulation_phase(sim: Simulation, condition: str, tasks: list[dict]) -> SessionLog
      - Runs structured posting tasks
      - tasks: [{"instruction": "Post something casual", "required_tier": None}, ...]
      - Logs everything
      - Returns full session log

   D. Function: run_reflection_phase(session_log: SessionLog) -> ReflectionData
      - Prompts: "What surprised you?" "What did you notice?"
      - Captures responses

3. Create session_log.py with:
   - Full logging of all interactions
   - Export to JSON for analysis
   - Include: posts, reactions, violations, participant inputs, timestamps

4. (Optional) Create simple HTML output:
   - Function: generate_html_feed(posts: list[Post]) -> str
   - Generates standalone HTML that can be opened in browser
   - More visually appealing than terminal output

SUCCESS CRITERIA:
- Feed is scannable and clear
- Audience preview warns about context mixing
- What-happened explains reactions with agent reasoning
- Session protocol captures all data needed for research
- Logs are complete and exportable
```

---

# STAGE 7: Integration & End-to-End Test

## Goal
Wire everything together, run end-to-end test, fix integration issues.

## Claude Code Instruction

```
CONTEXT:
All components are built. Now we integrate them and run a full end-to-end test simulating an actual study session.

TASK:
1. Create main.py as the entry point:

   A. Command-line interface:
      ```
      python main.py --mode interactive  # For testing
      python main.py --mode session --condition caps  # For study
      python main.py --mode demo  # Quick demo with sample data
      ```

   B. Demo mode should:
      - Use pre-defined sample network (5 key ties, 3 contexts, ~40 agents)
      - Run through 3 posts automatically
      - Show all outputs
      - Take <2 minutes

2. Create end_to_end_test.py:

   A. Test: Full session simulation
      - Setup: 6 key ties, 4 contexts, ~50 agents
      - Condition A (caps):
        - Post 1: Casual content to close_friends
        - Post 2: Vulnerable content to friends
        - Post 3: Public post
      - Switch to Condition B (no caps)
      - Repeat same posts
      - Trigger one violation in each condition
      - Compare outcomes

   B. Test: Verify all logging works
      - Run session
      - Export logs
      - Verify all required fields present

   C. Test: Performance benchmark
      - Time: full tick with 50 agents
      - Target: <30 seconds
      - Identify bottlenecks

3. Create sample_data.py with:
   - Pre-defined key ties for testing:
     ```
     SAMPLE_KEY_TIES = [
         {
             "name": "Alex",
             "relationship": "close friend",
             "description": "My best friend from high school, very supportive but loves gossip",
             "contexts": ["school", "close_friends"]
         },
         {
             "name": "Jordan",
             "relationship": "friend",
             "description": "Friend from work, pretty chill, doesn't post much",
             "contexts": ["work", "friends"]
         },
         # ... 4 more
     ]
     
     SAMPLE_CONTEXTS = {
         "close_friends": 6,
         "school": 25,
         "work": 15,
         "family": 8
     }
     ```

4. Fix any integration issues discovered during testing

5. Create README.md with:
   - Project overview
   - Setup instructions
   - How to run demo
   - How to run study session
   - File structure explanation

SUCCESS CRITERIA:
- Demo runs end-to-end without errors
- All components integrate correctly
- Logs capture everything needed
- Performance is acceptable (<30s per tick)
- README enables someone else to run it
```

---

## File Structure (Target)

```
codesign_bot/
├── config.py                 # Settings, tier caps, API config
├── schema.py                 # Data models (Agent, Post, Reaction, etc.)
├── database.py               # SQLite/storage layer
├── agent_generator.py        # Key tie + filler agent creation
├── tier_manager.py           # Tier assignment, audience logic
├── behavior_generator.py     # LLM reaction generation
├── llm_client.py             # LLM API wrapper
├── violations.py             # Screenshot, unexpected viewer logic
├── simulation.py             # Core simulation loop
├── interface.py              # Display/output formatting
├── session_protocol.py       # Study session flow
├── session_log.py            # Logging for research
├── formatters.py             # Text formatting utilities
├── main.py                   # Entry point
├── sample_data.py            # Test data
├── end_to_end_test.py        # Integration tests
└── README.md
```

---

## Notes for Claude Code Usage

1. **Run stages sequentially.** Each stage depends on previous ones.

2. **Test after each stage.** Don't proceed until current stage works.

3. **For LLM integration:** Start with mock responses, then add real API calls. This lets you test structure before incurring API costs.

4. **If something breaks:** Tell Claude Code what error you're seeing and which stage. It should have full context from the instruction.

5. **Modify as needed:** These instructions are a starting point. If your existing code has good patterns, tell Claude Code to preserve them.
