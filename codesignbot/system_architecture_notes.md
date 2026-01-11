# LLM Social Media Simulation: System Architecture Notes

## Purpose of This Document
This document captures all technical architecture decisions made during research planning discussions. It is intended as a starting point for implementation-focused conversations. The architecture is incomplete and requires further refinement.

---

## Key Design Principle: Four-Layer Agent Architecture

Agent state is organized into four layers:

1. **Traits (stable dispositions):** Node characteristics (about the person) and edge characteristics (about relationships). These don't change fast.

2. **Beliefs (learned, update over time):** TPB-style variables—subjective norm estimates, attitude toward posting, perceived behavioral control. Design changes can directly shift these.

3. **State (transient):** Temporary factors—emotional state, recent experiences, current context.

4. **Memory:** Evidence that updates beliefs—what happened when they posted, how others reacted, what they observed.

This separation matters because:
- Some behavior differences are "this person just posts" (trait)
- Some are "I learned this space is unsafe" (belief)
- Some are "today I'm drained" (state)

Blending them makes agents harder to reason about. Keeping them separate lets us model design interventions cleanly (e.g., prompting increases perceived behavioral control; visibility changes norm salience).

Each trait has:
- Natural language anchors (for LLM prompting and participant selection)
- Numerical values (0-1, for quantitative analysis and tracking change over time)

---

## Core Architecture: Three Functions

### 1. Behavior Generation (Per Agent, Bounded Knowledge)
Each agent makes decisions based on what they plausibly know (non-omniscient).

**Input Context (what goes into each decision):**
- Traits: node characteristics (initiation scaffold need, reciprocation need, norm sensitivity, privacy vigilance, etc.) and edge characteristics (tie strength, judgment safety, boundary respect, reciprocity obligation)
- Beliefs: subjective norm estimate, attitude toward posting, perceived behavioral control, beliefs about specific others
- State: current emotional state, recent experiences
- Memory: recent observations, salient past events
- Feedback from previous iteration:
  - Did my last post get reactions? What kind?
  - Did I see others' posts? What happened to them?
  - Any events that affected me?

**Decision Pipeline:**
```
INPUT CONTEXT
    ↓
LLM 1: REASONING
    "Given this context, what is this agent likely to do and why?"
    Output: reasoning trace (natural language)
    ↓
LLM 2: ACTION SELECTION
    "Convert this reasoning into a concrete action"
    Output: action (post X, react Y, self-censor, lurk, etc.)
    ↓
ACTION EXECUTES
    • Post appears (or doesn't)
    • Reaction is logged
    • Other agents can now observe this
    ↓
CONSEQUENCES OCCUR
    • Others react (or don't)
    • Content spreads (or doesn't)
    • Violations happen (or don't)
    ↓
LLM 3: AFTERTHOUGHTS
    "Given what happened, what does the agent now think/feel?"
    Output: reflection, updated beliefs, emotional response
    ↓
UPDATE AGENT STATE
    • Beliefs may shift
    • Memory updated
    • State updated
    ↓
→ NEXT ITERATION (feedback from this iteration feeds into next)
```

**Agent Record (per iteration):**
```
Agent Record (iteration t):
├── Input context summary
│     "Agent was feeling anxious after yesterday's post got no reactions.
│      Believes vulnerability is okay but thinks others judge it.
│      Saw Friend X's vulnerable post get supportive replies."
│
├── Reasoning trace
│     "Considering posting about stress. Worried about classmates seeing.
│      But Friend X got support, maybe it's safe? Still nervous though."
│
├── Action
│     {type: "self_censor", intended: "vulnerable_post", audience: "close_friends"}
│
├── Consequences
│     {observed: null}  // nothing happened because didn't post
│
└── Afterthoughts
      "Feeling relieved I didn't post. But also frustrated.
       Maybe I'll try tomorrow if things seem calmer."
```

### 2. Ecosystem Monitoring (Omniscient, Emergent Patterns)
Tracks platform-level phenomena that no individual agent can see.

**What it tracks:**
- Norm trajectories
- Sentiment climate
- Diffusion patterns
- Harm indicators

**Phenomena it can detect:**
- Pluralistic ignorance (gap between private beliefs and perceived norms)
- Chilling effects (self-censorship rate increases after harm events)
- Norm formation and drift
- Influencer emergence (attention distribution, behavior propagation)
- Spiral of silence
- Echo chambers
- Reputation cascades

### 3. Sense-Making Interface (Youth's Multi-Level Window)
How participants explore the simulation.

**Views:**
1. **Ego-anchored view (default):** "Your feed, your reactions, what happened to you"
2. **Network zoom:** "What's happening in your friend group that you don't see directly"
3. **Platform zoom:** Aggregate patterns, harm hotspots, norm climate
4. **Causal tracing:** Connects individual actions → aggregate patterns → feedback effects → design affordances

---

## State Tracking Architecture

Agent state is organized into four layers: **traits** (stable dispositions), **beliefs** (learned, update over time), **state** (transient), and **memory** (evidence that updates beliefs). This structure follows Theory of Planned Behavior insights: design changes can directly shift attitude/norm/efficacy signals, and separating stable vs. learned vs. momentary factors makes agent behavior easier to reason about and debug.

### Layer 1: Traits (Stable Dispositions)

These are the node and edge characteristics that don't change fast. Each has natural language anchors (for LLM prompting and participant selection) and numerical values (0-1, for quantitative analysis).

#### Node Characteristics (About the Person)

**Core dimensions (rated for all ties):**

**Initiation Scaffold Need:** How much prompting they need to post
- 1 (0.0-0.2): "Almost never posts, even with prompts; mostly observes"
- 2 (0.2-0.4): "Might post once in a while if the prompt feels extremely easy or fun"
- 3 (0.4-0.6): "Prompts reliably get them to post; without prompts they rarely initiate"
- 4 (0.6-0.8): "Posts sometimes on their own, but prompts noticeably increase how often they share"
- 5 (0.8-1.0): "Posts whenever they want; prompts are optional and rarely change behavior"

**Reciprocation Need:** How much they need feedback to keep posting
- 1 (0.0-0.2): "Does not care if nobody reacts; silence feels neutral"
- 2 (0.2-0.4): "Notices low reactions but it barely changes what they do"
- 3 (0.4-0.6): "Wants at least some acknowledgment; repeated silence makes them post less"
- 4 (0.6-0.8): "Strongly wants feedback; low reactions quickly lead to self-censoring or stopping"
- 5 (0.8-1.0): "Highly feedback-dependent; a 'flop' feels embarrassing and can shut them down fast"

**Norm Sensitivity:** How much they calibrate to what others do
- 1 (0.0-0.2): "Ignores the vibe; posts/reacts however they want even if it is unusual"
- 2 (0.2-0.4): "Aware of norms but does not feel compelled to follow them"
- 3 (0.4-0.6): "Usually follows what seems normal, especially with weak ties"
- 4 (0.6-0.8): "Strongly calibrates to what others do; hesitates if norms are unclear"
- 5 (0.8-1.0): "Extremely norm-driven; will not post or engage until they see others doing it"

**Audience Risk / Privacy Vigilance:** How cautious they are about who sees what
- 1 (0.0-0.2): "Posts freely; rarely worries about who sees it or how it spreads"
- 2 (0.2-0.4): "Some caution, but mostly comfortable posting across mixed audiences"
- 3 (0.4-0.6): "Selective; shares personal things only in safer contexts or smaller audiences"
- 4 (0.6-0.8): "High vigilance; assumes things can be judged or spread; self-censors a lot"
- 5 (0.8-1.0): "Maximum caution; treats posting as risky by default and avoids vulnerable content"

**Extended dimensions (rated for key ties, time permitting):**

**Norm Enforcement Style:** How much they police others' behavior
- 1 (0.0-0.2): "Never polices others; avoids judgment; lets people be"
- 3 (0.4-0.6): "Occasionally signals cringe or disapproval, but not intensely"
- 5 (0.8-1.0): "Strong enforcer; quick to shame/mock/pile on, creating a high-risk climate"

**Leakiness:** Tendency to share content beyond intended audience
- 1 (0.0-0.2): "Never screenshots/forwards; treats content as not theirs to share"
- 3 (0.4-0.6): "Occasionally screenshots or recounts things to close friends"
- 5 (0.8-1.0): "Very leaky; regularly saves, forwards, and uses content as evidence or gossip fuel"

**Response Expressiveness:** How much effort they put into reactions
- 1 (0.0-0.2): "Rarely reacts; if they do, it is minimal and generic"
- 3 (0.4-0.6): "Reacts fairly often; sometimes thoughtful, sometimes quick"
- 5 (0.8-1.0): "Highly expressive responder; makes others feel seen with tailored, warm engagement"

**Interpretive Sensitivity:** How much they read into signals
- 1 (0.0-0.2): "Takes signals at face value; assumes benign intent"
- 3 (0.4-0.6): "Sometimes overthinks reactions, views, and silence"
- 5 (0.8-1.0): "Constantly parses intent; avoids many interactions to reduce ambiguity"

**Performance and Validation Orientation:** How much posting feels like reputation management
- 1 (0.0-0.2): "Posts casually; does not worry about how it looks"
- 3 (0.4-0.6): "Edits or overthinks sometimes; audience matters"
- 5 (0.8-1.0): "Treats posting like reputation management; avoids anything that could 'look bad'"

**Metric and Comparison Sensitivity:** How much numbers affect them
- 1 (0.0-0.2): "Barely notices metrics; numbers do not matter"
- 3 (0.4-0.6): "Some comparison; metrics affect mood occasionally"
- 5 (0.8-1.0): "Highly metric-driven; numbers heavily shape content, timing, and willingness to post"

#### Edge Characteristics (About the Relationship)

**Tie Strength:** How close is this relationship?
- 1 (0.0-0.2): "Acquaintance; rarely interact directly"
- 3 (0.4-0.6): "Friendly; talk sometimes; share some things"
- 5 (0.8-1.0): "Very close; share everything; best friend"

**Judgment Safety:** How safe do you feel being vulnerable around them?
- 1 (0.0-0.2): "Feel like they're always evaluating you; very careful"
- 3 (0.4-0.6): "Generally supportive but still careful about some things"
- 5 (0.8-1.0): "Can be fully yourself; they'd never judge you"

**Boundary Respect:** How much do you trust them to keep things contained?
- 1 (0.0-0.2): "Assume they'd share anything; no privacy expectations"
- 3 (0.4-0.6): "Generally respects boundaries but might slip sometimes"
- 5 (0.8-1.0): "Complete trust; would never share without asking"

**Reciprocity Obligation:** How much pressure do you feel to respond to them?
- 1 (0.0-0.2): "No pressure; can ignore without consequence"
- 3 (0.4-0.6): "Some social expectation to respond"
- 5 (0.8-1.0): "Strong obligation; not responding would be noticed and cause issues"

### Layer 2: Beliefs (Learned, Update Over Time)

These are TPB-style variables that get updated based on what agents observe. Design changes can directly shift these.

**Subjective Norm Estimate:** "What do I think people do here? How common/expected is posting?"
- Updated by: seeing others post, seeing reactions to posts, observing what gets ignored vs. engaged

**Attitude Toward Posting:** "Is posting here worth it? Fun vs. performative vs. risky?"
- Updated by: own experiences posting, seeing what happens to others, metric feedback

**Perceived Behavioral Control / Collective Efficacy:** "How doable does it feel? Will others respond supportively?"
- Updated by: past success/failure, seeing supportive vs. judgmental responses, platform affordances

**Beliefs About Specific Others:** "What do I expect from [specific person]?"
- Updated by: direct interactions, observed behavior toward others

### Layer 3: State (Transient)

Temporary factors that affect behavior in the moment:
- Current emotional state (tired, anxious, excited, drained)
- Recent experience (just got embarrassed, just had a win, fighting with someone)
- Current context (at school, at home, in public)

### Layer 4: Memory

Evidence that updates beliefs. Doesn't need to be fancy—short summaries:
- "Last 3 posts got low reactions"
- "Got teased once for oversharing"
- "Saw 5 friends answer the daily question today"
- "Friend X screenshotted something last week"

These experiences shape norms, perceived efficacy, and motivation over time.

### Aggregate State (Platform Level)

**Norm Climate:**
```
- Aggregate private beliefs (e.g., 78% think vulnerability is acceptable)
- Aggregate perceived norms (e.g., 35% think others find vulnerability acceptable)
- Actual behavior rate (e.g., 12% actually post vulnerable content)
→ Gap detection for pluralistic ignorance
```

**Influence Patterns:**
- Attention distribution (whose posts get seen by many)
- Behavior propagation (does behavior spread after X does it?)
- Network centrality metrics

**Health Indicators:**
- Harm event count and type
- Support event count and type
- Self-censorship rate
- Withdrawal indicators
- Chilling effect detection (self-censorship rate before/after harm events)

**Trajectories:**
- Norm drift over time
- Climate shift
- Network evolution

### Bi-Directional Feedback Loop
```
Agents observe platform
    ↓
Infer norms from observations
    ↓
Adjust behavior based on inferred norms
    ↓
Behavior aggregates into patterns
    ↓
Patterns become observable to other agents
    ↓
→ Loop continues
```

---

## Aggregation Pipeline

### Problem
Simply feeding all agent "thoughts" and "beliefs" to an LLM for summarization produces unreliable, non-transparent, non-replicable, unverifiable results.

### Solution: Four-Layer Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  AGENT ENGINE (per agent, per iteration)                           │
│    Input context → Reasoning → Action → Consequences → Afterthoughts│
│    Output: Natural language record + structured action             │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  CODING LAYER                                                       │
│                                                                     │
│  Two parallel tracks:                                               │
│                                                                     │
│  Track A: Structured tagging                                        │
│    - Apply predefined codes (literature + pilot + emergent)        │
│    - LLM-assisted classification into closed categories            │
│    - Output: tags per record                                        │
│                                                                     │
│  Track B: Raw dump + summary                                        │
│    - Collect all natural language outputs                          │
│    - LLM summarizes themes, surfaces uncaptured patterns           │
│    - Output: supplementary summary + suggestions for new tags      │
│                                                                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  COUNTING LAYER                                                     │
│                                                                     │
│  From Track A (tags):                                               │
│    - Count behavioral outcomes                                      │
│    - Count coded themes                                             │
│    - Compute belief distributions                                   │
│    - Detect phenomena (rule-based)                                  │
│                                                                     │
│  From Track B (summaries):                                          │
│    - Flag emergent themes not in coding scheme                     │
│    - Note qualitative patterns for interpretation                  │
│                                                                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  INTERPRETATION LAYER                                               │
│                                                                     │
│  Input:                                                             │
│    - Structured counts (from tags)                                 │
│    - Supplementary summaries (from raw dump)                       │
│    - Event log                                                      │
│    - Trajectories                                                   │
│                                                                     │
│  LLM interprets:                                                    │
│    - Pattern identification                                         │
│    - Phenomenon labeling                                            │
│    - Causal narrative                                               │
│    - Design implications                                            │
│    - Notes where tags vs summaries diverge                         │
│                                                                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  SENSE-MAKING INTERFACE                                             │
│    - Shows counts (verifiable)                                      │
│    - Shows interpretations                                          │
│    - Shows raw examples when youth want to drill down              │
│    - Supports cross-level exploration                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Both Tagging and Raw Summary?

| Situation | Tags alone | Raw summary alone | Both together |
|-----------|------------|-------------------|---------------|
| Need precise counts | ✓ Works | ✗ Can't count reliably | ✓ |
| Need to detect phenomena | ✓ Rule-based on counts | ✗ Unreliable | ✓ |
| Nuance tags miss | ✗ Lost | ✓ Captured | ✓ |
| Emergent patterns | ✗ Constrained by scheme | ✓ Can surface | ✓ |
| Youth can verify | ✓ Can check counts | ✗ Have to trust summary | ✓ |
| Auditable for research | ✓ Replicable | ✗ May vary | ✓ |

### Coding Scheme Evolution

```
Coding Scheme v1 (start of study):
├── Predetermined from literature
│     peer_judgment, context_collapse, self_censorship, etc.
│
└── From pilot data
      timing_sensitivity, observed_example_influence, etc.

        ↓ During iterations, raw summaries surface new patterns

Coding Scheme v2 (mid-study):
├── Original codes
├── Pilot codes
└── Emergent codes added:
      relief_after_censorship, waiting_for_safer_moment,
      authenticity_frustration, etc.

        ↓ Continue refining

Coding Scheme v3 (final):
├── All codes, validated
└── Documentation of what each captures
```

### Example Coding Scheme Categories

**Concerns expressed:**
- peer_judgment
- authority_judgment
- audience_uncertainty
- context_collapse
- screenshot_risk
- specific_agent_avoidance

**Intents:**
- share_vulnerability
- seek_support
- seek_validation
- signal_identity
- test_waters
- maintain_connection

**Outcomes:**
- posted
- self_censored
- modified_content
- posted_to_different_tier

**Influencing factors:**
- recent_negative_experience
- recent_positive_experience
- observed_positive_example
- observed_negative_example
- perceived_norm_uncertainty
- specific_relationship_consideration

**Afterthought sentiment:**
- satisfaction
- dissatisfaction
- relief
- frustration
- regret
- neutral

**Belief state:**
- private_belief[topic]: acceptable/uncertain/risky
- perceived_norm[topic]: acceptable/uncertain/risky

### Phenomena Detection Rules (Examples)

**Pluralistic ignorance:**
```
IF |aggregate_private_belief[topic] - aggregate_perceived_norm[topic]| > threshold
THEN flag pluralistic_ignorance for topic
```

**Chilling effect:**
```
IF self_censorship_rate(post_harm_event) > self_censorship_rate(pre_harm_event) * threshold
THEN flag chilling_effect
```

**Influencer identification:**
```
RANK agents by:
  - views_received
  - behavior_propagation_score (did others copy after they did X?)
  - network_centrality
```

---

## Agent Input Methods

### Approaches

1. **Narrative-based input:**
   Youth describe each key tie through guided prompts:
   - "Tell me about this friend"
   - "How do they usually act on social media?"
   - "What's something they would definitely do? Something they would never do?"
   
   The system infers node and edge characteristics from narratives:
   - "Very supportive, always hypes me up" → response_expressiveness=5
   - "We tell each other everything" → tie_strength=5, judgment_safety=5
   - "Gossips sometimes" → leakiness=4, boundary_respect=2

2. **Dimension rating with behavioral anchors:**
   For each key tie, participants select levels on each dimension using natural language anchors:
   - "How much does this person need prompting to post?" → select from 1 ("Almost never posts") to 5 ("Posts whenever they want")
   - "How cautious are they about who sees their stuff?" → select from 1 ("Posts freely") to 5 ("Maximum caution")

3. **Behavioral exemplars:**
   - "Things my friend would definitely do"
   - "Things my friend would never do"
   - Multiple-choice scenarios with behavioral options

4. **Iterative calibration via contrast pairs:**
   - System generates two candidate agent responses to a scenario
   - Youth select which is more realistic
   - Selection data refines characteristic assignments

5. **Flagging for explanation:**
   - When system generates unexpected behavior, youth explain or reject
   - Explanations provide additional calibration data and may adjust characteristics

6. **LLM inference with validation:**
   - LLM infers node/edge characteristics from narratives
   - Youth validate inferences (e.g., "We think Alex has high response expressiveness and high leakiness—does that sound right?")
   - Youth can adjust using natural language ("Actually she's not that gossipy with my stuff")

7. **Agent audition:**
   - Show 2-3 candidate reactions to sample post
   - Youth select most realistic
   - Built-in credibility measure + calibration

### Filler Agent Characteristics
Filler agents don't go through this process—their characteristics are assigned from relationship template defaults (see Network Generation).

### Alignment with Existing Work
Social Simulacra, Generative Agents, and the 1,000 People study all use natural language descriptions or interview transcripts for agent characterization.

---

## Interface Concept

### Main View (Ego-Anchored)
Feed showing "your" posts, reactions, who saw what.

### Audience Lens
Context-grouped map showing:
- Who is in the audience (by tier and context)
- Who saw the post (attention-driven)
- Bridges between contexts
- Highlighted risky pathways (leaky ties, high-influence nodes)
- Diffusion pathways

### Explanation Panel
Multi-level "Why did [Friend] react this way?"
- **Individual:** "Friend A tends to be supportive. They check the app often."
- **Relationship:** "You and Friend A have high trust."
- **Peer norms:** "In your close friends group, supportive responses to stress posts are common."
- **Platform:** "The 'close friends' tier made this audience feel safe."

### Platform Lens (Zoom Out)
Platform Health Summary showing:
- Norm climate
- Activity patterns
- Harm indicators
- Drill-down capability
- Cross-condition comparison

### Count Inspection View
Youth can inspect counts directly:
```
Vulnerability Climate
Private acceptance:  78%
Perceived acceptance: 35%
Actual posting rate:  12%

⚠️ Pluralistic ignorance detected
Concern about peer judgment: 45 mentions
Saw negative reaction to others: 23 mentions

[See the specific events that shaped this]
```

### Causal Trace
Shows pathway:
```
Individual action (Agent A self-censored)
    ↓
Contributing factors (saw Agent B get snarky comment, peer_judgment concern)
    ↓
Aggregate pattern (28 self-censorship events this iteration)
    ↓
Feedback effects (perceived norm shifts toward "vulnerability is risky")
    ↓
Design factor (audience uncertainty in no-caps condition)
```

---

## Novelty vs. Existing Work

### Existing Systems (SimSpark, Social Simulacra, ABM visualization)
- Researcher-facing god-view
- Agent inspection/debugging tools
- Statistics for researchers
- Purpose: understand/analyze simulation

### Our Approach (Novel Combination)
- Youth participants as primary users
- Ego-anchored frame of reference (inside the simulation)
- Zoom-out designed for participant sense-making, not researcher analysis
- Explicit cross-level causal tracing as core feature
- Socioecological multi-level explanations (individual → dyadic → peer group → platform → external)
- Rigorous aggregation pipeline (code → count → interpret)
- Purpose: support design reasoning and evaluation

---

## Network Model

### Context-First Representation
Instead of only school-based graphs, the network is built from participant-defined contexts (applicable to teens and young adults).

**Example contexts:**
- School (class, grade, club, team)
- Family
- Work
- Dorm/roommates
- Neighborhood/community
- Online-first communities (fandoms, gaming, group chats)

### Multiplex Layers
At minimum, three layers:

1. **Offline proximity/context layer:** Co-membership in contexts (who shares which real-world spaces)

2. **Visible connection layer:** Follow/friend/close-friends tiers (platform-visible relationships)

3. **Private channels layer:** DMs and group chats (modeled as diffusion pathways, not fully revealed transcripts)

### Network Generation

**Ego anchor + scalable expansion:**

1. **Ego network:** 6-10 participant-specified ties (pseudonymous), with narrative descriptions
2. **Context groups:** 3-6 contexts defined by participant; ties assigned to contexts; participant specifies approximate size of each context
3. **Filler agents:** System auto-generates agents to populate each context to specified size, assigning node and edge characteristics based on relationship templates

**Filler agent generation process:**
```
For each context:
  target_size = participant-specified size (e.g., "school" = 30)
  key_ties_in_context = count of described ties in this context
  fillers_needed = target_size - key_ties_in_context
  
  Generate fillers_needed agents using:
    - Relationship template appropriate to context
    - Default node/edge characteristics for that template (see below)
    - Tier assignment based on relationship type
```

**Relationship templates with default characteristics:**

Each template has default trait values (1-5 scale):

```
School:
  "Classmate you recognize but don't talk to":
    Node: initiation_scaffold=1, reciprocation_need=2, norm_sensitivity=3, privacy_vigilance=3, leakiness=1, norm_enforcement=1
    Edge: tie_strength=1, judgment_safety=3, boundary_respect=3, reciprocity_obligation=1
    
  "Acquaintance from a class":
    Node: initiation_scaffold=2, reciprocation_need=3, norm_sensitivity=3, privacy_vigilance=3, leakiness=2, norm_enforcement=2
    Edge: tie_strength=2, judgment_safety=3, boundary_respect=3, reciprocity_obligation=2
    
  "Someone popular you don't know well":
    Node: initiation_scaffold=4, reciprocation_need=4, norm_sensitivity=4, privacy_vigilance=2, leakiness=3, norm_enforcement=3
    Edge: tie_strength=1, judgment_safety=2, boundary_respect=2, reciprocity_obligation=1
    
  "Friend of a friend":
    Node: initiation_scaffold=3, reciprocation_need=3, norm_sensitivity=3, privacy_vigilance=3, leakiness=2, norm_enforcement=2
    Edge: tie_strength=2, judgment_safety=3, boundary_respect=3, reciprocity_obligation=2

Work:
  "Coworker you're friendly with":
    Node: initiation_scaffold=3, reciprocation_need=3, norm_sensitivity=3, privacy_vigilance=3, leakiness=1, norm_enforcement=1
    Edge: tie_strength=3, judgment_safety=4, boundary_respect=4, reciprocity_obligation=3
    
  "Manager/supervisor":
    Node: initiation_scaffold=2, reciprocation_need=2, norm_sensitivity=4, privacy_vigilance=4, leakiness=1, norm_enforcement=2
    Edge: tie_strength=1, judgment_safety=2, boundary_respect=4, reciprocity_obligation=2
    
Family:
  "Extended family at holidays":
    Node: initiation_scaffold=2, reciprocation_need=2, norm_sensitivity=3, privacy_vigilance=3, leakiness=3, norm_enforcement=2
    Edge: tie_strength=2, judgment_safety=3, boundary_respect=3, reciprocity_obligation=2
    
  "Parent's friend who follows you":
    Node: initiation_scaffold=1, reciprocation_need=2, norm_sensitivity=3, privacy_vigilance=2, leakiness=4, norm_enforcement=2
    Edge: tie_strength=1, judgment_safety=2, boundary_respect=2, reciprocity_obligation=1
    
Online community:
  "Mutual you interact with sometimes":
    Node: initiation_scaffold=3, reciprocation_need=3, norm_sensitivity=3, privacy_vigilance=3, leakiness=2, norm_enforcement=2
    Edge: tie_strength=2, judgment_safety=4, boundary_respect=3, reciprocity_obligation=2
    
  "Follower you don't know":
    Node: initiation_scaffold=1, reciprocation_need=2, norm_sensitivity=3, privacy_vigilance=3, leakiness=2, norm_enforcement=1
    Edge: tie_strength=1, judgment_safety=3, boundary_respect=3, reciprocity_obligation=1
```

**Adding variance:**
To avoid all fillers of the same template being identical:
- Randomly vary some characteristics by one level (e.g., low → medium)
- ~10% of fillers get one "notable" characteristic (high leakiness, judgmental, etc.)
- This creates occasional surprises without requiring per-agent customization

**Tier assignment:**
- Close relationship templates → Friends tier
- Distant relationship templates → Followers tier
- Authority figures (manager, older relative) → Followers tier with special handling

4. **Bridge nodes:** Small number of agents belong to multiple contexts; these drive context collapse and diffusion. System identifies natural bridges:
   - Key ties already in multiple contexts (participant-specified)
   - Auto-generate 2-3 filler bridges (e.g., "family friend who's also a coworker")

**Total population target:** 50-100 agents
- 6-10 key ties (rich descriptions)
- 40-90 filler agents (templates + default characteristics)
- Enough for realistic audience dynamics without overwhelming complexity

**Network size for design manipulation:** For connection caps to matter, participants' networks must exceed the cap sizes. If the Close Friends cap is 15, there should be 25+ potential close friends in the network, forcing selection decisions.

### Network Regimes (Optional Experimental Factor)
To capture platform differences (e.g., Reddit-like vs Instagram-like experiences):

- **Tight-knit clustered:** High mutuals, dense connections within contexts
- **Hub-and-spoke influence:** Attention inequality, few high-influence nodes
- **Context-collapsed mixed:** High overlap across contexts
- **Fragmented interest-based:** Multiple weakly connected clusters

These can be participant-selected (ecological validity) or experimentally assigned (causal leverage).

---

## Violation Events

### Why Violations Are Necessary
Privacy and safety in youth/YA contexts often hinge on networked events: leakage, resharing, screenshotting, misinterpretation, and group escalation. Without raised violation events, participants may not be able to reason about harm pathways or recovery strategies.

### Violation Event Library

Events are represented as structural facts and consequences, not lurid content:

1. **Unintended audience expansion:** Post reaches a different context than intended
2. **Screenshot/reshare occurred:** Binary event with pathway trace (who took it, where it went)
3. **Private message forwarded:** Raised without showing raw forwarded content
4. **Pile-on initiation:** Multiple negative or teasing responses cascade
5. **Exclusion signal:** Being left out of a tier or group becomes socially meaningful
6. **Misinterpretation spread:** Content is reframed/misunderstood as it travels
7. **Authority discovery:** Content reaches parents, teachers, employers

### Participant-Facing Presentation

Participants see:
- That a violation occurred
- Diffusion pathway (how it traveled, via which bridge/tie)
- Aggregate backchannel signals (counts and categories, not raw gossip)
- Relationship/norm consequences (trust shifts, increased silence, increased self-censorship)

Participants do NOT see:
- Detailed humiliating transcripts
- Lurid descriptions of harm
- (Optional: sanitized paraphrases available only if ethically justified and participant-controlled)

### Violation Triggering

Options (to be decided):
- **Probabilistic:** Based on agent leakiness characteristics and content sensitivity
- **Deterministic:** Pre-planned events inserted at specific points
- **Participant-controlled:** Opt-in to "what if this leaked?" scenarios
- **Hybrid:** Some guaranteed events + some probabilistic

---

## Interface Details

### Three-Pane Layout

**Pane 1: Feed**
A realistic but minimal feed showing:
- Posts (own and others')
- Likes, comments
- Private replies (summarized, not full transcripts)
- Time indicators

**Pane 2: Audience and Network Lens**
Context-grouped map showing:
- Who is in the audience (by tier and context)
- Who saw the post (attention-driven visibility)
- Bridges between contexts
- Highlighted risky pathways (leaky ties, high-influence nodes)
- Diffusion animation (how content spread)

**Pane 3: Why Panel + Norms/Safety Summaries**
- Action explanations grounded in structured state
- Norm summary (what is rewarded/punished in this climate)
- Safety summary (leak risk, audience uncertainty, diffusion beyond intended audience)
- Drill-down to specific events and counts

### Key Features Supporting Causal Reasoning

1. **Time compression:** Fast-forward minutes/hours/days to observe norm drift and diffusion

2. **Replay:** Revisit a marked moment to prompt reflection; see what led to it

3. **Counterfactual A/B:** Re-run the same post under the other condition (same initial state, same random seed) to support direct comparison

4. **Moment tagging:** One-click "mark this moment" with tags:
   - Supportive
   - Unsafe
   - Embarrassing
   - Drama
   - Authentic
   - Surprising
   
   Used later for replay interviewing

5. **Condition switching:** Toggle between Caps and No Caps conditions to compare

### Multi-Level Explanation Structure

When participant clicks "Why did [Agent] do this?":

**Level 1 - Individual:**
"Agent A tends to be supportive (trait). They check the app often (attention). They were in a good mood (state)."

**Level 2 - Relationship:**
"You and Agent A have high trust. You're in the same close friends tier. They've been supportive to you before (history)."

**Level 3 - Peer Norms:**
"In your close friends group, supportive responses to stress posts are common. Agent A has observed this pattern."

**Level 4 - Platform Design:**
"The 'close friends' tier created a smaller, more coherent audience. Agent A knew exactly who else would see their response."

**Level 5 - Aggregate Climate:**
"Overall, vulnerability is being rewarded on the platform right now (12 supportive responses to vulnerable posts in last iteration, 2 negative)."

---

## Open Questions / To Be Decided

1. **Simulation tick structure:** How does time advance? Turn-based? Continuous? Event-driven?

2. **Violation event triggering:** Random? Deterministic based on conditions? Participant-controlled?

3. **Platform mechanics to simulate:** Feed algorithm? Notification system? What's the minimal feature set?

4. **Belief update mechanics:** How do agents update their perceived norms based on observations? What's the learning rate?

5. **Memory architecture:** How much history do agents retain? How is salience determined?

6. **Content generation:** How do we generate realistic post content without generating harmful content? What guardrails?

7. **Calibration protocol details:** How many contrast pairs per agent? What scenarios?

8. **Inter-agent relationships:** Do we model agent-to-agent relationships beyond agent-to-participant? How complex?

9. **Implementation stack:** What technologies? How to balance research flexibility with participant-facing polish?

---

## Implementation Notes

### Full Population Simulation + Reactive Radius Display
Simulate 50-100 agents total:
- 6-10 key ties (participant-described, rich narratives)
- 40-90 filler agents (auto-generated from templates + default characteristics)

Display strategy:
- Primary display: Key ties + filler agents who actually engage (react, comment)
- Secondary display: Aggregated signals from rest of population
  - "12 others saw this post"
  - "Your post reached your school context"
  - Norm summaries
  - Diffusion counts

**Rationale:** Aggregate dynamics require sufficient population size, but participants can't process 100 individual agent interactions. Key ties get detailed treatment; filler agents provide realistic backdrop and occasional surprises (distant acquaintance reacts unexpectedly).

### Non-Omniscience Design

**For agents:**
- Agents have partial views (only see posts in their feed/tier)
- Agents have biased beliefs (infer norms from limited observations)
- Agents don't know others' private beliefs
- Agents don't know full diffusion pathways

**For participants:**
- Participants receive inspection lenses that reveal MORE than agents see (audience map, diffusion traces)
- But not raw access to all private content
- Preserves reasoning scaffold without omniscient voyeurism

### Structured Explanation Layer Implementation

**Critical:** The "why" panel must be implemented as a lookup into logged decision factors, NOT as an LLM call.

**Process:**
1. When agent takes action, system logs which factors (and their values) contributed
2. Logged factors include: characteristic values consulted, beliefs accessed, state considered, relationship attributes used
3. Explanation shown to participants is generated by templating these logged factors into natural language
4. Templates are pre-written, not LLM-generated at display time

**Why this matters:**
- Ensures explanations are faithful to actual decision process
- Prevents LLM confabulation (inventing plausible but inaccurate reasons)
- Allows researchers to audit explanation accuracy
- Makes explanations consistent across sessions

### Content Generation Guardrails

**Post content generation:**
- LLM generates post text given: topic category, emotional tone, intended audience, agent characteristics
- Content moderation filter applied before display
- Severity bounds enforced (no graphic content, no specific names/details that could feel too real)
- Youth-appropriate vocabulary and scenarios

**Violation content:**
- Violations are raised as structural events, not depicted content
- "A screenshot was taken and shared to [context]" — not the content of the screenshot
- Consequences shown through aggregate signals, not detailed transcripts

### Latency Considerations

**Per-iteration latency budget:**
- Key tie reasoning: LLM call with full context (~1-3 seconds per agent)
- Filler agent reasoning: Simpler prompt, can often use cheaper/faster model or rule-based shortcuts
- With 50+ agents: potentially 50-150 seconds if sequential
- **Mitigation:** 
  - Parallel agent processing
  - Filler agents: batch processing, simpler prompts, or probabilistic shortcuts (most lurkers just lurk)
  - Caching common responses
  - Pre-computation of likely reactions

**User-facing latency:**
- "Post" action should feel responsive (<2 seconds to initial feedback)
- Fast-forward can take longer (5-10 seconds acceptable with loading indicator)
- Explanation panel: should be instant (lookup, not generation)

### Data Storage Requirements

**Per-session storage:**
- Full agent state snapshots per iteration
- All natural language outputs (reasoning, afterthoughts)
- All coded tags
- All counts and trajectories
- Participant interaction logs

**Estimated size:** 
- ~10-50 MB per participant session (depending on iteration count and agent population)
- Needs to support replay and analysis

---

## Dependencies and Constraints

- LLM API costs (multiple LLM calls per agent per iteration)
- Latency for real-time interaction
- Content moderation requirements
- Data storage for logs and state
- Youth-appropriate interface design
- Session length constraints (~60 min)

---

## Next Steps for Architecture Refinement

1. Define minimal viable simulation loop
2. Specify LLM prompt templates for each stage
3. Design coding scheme v1 (from literature)
4. Prototype single-agent decision cycle
5. Define aggregate phenomena detection rules
6. Design interface wireframes
7. Plan pilot study for calibration and coding scheme validation
