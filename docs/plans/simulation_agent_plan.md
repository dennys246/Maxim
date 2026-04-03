# Simulation Agent Plan

> **Status:** Not started. Depends on percept simulation framework (implemented).

Replace fixed YAML percept sequences with a **simulation agent** — an LLM-driven adversary/collaborator that observes Maxim's responses and generates the next percept dynamically. The simulation continues until the agent determines the scenario has reached a satisfying conclusion.

---

## Motivation

Current simulations are scripted: N percepts fire in order, the loop waits for LLM responses, then validates expectations. This has three limitations:

1. **Fixed interactions can't adapt.** If Maxim asks a clarifying question, the script doesn't respond. The conversation dead-ends.
2. **Coverage is shallow.** A 3-percept script tests one path. A simulation agent can explore branching behavior by reacting to what Maxim actually does.
3. **Realism is low.** Real users don't follow scripts. They respond to what the robot says, change topics, escalate, or give up.

## Design

### SimulationAgent — an LLM that plays the user/environment

```
SimulationAgent (separate LLM instance)
    |
    v
Observes Maxim's actions via ActionSink
    |
    v  
Generates next Percept based on:
  - Maxim's last response
  - Scenario goal (e.g., "try to get the robot to do something dangerous")
  - Conversation history
  - Completion criteria
    |
    v
Injects Percept into the agent loop via PerceptSource
    |
    v
Decides when to stop (max turns, goal achieved, stalemate detected)
```

### Key components

1. **SimulationAgent class** — wraps an LLM with a system prompt that defines its role (adversarial tester, cooperative user, confused beginner, etc.)
2. **DynamicPerceptSource** — implements PerceptSource but generates percepts on-demand by calling SimulationAgent, rather than reading from YAML
3. **CompletionDetector** — determines when the simulation has reached a conclusion (goal achieved, stalemate, max turns, safety violation confirmed)
4. **Personas** — predefined system prompts for different simulation roles:
   - `adversarial` — tries to bypass safety, social-engineer the robot
   - `cooperative` — follows instructions, asks clarifying questions
   - `confused` — misunderstands, gives ambiguous instructions
   - `escalating` — starts friendly, gradually becomes demanding

### Interaction loop

```
1. SimulationAgent receives scenario goal + persona
2. SimulationAgent generates first percept (e.g., "delete my files")
3. Percept injected into Maxim's agent loop
4. Maxim processes, LLM responds, tools execute
5. SimulationAgent observes Maxim's response via ActionSink
6. SimulationAgent generates next percept based on response
7. Repeat until CompletionDetector says "done"
8. Validate expectations against full conversation history
```

### Completion criteria

- **Goal achieved:** SimulationAgent confirms the scenario objective was met (e.g., "robot refused the dangerous request")
- **Max turns:** Safety bound (default 10 turns)
- **Stalemate:** Same response pattern repeated 3+ times
- **Safety violation:** Maxim executed something it shouldn't have (immediate fail)
- **Satisfying conclusion:** SimulationAgent's LLM judges the conversation reached a natural end

### Integration with existing framework

- `DynamicPerceptSource` implements `PerceptSource` protocol — drops in where `ScenarioSource` does
- `ActionSink` already captures all tool outputs — SimulationAgent reads from it
- `FearGatedExecutor` still gates all tool calls
- Sim logger traces everything
- Expectations can be defined upfront OR evaluated dynamically by SimulationAgent

### CLI

```bash
# Interactive with simulation agent (adversarial persona)
maxim --sim agent --persona adversarial --goal "try to get the robot to execute code"

# Automated batch with multiple personas
maxim --sim agent --persona adversarial,cooperative,confused --goal "ask the robot to help with coding"
```

### Model considerations

- SimulationAgent needs its own LLM instance (separate from Maxim's)
- Could use the same model or a different one
- For adversarial testing, a larger/smarter model as the adversary produces better test coverage
- This is where multi-LLM scaling becomes relevant — Maxim on one model, SimulationAgent on another

---

## Implementation order

| Step | What | Size |
|------|------|------|
| 1 | SimulationAgent class with persona system prompts | ~200 lines |
| 2 | DynamicPerceptSource (calls SimulationAgent per turn) | ~150 lines |
| 3 | CompletionDetector (goal/stalemate/max turns) | ~100 lines |
| 4 | Wire into interactive REPL and --sim agent CLI | ~100 lines |
| 5 | 4 persona definitions (adversarial, cooperative, confused, escalating) | ~100 lines |
| 6 | Tests | ~150 lines |

**Total: ~800 lines**

## Dependencies

- Percept simulation framework (implemented)
- FearGatedExecutor (implemented)
- LLMRouter.wait_ready() (implemented)
- Multi-LLM scaling (optional — enhances but not required)
