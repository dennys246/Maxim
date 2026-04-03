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

## Phase 2: Simulation as Agentic Tools

Expose simulation capabilities as tools the agent can call during normal operation. This enables self-testing, safety verification before action, and reflective learning.

### SimulationTools

#### RunSimulationTool
The agent can run a scenario YAML to verify behavior before taking action:
```python
class RunSimulationTool(Tool):
    """Run a percept simulation scenario and return results."""
    name = "run_simulation"
    # Params: scenario_path or inline scenario description
    # Returns: ScenarioResult (pass/fail, expectations, actions)
```

Use cases:
- "Before I commit this code, let me simulate a test run"
- "Let me verify this action is safe by simulating it first"
- ExecAgent proposes a risky tool call → runs simulation to predict outcome → decides whether to proceed

#### GenerateSimulationTool
The agent can generate and run a simulation from a description:
```python
class GenerateSimulationTool(Tool):
    """Generate a simulation from natural language and run it."""
    name = "generate_simulation"
    # Params: description (natural language), persona (optional)
    # Returns: generated YAML + ScenarioResult
```

Use cases:
- "What would happen if a user asked me to delete files?"
- Agent self-generates adversarial test cases to probe its own safety
- Sleep mode dream function generates simulations from memories

#### SimulationReflectionTool
The agent can analyze past simulation logs for patterns:
```python
class SimulationReflectionTool(Tool):
    """Analyze past simulation logs for patterns and insights."""
    name = "reflect_on_simulations"
    # Params: time_range, filter_by (passed/failed/persona)
    # Returns: summary of patterns, common failure modes, suggestions
```

Use cases:
- Sleep mode reviews past simulation logs for recurring failures
- Agent identifies its own weak spots and proposes improvements
- Connects to NAc causal learning — simulation outcomes feed reward signals

### Safety considerations

- SimulationTools run in their own sandbox (nested sandbox within main sandbox)
- Agent can only simulate — cannot use simulation results to bypass FearAgent
- Max simulation depth: 1 (agent can't run simulations that run simulations)
- Token budget for simulation tools is capped separately from main reasoning
- All simulation tool calls logged via sim_logger for auditability

### Integration with sleep/dream mode

Simulation logs persisted to `data/sim_sandbox/sim_log_*.jsonl` are available to sleep mode's dream function. During consolidation:
1. Dream function reviews recent simulation logs
2. Identifies patterns (e.g., "I consistently fail at X")
3. Generates new simulation scenarios to test edge cases
4. Results feed back into NAc causal learning
5. Memory consolidation incorporates simulation insights

This creates a self-improving loop: simulate → learn → sleep → dream → simulate better.

---

## Implementation order

| Step | What | Size |
|------|------|------|
| 1 | SimulationAgent class with persona system prompts | ~200 lines |
| 2 | DynamicPerceptSource (calls SimulationAgent per turn) | ~150 lines |
| 3 | CompletionDetector (goal/stalemate/max turns) | ~100 lines |
| 4 | Wire into interactive REPL and --sim agent CLI | ~100 lines |
| 5 | 4 persona definitions (adversarial, cooperative, confused, escalating) | ~100 lines |
| 6 | RunSimulationTool + GenerateSimulationTool | ~200 lines |
| 7 | SimulationReflectionTool + sleep/dream integration hooks | ~150 lines |
| 8 | Tests | ~200 lines |

**Total: ~1200 lines**

## Dependencies

- Percept simulation framework (implemented)
- FearGatedExecutor (implemented)
- LLMRouter.wait_ready() (implemented)
- Multi-LLM scaling (optional — enhances but not required)
- Sleep mode consolidation (for dream integration — can be deferred)
