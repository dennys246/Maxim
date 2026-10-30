- **Every tool now declares what it can cause, and a tool registry can refuse a tool by it** (#1315 item 2; design:
  `docs/plans/atonia_gate_approach_notes.md` §2).
  - New `maxim.tools.base.Effect`, a set of seven flags (`READ`, `SELF_SCOPE`, `WRITE_LOCAL`, `NETWORK`, `COMMS`,
    `ACTUATE`, `GOVERN`; `Effect.ALL` is all of them and cannot be reassigned), plus `effect_of`, `exceeds` and
    `effect_names`. Every built-in tool declares `effect` in its own class body; one that runs a process or
    model-chosen code declares `ALL`. A tool class that declares nothing counts as `ALL`. Declaring `Effect(0)`
    raises `TypeError` when the class is created.
  - `ToolRegistry(ceiling=...)` and `SimToolRegistry(ceiling=...)` refuse, with `maxim.exceptions.ToolEffectRefused`
    and a `tool_effect_refused` WARNING, a tool whose effect exceeds the ceiling and, below `ALL`, any same-name
    replacement. Below `ALL`, `SimToolRegistry.get` of an unknown name raises `KeyError` instead of returning its
    redirect tool. The default ceiling is `ALL`, and no shipped path builds a narrower registry yet.
  - `@maxim.tool(effect=...)` and `maxim.register_tool(tool, effect=...)` accept an optional declaration. Existing
    user tools run under `maxim.run` exactly as before, including one whose class already has its own `effect`
    attribute that is not an `Effect`, or sets `effect` after the class is defined: such a tool counts as `ALL`,
    with one `tool_effect_invalid` WARNING.
  - Under `ALL`, replacing a registered tool with one of a different class (for example a user tool that takes a
    built-in tool's name) now logs a `tool_replaced` WARNING.
  - Entity tool generation (`generate_tools_for_entity`) now registers all of an entity's tools or none, whatever
    stops it, and adds the entity to the `EntityMap` only after its tools register. `create_stub_tool` takes a
    required `effect=`.
