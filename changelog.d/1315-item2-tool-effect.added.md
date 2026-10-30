- **Every tool now declares what it can cause, and a tool registry can refuse a tool by it** (#1315 item 2; design:
  `docs/plans/atonia_gate_approach_notes.md` §2).
  - New `maxim.tools.base.Effect`, a set of seven flags (`READ`, `SELF_SCOPE`, `WRITE_LOCAL`, `NETWORK`, `COMMS`,
    `ACTUATE`, `GOVERN`; `Effect.ALL` is all of them), plus `effect_of`, `exceeds` and `effect_names`. Every
    built-in tool declares `effect` in its own class body. A tool that runs a process or model-chosen code declares
    `ALL`. A tool class that declares nothing counts as `ALL`. Declaring `Effect(0)` or a non-`Effect` value raises
    `TypeError` when the class is created.
  - `ToolRegistry(ceiling=...)` and `SimToolRegistry(ceiling=...)` refuse, with `maxim.exceptions.ToolEffectRefused`,
    a tool whose effect exceeds the ceiling, a tool whose class `effect` was reassigned after creation, and, below
    `ALL`, any same-name replacement. Below `ALL`, `SimToolRegistry.get` of an unknown name raises `KeyError` instead
    of returning its redirect tool. The default ceiling is `ALL`, so nothing changes for `maxim.run` or any shipped
    run: no built-in path builds a narrower registry yet.
  - `@maxim.tool(effect=...)` and `maxim.register_tool(tool, effect=...)` accept an optional declaration. Without
    one a user tool is `ALL` and runs exactly as before; a narrower registry drops it with a WARNING naming the tool
    and the ceiling.
  - Under `ALL`, replacing a registered tool with one of a different class now logs a `tool_replaced` WARNING.
  - Entity tool generation (`generate_tools_for_entity`) now registers all of an entity's tools or none, and adds
    the entity to the `EntityMap` only after they register.
