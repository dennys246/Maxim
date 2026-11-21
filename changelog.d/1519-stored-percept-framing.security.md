- **Stored outside text reaches a prompt only inside an untrusted-content frame** (#1519, #1516 F15): bio-enrichment's
  working-memory recap now frames a stored percept's text by its provenance (`runtime/loop_perception.py::
  stored_percept_text`: the operator's own `cli`/`voice`/`speech`/`transcript` words are defanged, every other source,
  inbound `comms:*` included, is framed external), so it can no longer reach the enrichment section, working-memory
  thoughts or the deliberation transcript bare. A recalled episode's goal and the ATL names under "Related concepts"
  are framed too. The Dormant renderers (PromptBuilder's memory sections, `ExecAgent._build_llm_context`) frame
  their stored text, and the latter no longer shows an inbound message's sender.
