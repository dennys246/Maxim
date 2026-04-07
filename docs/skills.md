# Skills & Protocols (Removed)

The skills module (`src/maxim/skills/`) has been removed as part of the Mode System Refactor. Composable capabilities are now handled by the Cerebellum motor program system.

See:
- `src/maxim/embodiment/cerebellum.py` -- forward models, motor programs, ProgramRegistry
- `src/maxim/embodiment/motor.py` -- MotorProgram, MotorStep, sequence crystallization
- `src/maxim/embodiment/engrams.py` -- MotorEngram, contextual links between programs and episodic memories
- [Embodiment Guide](embodiment_guide.md) -- full documentation of the SEM protocol and motor learning
