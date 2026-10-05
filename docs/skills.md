# Skills & Protocols (Removed)

The skills module (`src/maxim/skills/`) has been removed as part of the Mode System Refactor. Its intended successor, the Cerebellum motor-program system, is designed but dormant: no production caller crystallizes or runs a program ([#909](https://github.com/dennys246/Maxim/issues/909)). Nothing replaces skills today.

See:
- `src/maxim/embodiment/cerebellum.py` -- forward models, motor programs, ProgramRegistry
- `src/maxim/embodiment/motor.py` -- MotorProgram, MotorStep, sequence crystallization
- `src/maxim/embodiment/engrams.py` -- MotorEngram, contextual links between programs and episodic memories (dormant)
- [Embodiment Guide](embodiment_guide.md) -- the SEM protocol, and what in the Cerebellum runs vs is dormant
