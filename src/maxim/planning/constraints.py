

class ConstraintViolation(Exception):
    pass


class ConstraintSet:
    def check(self, plan, state):
        steps_taken = int(getattr(state, "steps_taken", 0) or 0)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps and steps_taken > max_steps:
            raise ConstraintViolation("Max steps exceeded")
