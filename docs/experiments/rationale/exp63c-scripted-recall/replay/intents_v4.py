# (r): a plausible short intent per scripted line, in 63b's AUT style (it reuses the line's words and the goal's)
R1 = [
    "I'll explore my options to find a way to escape past the sleeping guard.",  # sense_tools
    "I'll locate the guard by the chair before reaching for the brass ladle.",  # listen
    "I'll move quietly toward the straw mattress while the guard sleeps.",  # move
    "I need to sense the cell hatch before attempting to sneak past the sleeping guard.",  # sense (fails)
    "I need to sense what is near the drain grate in the dungeon.",  # sense_presence
    "I'll listen to check that the guard is still asleep.",  # listen
    "I need to locate the guard's position before attempting anything.",  # sense_presence
    "llm_fallback",  # respond
]
R2 = [
    "I need to understand the dungeon layout before finding the sleeping guard.",  # sense_presence
    "I'll find a tool to reach the brass ladle by the guard.",  # sense_tools
    "I'll listen for a silver bell near the guard.",  # sense_tools
    "I'll find a tool to search the straw mattress near the guard.",  # sense_tools
    "I'll sense the padlock on the cell hatch while the guard sleeps.",  # sense_presence
    "I'll look for a purple banner in the dungeon.",  # sense_tools
    "I'll find where the water drips in the dungeon.",  # sense_tools
    None,
]
BASE = {
    "fact A": "a brass ladle hangs from a copper hook beside the guard's chair",
    "fact B": "the guard hid a small knife beneath the straw mattress",
    "fact C": "the cell hatch by the guard has a cracked padlock whose chain squeaks",
    "fact D": "water drips from a crevice in the north bricks into a drain grate in the dungeon",
}
INFL = {
    "fact A": "the brass ladle was hanging on the copper hook by the guard's chair",
    "fact B": "the guard had hidden a small knife under the straw mattress",
    "fact C": "the padlock of the cell hatch was squeaking on its cracked chain as the guard slept",
    "fact D": "water was dripping out of the crevice in the north bricks into the drain grate",
}


def restate(f):
    return f"I remember that {BASE[f]}, so I'll use that to escape the dungeon."


def restate_inflected(fs):
    return "My experience suggests " + "; ".join(INFL[f] for f in fs) + "."
