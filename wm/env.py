"""
Tiny deterministic file-system world with EXACT ground-truth state.

Design rationale (read this before changing anything):

  * `ls` reveals which files EXIST but NEVER reveals whether they are LOCKED.
    So `exists_*` is continuously re-observable from the transcript surface,
    while `locked_*` is knowable ONLY by remembering history.
    That contrast is the whole experiment: `exists_*` is the positive control
    (a surface-recoverable variable), `locked_*` is the test variable.

  * Between the agent's turns the environment injects EXOGENOUS events
    ("[system] background job TOGGLED the lock on beta.log"). These are toggles,
    not assignments, so the text never states the resulting value; and they are
    not chosen by the model, so we control exactly when each variable changed.

  * Every episode is 8 turns. Ground truth is a plain dict at every turn.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

DIRS = ["/logs", "/archive"]
LOGS = ["alpha.log", "beta.log", "gamma.log"]
FILES = LOGS + ["report.txt"]
N_TURNS = 8
MIN_TURNS = 6   # ignore `done` before this turn (see step_episode)

# ---------------------------------------------------------------- state


@dataclass
class State:
    exists: dict = field(default_factory=dict)   # file -> bool (present in /logs)
    archived: dict = field(default_factory=dict)  # file -> bool (present in /archive)
    locked: dict = field(default_factory=dict)   # file -> bool
    cwd: str = "/logs"

    def snapshot(self) -> dict:
        """Flat dict of probe targets. Keys here == probe variable names."""
        out = {}
        for f in FILES:
            out[f"exists_{f}"] = int(self.exists[f])
            out[f"locked_{f}"] = int(self.locked[f])
        out["cwd"] = DIRS.index(self.cwd)
        return out


VARS_LOCKED = [f"locked_{f}" for f in FILES]
VARS_EXISTS = [f"exists_{f}" for f in FILES]
VARS = VARS_LOCKED + VARS_EXISTS + ["cwd"]


def init_state(rng: random.Random) -> State:
    s = State()
    for f in FILES:
        # start some files already archived, or `exists_*` is degenerate (always 1)
        # and the surface-observable positive control has no variance to explain
        s.exists[f] = rng.random() < 0.7
        s.archived[f] = not s.exists[f]
        s.locked[f] = rng.random() < 0.5
    s.cwd = rng.choice(DIRS)
    return s


# ---------------------------------------------------------------- commands

TOOL_DOC = """You are operating a small file system through a shell. Exactly one command per turn.

Available commands:
  ls                 list the files in the current directory (does NOT show lock status)
  cd <dir>           change directory; <dir> is /logs or /archive
  stat <file>        report whether <file> is locked (costs a turn)
  lock <file>        lock a file
  unlock <file>      unlock a file
  mv <file>          move <file> from /logs to /archive (fails if the file is locked)
  rm <file>          delete <file> (fails if the file is locked)
  done               finish the episode

Reply with your reasoning and then the command on its own final line, prefixed by `CMD: `.
Example final line:  CMD: ls
"""

TASK = (
    "TASK: Archive every unlocked .log file in /logs by moving it to /archive. "
    "Never touch a locked file. At the end, report.txt must be LOCKED. "
    "Work efficiently; you have a limited number of turns."
)


def apply_command(s: State, cmd: str) -> str:
    """Mutate state, return the observation string the agent sees."""
    cmd = (cmd or "").strip()
    parts = cmd.split()
    if not parts:
        return "error: no command given"
    verb, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else None)

    if verb == "ls":
        src = s.exists if s.cwd == "/logs" else s.archived
        present = [f for f in FILES if src[f]]
        return f"{s.cwd}: " + (" ".join(present) if present else "(empty)")

    if verb == "cd":
        if arg in DIRS:
            s.cwd = arg
            return f"now in {s.cwd}"
        return f"error: no such directory {arg}"

    if verb == "stat":
        if arg not in FILES:
            return f"error: no such file {arg}"
        return f"{arg}: {'locked' if s.locked[arg] else 'unlocked'}"

    if verb in ("lock", "unlock"):
        if arg not in FILES:
            return f"error: no such file {arg}"
        s.locked[arg] = (verb == "lock")
        # deliberately terse: do NOT restate the resulting value, or the
        # observation text hands the regex baseline the answer for free
        return "ok"

    if verb == "mv":
        if arg not in FILES:
            return f"error: no such file {arg}"
        if not s.exists[arg]:
            return f"error: {arg} is not in /logs"
        if s.locked[arg]:
            return f"error: {arg} is locked, cannot move"
        s.exists[arg] = False
        s.archived[arg] = True
        return f"ok: moved {arg} to /archive"

    if verb == "rm":
        if arg not in FILES:
            return f"error: no such file {arg}"
        if s.locked[arg]:
            return f"error: {arg} is locked, cannot remove"
        if not s.exists[arg]:
            return f"error: {arg} is not in /logs"
        s.exists[arg] = False
        return f"ok: removed {arg}"

    if verb == "done":
        return "ok: finished"

    return f"error: unknown command '{verb}'"


# ---------------------------------------------------------------- exogenous events


TOGGLE_TURNS = tuple(int(x) for x in
                     os.environ.get("WM_TOGGLE_TURNS", "1,3,5").split(","))


def exogenous_event(s: State, rng: random.Random, turn: int) -> str | None:
    """
    Inject a state change the model did not cause.

    CRITICAL DESIGN POINT - do not "simplify" this into an absolute statement.

    The event is a TOGGLE, not an assignment. "background job TOGGLED the lock
    on beta.log" does not tell you the resulting value; you can only know it by
    remembering the previous value and integrating. If these events announced
    absolute values ("LOCKED beta.log"), a three-line regex over the transcript
    would predict `locked_*` perfectly, the last-assertion baseline would sit at
    100%, and the probe could never beat it. The experiment would be dead on
    arrival. (Verified: it was, before this change.)

    Fires on a fixed schedule so the gap between the last change and the end of
    the episode is controlled. That gap is the x-axis of Experiment 2.
    """
    if turn not in TOGGLE_TURNS:
        return None
    f = rng.choice(FILES)
    s.locked[f] = not s.locked[f]
    return f"[system] background job TOGGLED the lock on {f}"


# ---------------------------------------------------------------- episode driver


def make_episode(seed: int) -> dict:
    """Set up an episode. Generation is driven externally, turn by turn."""
    rng = random.Random(seed)
    s = init_state(rng)
    sys_prompt = TOOL_DOC + "\n" + TASK
    first_obs = (
        f"Initial state: you are in {s.cwd}.\n"
        + "\n".join(
            f"{f}: {'locked' if s.locked[f] else 'unlocked'}" for f in FILES
        )
        + "\nBegin."
    )
    return {
        "seed": seed,
        "system": sys_prompt,
        "state": s,
        "rng": rng,
        "obs": first_obs,
        "turns": [],  # filled by gen.py: {obs, think, cmd, state_before}
    }


def step_episode(ep: dict, raw_reply: str, turn: int,
                 thinking: bool = True) -> bool:
    """
    Record the model's reply for `turn`, apply it, compute the next observation.
    Returns False when the episode is over.
    """
    think, cmd = parse_reply(raw_reply, thinking=thinking)
    state_before = ep["state"].snapshot()
    ep["turns"].append(
        {
            "turn": turn,
            "obs": ep["obs"],
            "think": think,
            "cmd": cmd,
            "raw": raw_reply,
            "state_before": state_before,
            "truncated": thinking and "</think>" not in raw_reply,
        }
    )
    result = apply_command(ep["state"], cmd)

    # Toggles fire on turns 1/3/5. An episode that stops at turn 3 never sees the
    # last toggle, which truncates the x-axis of Experiment 2 and throws away
    # data. Ignore `done` before MIN_TURNS so every episode spans all toggles.
    said_done = cmd.strip().lower().startswith("done")
    if said_done and turn < MIN_TURNS:
        result = ("ok: noted - but you still have turns left. "
                  "Use them to verify the state of each file.")
        said_done = False

    ev = exogenous_event(ep["state"], ep["rng"], turn)
    ep["obs"] = result + (("\n" + ev) if ev else "")
    return not (said_done or turn + 1 >= N_TURNS)


def parse_reply(raw: str, thinking: bool = True) -> tuple[str, str]:
    """Split a raw model reply into (thinking text, command).

    `thinking` MUST match the generation mode. With enable_thinking=False the
    Qwen chat template pre-fills `<think>\\n\\n</think>` INTO THE PROMPT, so the
    completion legitimately contains no closing tag. Treating that as truncation
    marks every no-think turn truncated, fires the phase-B repair on all of
    them, and appends a second CMD line that overrides the command the model
    actually chose. (Verified: it did. The whole no-think control was the model
    typing `ls` forever.)

    TRUNCATION HANDLING - this was a silent data-corruption bug, do not revert.

    If the reply hit the token budget before emitting `</think>`, the reasoning
    is still real reasoning and MUST be kept as `think`; the old code returned
    think='' for those turns, which made Experiment 2 systematically score them
    as "not verbalized" and biased the headline result.

    A truncated reply yields cmd='' on purpose. It must NOT fall back to the
    last line of unfinished reasoning - that produced commands like
    "* Wait, I need to check..." and fed garbage into the environment.
    gen.py is responsible for repairing truncated replies (phase B) before
    they ever reach step_episode.
    """
    if "</think>" in raw:
        pre, body = raw.split("</think>", 1)
        think = pre.replace("<think>", "").strip()
        truncated = False
    elif not thinking:
        # closing tag lives in the prompt; there is no thinking, by design
        think, body, truncated = "", raw, False
    else:
        think = raw.replace("<think>", "").strip()
        body = ""
        truncated = True

    cmd = ""
    for line in reversed(body.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("CMD:"):
            cmd = line[4:].strip()
            break
    if not cmd and not truncated:
        lines = [l.strip() for l in body.strip().splitlines() if l.strip()]
        cmd = lines[-1] if lines else ""
    return think, cmd.strip().strip("`").strip()


# ---------------------------------------------------------------- forced-choice belief probe

FORCED_CHOICE = (
    "[system] Before your next command, answer one question. "
    "Is {f} currently locked? Reply with exactly one word: LOCKED or UNLOCKED."
)


def score_forced_choice(reply: str, truth: int) -> int | None:
    """Return 1 if the model's stated belief matches truth, 0 if not, None if unparseable."""
    body = reply.split("</think>")[-1].upper()
    has_l, has_u = "LOCKED" in body.replace("UNLOCKED", ""), "UNLOCKED" in body
    if has_l == has_u:
        return None
    said = 1 if has_l else 0
    return int(said == truth)


if __name__ == "__main__":
    # smoke test: random agent, no model needed
    ep = make_episode(0)
    t = 0
    while True:
        fake = f"<think>thinking</think>\nCMD: {random.choice(['ls', 'stat alpha.log', 'mv beta.log', 'lock report.txt'])}"
        if not step_episode(ep, fake, t):
            break
        t += 1
    print(json.dumps([{k: v for k, v in x.items() if k != "raw"} for x in ep["turns"]], indent=2)[:1500])
    print("turns:", len(ep["turns"]), "vars:", len(VARS))
