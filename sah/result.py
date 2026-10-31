"""sah.result — the answer a run left, projected from its log.

The one question a reader of a finished (or finishing) run asks: what did it answer, and how
did it end. This is a projection over the same `log.jsonl` every other consumer reads — the
answer is the last assistant message that says something, the ending is the last `ended`
step — so it can never disagree with the record: it re-derives nothing, it selects what the
record already says. A run that ended with an answer has one; a run that spent its budget on
a tool call ends with the budget's word and no answer, and that emptiness is the honest shape
of "the run is over and did not deliver", which is a different fact from "the run is still
going" (no `ended` step yet). The distinction a parent making a decision on the child's
answer needs is exactly the one the record draws.

One projection gap this closes: when the model's last turn left no words, the answer (if any)
is stale by definition — it came from an earlier turn — and the words that happened after it
(budget spent, completion declared, the run's own notices) are what tell a reader how the run
actually ended. Those are carried beside the answer as `tail_notices`, and only in that case:
a run whose last turn said its answer has nothing to add, and a projection that appended its
notices anyway would narrate a delivery that was not stale."""
from __future__ import annotations


def result(steps) -> dict:
    """The run's answer and its ending, read from its steps.

    `text` is the last assistant message that says something — the answer the run left — and
    `answer_id` the log step it came from, so a reader can point at it in the log rather than
    at a copy. Both are None while there is none: a run whose last turn was a tool call has
    not delivered, and an empty answer must not be dressed up as a delivered one. `ended` is
    False while the log carries no `ended` step — a run still running, or one that died
    writing its closing step, which the reader's damage note (carried beside this projection
    by its caller) is the witness to. `run_id` comes from the run's own `started` step, or is
    empty for a log that has none, which the caller fills from the folder name.

    `tail_notices` is empty unless the run ended and the model's last turn left no words — the
    case where the answer, if any, is stale by definition. Then it holds the text of every
    notice step that comes after the answer step (all of them when there is no answer): the
    budget's word, the completion's declaration, whatever the record said between the last
    words and the ending. A reader of the answer alone cannot tell those apart from silence; a
    reader who mistakes a mid-build fragment for a delivery is a reader the record failed."""
    ended = [step for step in steps if step["kind"] == "ended"]
    answer = next((step for step in reversed(steps)
                   if step["kind"] == "message" and step.get("role") == "assistant"
                   and (step.get("text") or "").strip()), None)
    last_turn = next((step for step in reversed(steps)
                      if step["kind"] == "message" and step.get("role") == "assistant"), None)
    run_id = next((step.get("run_id", "") for step in reversed(steps)
                   if step["kind"] == "started"), "")
    projection = {"run_id": run_id, "ended": bool(ended), "reason": None,
                  "exit_code": None, "text": None, "answer_id": None, "tail_notices": []}
    if ended:
        projection["reason"] = ended[-1]["reason"]
        projection["exit_code"] = ended[-1]["exit_code"]
    if answer is not None:
        projection["text"] = answer["text"].strip()
        projection["answer_id"] = answer["id"]
    if ended and last_turn is not None and not (last_turn.get("text") or "").strip():
        answer_pos = (next(i for i, step in enumerate(steps) if step is answer)
                      if answer is not None else -1)
        projection["tail_notices"] = [step["text"] for i, step in enumerate(steps)
                                      if step["kind"] == "notice" and i > answer_pos]
    return projection
