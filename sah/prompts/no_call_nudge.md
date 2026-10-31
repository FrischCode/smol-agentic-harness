<!-- The clause is corrective, not suggestive: a turn with no call invites an inference about why
none came, and a word that only suggests ("if you meant to...") leaves that inference standing, so
the model works from a cause the harness never measured. The turn's own wire facts — the finish
word and the completion count the server reported for it, the fields its message step records —
are the correction. `{{evidence}}` always carries the clause and never renders empty: an empty
value would leave the file's space before the period ("off ."), and the floor "— it ended" is a
fact the harness always knows — the turn finished cleanly, uncut and unbroken, which is what the
nudge stands on. When the server reported the wire facts the clause extends to "— it ended at
{tokens_out} tokens with finish={finish}"; when it withheld them the clause stays at the floor.
The numbers are a projection of a measurement, never a guess. The "was not cut off" clause is the
harness's own output-limit machinery, not a wire assertion: a turn the limit ended takes the
continue_* prompts and never reaches this nudge.
The mechanism clause ("Text is never executed — a tool call in text is not a call") is stated as
a rule, never as an assertion about the turn: the harness does not know whether the turn's text
carried a call, in what shape, or how malformed one was — in-prose call shapes differ across
templates and backends and the harness cannot enumerate them — so the nudge states the rule and
the model applies it to its own text, which is in the conversation. A model that wrote a call in
its text believes the call was made, and the flat "no tool call" alone reads as "the call got
lost," which re-emits the same text. "in full" is gone with it: this nudge is only reached on a
turn the output limit did not end, so "in full" framed a truncation that did not happen, and in
the prose case it read as "emit the full text again." -->
Your last turn made no tool call and was not cut off {{evidence}}. Text is never executed — a tool call in text is not a call. If you intended a tool call, make it now. If the task is complete, call verify_objective.
