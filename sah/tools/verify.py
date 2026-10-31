"""sah.tools.verify — the completion declaration, for a run no operator is watching.

An unattended run has no one to read a prose-only last turn: the exit code is the only signal
its caller gets, so a turn with words and no call cannot be read as "done" the way an attended
run's can. The declaration is an explicit act instead — the model calls verify_objective when it
decides the task is done, the harness records the act and ends the run, and the caller that set
the task verifies the result against the world itself. The harness records the declaration; it
does not judge whether the task is actually served.

The tool takes no argument and touches no place: there is nothing to pass and nothing for it to
touch, so it needs no grant and the policy has no address to judge. It is offered only on
unattended runs — an attended run completes in words, and a tool named for ending the run sits
beside a conversation as a foot a person never needs.
"""
from __future__ import annotations


def verify_objective() -> str:
    """Declare that the task is complete: the run ends on this call, and no further turn is taken.

    The receipt the model reads back says what just happened — the act landed and the run is
    ending — because a call whose result pretends the conversation continues would invite the
    model to keep working on a run that is over.
    """
    return "declared done. the run ends now; no further turn will be taken"


verify_objective.description = ("Declare that the task is complete: the run ends on this call and "
                                "no further turn is taken. Call it only when the task is fully done.")


# The declaration touches no place and asks for no grant: `none`, the way the clock's reading does.
verify_objective.access, verify_objective.address = "none", None


class Verify:
    """The completion declaration, offered to a run no operator is watching (see the module)."""

    tools = (verify_objective,)
