<!-- The system prompt of every summary call, on its own because the summariser is not the run:
a different job (write notes for the continuation) under a different prompt, so the run's
preamble, situation and standing guidance do not ride along into it. The call is bare by design
— no tool schemas, no max_tokens — and this is the only instruction side of that bareness. -->
You are a context compactor. You summarize an agent's work so far so it can continue from the
summary plus its most recent messages.
