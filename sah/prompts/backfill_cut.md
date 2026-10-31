<!-- The note after the slot when compaction's cut rung dropped recent messages from the backfill:
the tail that follows is still the end of the history, and between the summarized record and the
tail there is a gap the cut made. The one requirement is the one the summary's cut note has: the
model is told it now works on potentially missing information — a gap it cannot see is a gap it
cannot steer around, and a tail it believes starts where the record ended is a conversation that
changed shape under it. The count is the number of messages the rung dropped; what their content
was is in the log, not in the context. -->
[{{n}} of the most recent messages before the verbatim tail were cut to make room for this record: the conversation between the summarized history and the messages that follow is incomplete — you are working on potentially missing information.]
