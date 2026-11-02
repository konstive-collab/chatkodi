# ChatKodi 1.6.3 — conservative RAG rescue fix

- Rolled back aggressive RAG context replacement from 1.6.2 behavior.
- RAG retrieval pool widened from 5 to a safer dynamic range (12..24).
- Added minimal rescue snippets from already relevant documents to help LLM see answer chunks without degrading existing answers.
- Improved RAG prompts for yes/no questions with morphology-aware guidance.
- Added logs for rescue-context activation.
