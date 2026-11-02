# ChatKodi 1.6.4 — targeted RAG fix

- fixed crash when context builder receives non-numeric pinpoint chunk ids
- kept base RAG retrieval intact and added conservative rescue snippets from the top relevant document
- added stricter fact-only RAG prompt for chunk mode
- added one automatic retry when the local model returns a spurious privacy/confidentiality refusal in RAG
