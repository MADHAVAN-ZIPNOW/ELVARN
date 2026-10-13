# Data sources and licences

The training data is **not stored in this repository** (it is large, and some of it must keep its own licence and attribution).
This page records where it came from so the work can be repeated and credited correctly.

| Source | What was used | Licence |
|---|---|---|
| Wikipedia (English) | About 5,000 articles on law, government and general topics, collected with `backend/data_tools/collect_wikipedia.py` through the public Wikipedia API (cleaned text, one block per passage, each with its source URL) | CC BY-SA 4.0 |
| Databricks Dolly 15k | Short question/answer pairs (and, for the RAG data, some with a context passage), converted by `backend/data_tools/convert_open_datasets.py` | CC BY-SA 3.0 |
| OpenAssistant OASST1 | A few hundred short English conversations | Apache 2.0 |
| The project's own files | The original domain knowledge and constitution texts the project started with (`backend/data/raw/`) | the owner's own material |

**RAG training examples** (`backend/data_tools/build_rag_dataset.py`) are generated from the passages above: notes, a question and an answer
copied or lightly reworded from the notes. They are therefore derived from CC BY-SA material.

**Model weights:** `deploy/models/17m_rag` was trained on the data above. If you publish the weights or the data, keep the attribution and
the share-alike terms of the CC BY-SA sources.

**Web search results** shown by the chat are fetched live through SearXNG; the app only displays titles, short snippets and links to the original pages.
