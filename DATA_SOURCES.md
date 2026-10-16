# Data sources and licences

The training data is **not stored in this repository** (it is large, and some of it must keep its own licence and attribution).
This page records where it came from so the work can be repeated and credited correctly.

| Source | What was used | Licence |
|---|---|---|
| Wikipedia (English) | About 5,000 articles on law, government and general topics, collected with `backend/data_tools/collect_wikipedia.py` through the public Wikipedia API (cleaned text, one block per passage, each with its source URL) | CC BY-SA 4.0 |
| Databricks Dolly 15k | Short question/answer pairs (and, for the RAG data, some with a context passage), converted by `backend/data_tools/convert_open_datasets.py` | CC BY-SA 3.0 |
| OpenAssistant OASST1 | A few hundred short English conversations | Apache 2.0 |
| HuggingFaceTB everyday-conversations-llama3.1-2k | Short everyday chats and greetings (chat training, `collect_chat_data.py`) | Apache 2.0 |
| google Synthetic-Persona-Chat | Casual multi-turn chat (chat training) | CC BY 4.0 |
| allenai SODA (validation split) | Social dialogues (chat training) | CC BY 4.0 |
| OpenAssistant OASST2 | English assistant conversations (chat training) | Apache 2.0 |
| Wikipedia, English and Tamil (person by person) | The article of each Tamil Nadu politician on the roster (1947 to 2026), plus English articles about cases and scandals that really concern them; originals archived as raw HTML. Collected by `backend/pipeline/person_collector.py` through the official Wikimedia API with the project's User-Agent, one request every half second | CC BY-SA 4.0 |
| Wikidata | The roster itself (chief ministers, ministers, speakers, members of the Madras / Tamil Nadu assemblies) and each person's record (party, positions with dates, relatives, birth place) used for entities and relationships | CC0 1.0 |
| Supreme Court and Madras High Court judgments, person by person | The open datasets above, searched by case title and parties through a local metadata index (`backend/pipeline/official_index.py`); a judgment is kept only if its own text names the person next to a political role and shows corruption wording | CC-BY-4.0 (Dattam Labs) |
| DVAC and CBI press releases, person by person | The same two agency sources as the nightly collectors, matched to a person by name in the title | public agency pages, attributed |
| ELVARN's own chat examples | Greetings, small talk, who ELVARN is, remembering a name; written from fixed phrase lists in `backend/data_tools/chat_synthetic.py` (no real people) | the owner's own material |
| Court and agency documents | Madras High Court and Supreme Court judgments (open archives), DVAC and CBI public pages: stored in the knowledge base and exported to `kb_corpus.txt` | public records; each passage keeps its source link |
| The project's own files | The original domain knowledge and constitution texts the project started with (`backend/data/raw/`) | the owner's own material |

**RAG training examples** (`backend/data_tools/build_rag_dataset.py`) are generated from the passages above: notes, a question and an answer
copied or lightly reworded from the notes. They are therefore derived from CC BY-SA material.

**Model weights:** `deploy/models/17m_rag` was trained on the data above. If you publish the weights or the data, keep the attribution and
the share-alike terms of the CC BY-SA sources.

**Web search results** shown by the chat are fetched live through SearXNG; the app only displays titles, short snippets and links to the original pages.

**Politician-by-politician collection** (`docs/PIPELINE_CONTRACT.md`): the Start button in the app runs one person at a time, archives every original
under `backend/data/kb_raw/{source}/{date}/{doc_id}.{ext}` with its SHA-256 in `kb_raw/checksums.txt`, and announces it on Kafka. Spark turns it into the
Parquet corpus (`backend/data/corpus/`), the Neo4j documents, passages, entities and relationships; the nightly `check_stores` job fills Chroma.
Nothing is taken from news sites, and nothing from pages behind a login. The Wikimedia APIs are used under Wikimedia's API policy (their robots.txt keeps
generic crawlers off `/api/`); all other websites are fetched only where their robots.txt allows it.
