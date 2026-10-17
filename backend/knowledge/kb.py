"""Knowledge graph store on Neo4j; the graph follows ontology.ttl and schema.cypher.

Rules the code enforces (Neo4j Community cannot enforce them as database constraints, so every write goes through here):
* every CaseParty / CaseEvent / Claim is linked to exactly one Passage, and every Passage to a Source (a citation);
* a 'court_finding' may only be backed by a judgment or an order, never by news, a reference work or a press release;
* values for roles, statuses, stages, levels and document types come from fixed lists;
* each write is one transaction, so a fact is stored completely or not at all.
Answers are built from stored facts only, so a fact that is not in the graph cannot appear in an answer.
`audit()` re-checks these rules over the whole graph.

Connection: NEO4J_URI (e.g. bolt://localhost:7687), NEO4J_USER (default neo4j), NEO4J_PASSWORD.
"""
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = Path(__file__).with_name("schema.cypher")

DOC_TYPES = ("fir", "chargesheet", "judgment", "order", "commission_report", "assembly", "news", "reference",
             "press_release", "audit_report", "affidavit", "rti_response", "court_record")
COURT_DOC_TYPES = ("judgment", "order")
ENTITY_TYPES = ("Person", "Organisation", "InvestigatingAgency", "Court", "PoliticalParty", "Case", "Act", "Section",
                "Place", "Event", "Concept", "Scheme", "Date", "Reference", "Position")
ROLES = ("accused", "complainant", "investigating_officer", "judge", "witness", "prosecutor", "counsel")
STATUSES = ("allegation_only", "registered", "named_in_fir", "accused", "arrested", "on_bail", "charged", "convicted",
            "acquitted", "discharged", "quashed", "appeal_pending")
STAGES = ("fir", "investigation", "arrest", "chargesheet", "trial", "high_court", "supreme_court", "final_judgment")
LEVELS = ("alleged", "reported", "court_finding")
CONFIDENCE = ("low", "medium", "high")
CASE_FIELDS = ("case_number", "year_registered", "fir_ecr_number", "department", "location", "amount_inr", "confidence", "summary")

LEVEL_WORDS = {"alleged": "alleged", "reported": "reported", "court_finding": "court finding"}
STAGE_ORDER = {s: i for i, s in enumerate(STAGES)}
NOT_FOUND = "Not found in my data."
UNAVAILABLE = "The knowledge base is unavailable right now (it may be waking up: please try again in a minute), so I can't answer this reliably."
CORRUPTION_WORDS = re.compile(r"corrupt|scam|bribe|\bfir\b|dvac|\bcbi\b|vigilance|disproportionate|charge ?sheet|convict|acquit|"
                              r"money.?laundering|\bed case\b|judgment|embezzle|graft", re.I)


def _one_of(value, allowed, name, optional=False):
    if value is None and optional:
        return value
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


def _uid():
    return uuid.uuid4().hex


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(type_, name):
    return f"{type_}:{' '.join(name.lower().split())}"


def assertion_level(doc_type, method):
    """The strongest level a relation may carry. 'court_finding' only for a judgment or an order; a reference work is 'reported' at
    most; news only 'alleges'; co-occurrence (two names in one passage) is never above 'reported' because it states no link at all."""
    if method == "cooccurrence":
        return "alleged" if doc_type == "news" else "reported"
    if doc_type in COURT_DOC_TYPES:
        return "court_finding"
    return "alleged" if doc_type == "news" else "reported"


RELATION_TYPE = re.compile(r"[A-Z][A-Z_]{2,40}")


def passage_id(digest, position):
    """Deterministic id of a passage, shared by Neo4j (Passage.id) and Chroma (document id): reloading the same text
    gives the same ids in both stores, so loads are idempotent and a Chroma hit can be followed into the graph."""
    return hashlib.sha1(f"{digest}|{position}".encode("utf-8")).hexdigest()[:24] if digest else _uid()


class KnowledgeBase:
    def __init__(self, uri=None, user=None, password=None, database=None):
        uri = uri or os.environ.get("NEO4J_URI")
        if not uri:
            raise ValueError("NEO4J_URI is not set")
        from neo4j import GraphDatabase      # imported here, not at module level: a site without a graph (e.g. the Render deploy) never needs the package
        self.driver = GraphDatabase.driver(uri, auth=(user or os.environ.get("NEO4J_USER", "neo4j"),
                                                      password or os.environ.get("NEO4J_PASSWORD", "")))
        self.database = database
        self.driver.verify_connectivity()
        for line in SCHEMA.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("//"):
                self._run(line)

    @classmethod
    def from_env(cls):
        """The configured knowledge base, or None when NEO4J_URI is not set."""
        return cls() if os.environ.get("NEO4J_URI") else None

    def close(self):
        self.driver.close()

    # ---- plumbing --------------------------------------------------
    def _session(self):
        return self.driver.session(database=self.database) if self.database else self.driver.session()

    def _run(self, query, **params):
        with self._session() as s:
            return s.execute_write(lambda tx: [r.data() for r in tx.run(query, **params)])

    def _read(self, query, **params):
        with self._session() as s:
            return s.execute_read(lambda tx: [r.data() for r in tx.run(query, **params)])

    def _tx(self, fn):
        """Run fn(tx) as one transaction: all of its writes happen, or none do."""
        with self._session() as s:
            return s.execute_write(fn)

    # ---- ingestion -------------------------------------------------
    @staticmethod
    def _digest(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None

    def source_exists(self, text):
        h = self._digest(text)
        return bool(h and self._read("MATCH (s:Source {sha256:$h}) RETURN s.id AS id", h=h))

    def add_source(self, title, doc_type, text=None, url=None, publisher=None, published_date=None, licence=None):
        """Insert a source and return its id. If `text` was already ingested, returns the existing source's id."""
        _one_of(doc_type, DOC_TYPES, "doc_type")
        h = self._digest(text)
        if h:
            found = self._read("MATCH (s:Source {sha256:$h}) RETURN s.id AS id", h=h)
            if found:
                return found[0]["id"]
        sid = _uid()
        self._run("CREATE (s:Source:Document {id:$id, title:$title, url:$url, publisher:$publisher, doc_type:$doc_type, "
                  "published_date:$pd, created_at:$now, retrieved_at:$now, licence:$licence, sha256:$h})",
                  id=sid, title=title, url=url, publisher=publisher, doc_type=doc_type, pd=published_date,
                  now=_now(), licence=licence, h=h)
        return sid

    def add_document(self, title, doc_type, text, passages, url=None, publisher=None, published_date=None, licence=None):
        """Load one document (source + passages + proposed facts) in ONE transaction: all of it is stored, or none of it.
        `passages` is a list of {"locator", "text", "candidates": [{stage, status, event_date, sentence, level, sections, fir_no}]}.
        Returns (source_id, created); created is False if the same text was already loaded."""
        _one_of(doc_type, DOC_TYPES, "doc_type")
        h = self._digest(text)
        found = self._read("MATCH (s:Source {sha256:$h}) RETURN s.id AS id", h=h) if h else []
        if found:
            return found[0]["id"], False
        sid = _uid()
        rows, cands = [], {}                                   # cands keyed by uid: the same sentence repeated in a passage is ONE proposal
        for i, p in enumerate(passages):                      # validate everything BEFORE the transaction starts
            pid = passage_id(h, i)
            rows.append({"id": pid, "pos": i, "loc": p.get("locator"), "text": p["text"]})
            for c in p.get("candidates", ()):
                _one_of(c["level"], LEVELS, "level")
                uid = hashlib.sha1(f"{pid}|{c['stage']}|{c['status']}|{c['sentence']}".encode("utf-8")).hexdigest()[:12]
                cands.setdefault(uid, {"pid": pid, "uid": uid, "stage": c["stage"], "status": c["status"], "d": c.get("event_date"),
                                       "sentence": c["sentence"], "level": c["level"], "sections": c.get("sections"), "fir": c.get("fir_no")})

        def work(tx):
            tx.run("CREATE (s:Source:Document {id:$id, title:$title, url:$url, publisher:$publisher, doc_type:$doc_type, "
                   "published_date:$pd, created_at:$now, retrieved_at:$now, licence:$licence, sha256:$h})",
                   id=sid, title=title, url=url, publisher=publisher, doc_type=doc_type, pd=published_date,
                   now=_now(), licence=licence, h=h)
            if rows:
                tx.run("MATCH (s:Source {id:$sid}) UNWIND $rows AS r "
                       "CREATE (p:Passage {id:r.id, source_id:$sid, position:r.pos, locator:r.loc, text:r.text})-[:FROM]->(s)", sid=sid, rows=rows)
            if cands:
                tx.run("UNWIND $cands AS c MATCH (p:Passage {id:c.pid}) "
                       "CREATE (n:Candidate {uid:c.uid, stage:c.stage, status:c.status, event_date:c.d, sentence:c.sentence, "
                       "level:c.level, sections:c.sections, fir_no:c.fir, state:'pending'})-[:FROM_PASSAGE]->(p)", cands=list(cands.values()))
        self._tx(work)
        return sid, True

    def add_passage(self, source_id, position, text, locator=None):
        pid = _uid()
        rows = self._run("MATCH (s:Source {id:$sid}) CREATE (p:Passage {id:$pid, source_id:$sid, position:$pos, locator:$loc, text:$text})"
                         "-[:FROM]->(s) RETURN p.id AS id", sid=source_id, pid=pid, pos=position, loc=locator, text=text)
        if not rows:
            raise ValueError("unknown source")
        return pid

    @staticmethod
    def _entity_tx(tx, type_, name, aliases=()):
        _one_of(type_, ENTITY_TYPES, "entity type")
        key = _key(type_, name)
        rec = tx.run(f"MERGE (e:Entity:{type_} {{key:$key}}) ON CREATE SET e.type=$type, e.name=$name, e.aliases=[] "
                     "RETURN e.aliases AS aliases", key=key, type=type_, name=name).single()
        merged = sorted(set(rec["aliases"]) | {a for a in (name, *aliases) if a})
        tx.run("MATCH (e:Entity {key:$key}) SET e.aliases=$aliases", key=key, aliases=merged)
        return key

    def entity(self, type_, name, aliases=()):
        """Get or create an entity (unique by type + name), register aliases, return its key."""
        return self._tx(lambda tx: self._entity_tx(tx, type_, name, aliases))

    def add_case(self, name, **fields):
        """Create or update a case. Only the given fields change (case_number, year_registered, fir_ecr_number,
        department, location, amount_inr, confidence, summary)."""
        bad = set(fields) - set(CASE_FIELDS)
        if bad:
            raise ValueError(f"unknown case fields {sorted(bad)}")
        _one_of(fields.get("confidence"), CONFIDENCE, "confidence", optional=True)
        key = self.entity("Case", name)
        props = {k: v for k, v in fields.items() if v is not None}
        if props:
            self._run("MATCH (c:Entity {key:$key}) SET c += $props", key=key, props=props)
        return key

    def get_or_create_case(self, name):
        return self.entity("Case", name)

    def _check_level(self, tx, level, passage_id):
        """A 'court_finding' may only be backed by a judgment or an order."""
        _one_of(level, LEVELS, "level")
        rec = tx.run("MATCH (p:Passage {id:$pid})-[:FROM]->(s:Source) RETURN s.doc_type AS t", pid=passage_id).single()
        if rec is None:
            raise ValueError("a fact must cite an existing passage")
        if level == "court_finding" and rec["t"] not in COURT_DOC_TYPES:
            raise ValueError("a court_finding must cite a judgment or order")

    def _party_tx(self, tx, case_id, person_id, role, level, passage_id, status=None, status_date=None):
        _one_of(role, ROLES, "role")
        _one_of(status, STATUSES, "status", optional=True)
        self._check_level(tx, level, passage_id)
        pid = _uid()
        rec = tx.run("MATCH (c:Entity:Case {key:$c}), (e:Entity:Person {key:$p}), (src:Passage {id:$src}) "
                     "CREATE (cp:CaseParty {id:$id, role:$role, status:$status, status_date:$sd, level:$level}) "
                     "CREATE (cp)-[:IN_CASE]->(c), (cp)-[:PERSON]->(e), (cp)-[:SUPPORTED_BY]->(src) RETURN cp.id AS id",
                     c=case_id, p=person_id, src=passage_id, id=pid, role=role, status=status, sd=status_date, level=level).single()
        if rec is None:
            raise ValueError("unknown case or person")
        return pid

    def _event_tx(self, tx, case_id, stage, description, level, passage_id, event_date=None):
        _one_of(stage, STAGES, "stage")
        self._check_level(tx, level, passage_id)
        eid = _uid()
        rec = tx.run("MATCH (c:Entity:Case {key:$c}), (src:Passage {id:$src}) "
                     "CREATE (ev:CaseEvent {id:$id, stage:$stage, event_date:$d, description:$desc, level:$level}) "
                     "CREATE (ev)-[:OF_CASE]->(c), (ev)-[:SUPPORTED_BY]->(src) RETURN ev.id AS id",
                     c=case_id, src=passage_id, id=eid, stage=stage, d=event_date, desc=description, level=level).single()
        if rec is None:
            raise ValueError("unknown case")
        return eid

    def add_party(self, case_id, person_id, role, level, passage_id, status=None, status_date=None):
        return self._tx(lambda tx: self._party_tx(tx, case_id, person_id, role, level, passage_id, status, status_date))

    def add_event(self, case_id, stage, description, level, passage_id, event_date=None):
        return self._tx(lambda tx: self._event_tx(tx, case_id, stage, description, level, passage_id, event_date))

    def add_claim(self, subject_id, predicate, level, passage_id, object_id=None, object_value=None, valid_from=None, valid_to=None):
        if object_id is None and object_value is None:
            raise ValueError("a claim needs an object entity or a value")

        def work(tx):
            self._check_level(tx, level, passage_id)
            cid = _uid()
            rec = tx.run("MATCH (s:Entity {key:$s}), (src:Passage {id:$src}) "
                         "CREATE (cl:Claim {id:$id, predicate:$pred, object_value:$val, valid_from:$vf, valid_to:$vt, level:$level}) "
                         "CREATE (cl)-[:SUBJECT]->(s), (cl)-[:SUPPORTED_BY]->(src) RETURN cl.id AS id",
                         s=subject_id, src=passage_id, id=cid, pred=predicate, val=object_value, vf=valid_from, vt=valid_to, level=level).single()
            if rec is None:
                raise ValueError("unknown subject")
            if object_id is not None:
                if tx.run("MATCH (cl:Claim {id:$id}), (o:Entity {key:$o}) CREATE (cl)-[:OBJECT]->(o) RETURN 1 AS ok",
                          id=cid, o=object_id).single() is None:
                    raise ValueError("unknown object entity")
            return cid
        return self._tx(work)

    # ---- processing pipeline: Document / Passage / Entity / RELATION (every write is a MERGE, so a reload never duplicates) ----
    @staticmethod
    def _entity_rows(entities, doc_id, source, source_url, created_at, first_pid=None):
        """Validated entity rows grouped by type: {type: {key: row}}; an entity seen twice keeps the union of its aliases."""
        by_type = {}
        for e in entities:
            _one_of(e["type"], ENTITY_TYPES, "entity type")
            if not e.get("name") or not e["name"].strip():
                continue
            key = _key(e["type"], e["name"])
            props = {k: e[k] for k in ("person_key", "wikidata_id", "birth_date", "death_date") if e.get(k)}
            row = by_type.setdefault(e["type"], {}).setdefault(key, {
                "key": key, "name": e["name"].strip(), "aliases": set(), "props": {}, "pid": e.get("passage_id") or first_pid})
            row["aliases"] |= {a for a in (e["name"].strip(), *e.get("aliases", ())) if a}
            row["props"].update(props)
        return {t: [{**r, "aliases": sorted(r["aliases"])} for r in rows.values()] for t, rows in by_type.items()}

    def _entities_tx(self, tx, entities, doc_id=None, source=None, source_url=None, created_at=None, first_pid=None):
        """MERGE entities by their key; first_doc_id / first_passage_id / source / created_at are set only when the node is created."""
        for type_, rows in self._entity_rows(entities, doc_id, source, source_url, created_at, first_pid).items():
            tx.run(f"UNWIND $rows AS r MERGE (e:Entity:{type_} {{key:r.key}}) "
                   "ON CREATE SET e.type=$type, e.name=r.name, e.aliases=[], e.first_doc_id=$doc, e.first_passage_id=r.pid, "
                   "e.source=$source, e.source_url=$url, e.created_at=$created "
                   "SET e.aliases = coalesce(e.aliases, []) + [a IN r.aliases WHERE NOT a IN coalesce(e.aliases, [])], e += r.props",
                   rows=rows, type=type_, doc=doc_id, source=source, url=source_url, created=created_at or _now())

    def add_entities(self, entities, doc_id=None, source=None, source_url=None, created_at=None):
        """Get or create entities ({type, name, aliases, person_key, wikidata_id, ...}); returns their keys."""
        self._tx(lambda tx: self._entities_tx(tx, entities, doc_id, source, source_url, created_at))
        return [_key(e["type"], e["name"]) for e in entities if e.get("name")]

    @staticmethod
    def _mentions_tx(tx, mentions):
        """mentions = [{passage_id, type, name}]: (Passage)-[:MENTIONS]->(Entity)."""
        rows = [{"pid": m["passage_id"], "key": _key(m["type"], m["name"])} for m in mentions]
        if rows:
            tx.run("UNWIND $rows AS r MATCH (p:Passage {id:r.pid}), (e:Entity {key:r.key}) MERGE (p)-[:MENTIONS]->(e)", rows=rows)

    def add_mentions(self, mentions):
        self._tx(lambda tx: self._mentions_tx(tx, mentions))

    def _relations_tx(self, tx, relations, doc):
        """relations = [{type, from:{type,name}, to:{type,name}, method, confidence, start, end, kind, passage_id}]. The endpoints are created
        if missing. Each relation carries its doc_id / passage_id / source and an assertion_level that the document type allows."""
        rows, ends = [], []
        for r in relations:
            if not RELATION_TYPE.fullmatch(r["type"]):
                raise ValueError(f"bad relation type {r['type']!r}")
            _one_of(r["method"], ("wikidata", "cooccurrence", "rule"), "method")
            _one_of(r.get("confidence", "low"), CONFIDENCE, "confidence")
            level = r.get("assertion_level") or assertion_level(doc["doc_type"], r["method"])
            _one_of(level, LEVELS, "level")
            if level == "court_finding" and (doc["doc_type"] not in COURT_DOC_TYPES or r["method"] == "cooccurrence"):
                raise ValueError("a court_finding must come from a judgment or order, never from co-occurrence")
            ends += [r["from"], r["to"]]
            a, b = _key(r["from"]["type"], r["from"]["name"]), _key(r["to"]["type"], r["to"]["name"])
            pid = r.get("passage_id") or ""
            rk = hashlib.sha1("|".join(str(x or "") for x in (r["type"], r["method"], a, b, doc["doc_id"], pid, r.get("start"), r.get("end"), r.get("kind"))).encode("utf-8")).hexdigest()
            rows.append({"a": a, "b": b, "rk": rk, "type": r["type"], "method": r["method"], "conf": r.get("confidence", "low"), "pid": pid,
                         "level": level, "start": r.get("start"), "end": r.get("end"), "kind": r.get("kind")})
        if not rows:
            return
        self._entities_tx(tx, ends, doc["doc_id"], doc.get("source"), doc.get("source_url"), doc.get("created_at"))
        tx.run("UNWIND $rows AS r MATCH (a:Entity {key:r.a}), (b:Entity {key:r.b}) MERGE (a)-[x:RELATION {rel_key:r.rk}]->(b) "
               "SET x.type=r.type, x.method=r.method, x.confidence=r.conf, x.doc_id=$doc, x.passage_id=r.pid, x.source=$source, "
               "x.source_url=$url, x.created_at=$created, x.assertion_level=r.level, x.`start`=r.start, x.`end`=r.end, x.kind=r.kind",
               rows=rows, doc=doc["doc_id"], source=doc.get("source"), url=doc.get("source_url"), created=doc.get("created_at") or _now())

    def add_relations(self, relations, doc):
        """`doc` = {doc_id, doc_type, source, source_url, created_at}: the document the relations were read from."""
        _one_of(doc["doc_type"], DOC_TYPES, "doc_type")
        self._tx(lambda tx: self._relations_tx(tx, relations, doc))

    def add_document_graph(self, doc, records, entities=(), relations=()):
        """Load one processed document in ONE transaction, idempotently (MERGE on stable keys: re-loading it changes nothing).
        doc = {doc_id, title, doc_type, source, source_url, language, person_key, created_at, published_date, publisher, licence,
        sha256 (of the clean text), raw_sha256}. records = passage dicts as in the Parquet corpus plus optional "entities" (mentions) and
        "relations" (found in that passage). `entities` / `relations` are document-level extras (e.g. from a Wikidata record).
        Returns (document_id, created). If another document with the same clean text exists, nothing is written and its id is returned."""
        _one_of(doc["doc_type"], DOC_TYPES, "doc_type")
        did, created_at = doc["doc_id"], doc.get("created_at") or _now()
        prows, mentions, rels, ents = [], [], list(relations), list(entities)
        for rec in records:
            prows.append({"id": rec["passage_id"], "props": {
                "passage_id": rec["passage_id"], "source_id": did, "doc_id": did, "position": rec["position"], "locator": f"passage {rec['position'] + 1}",
                "text": rec["text"], "source": doc.get("source"), "source_url": doc.get("source_url"), "created_at": created_at,
                "language": rec.get("language"), "token_estimate": rec.get("token_estimate"), "sha256": rec.get("sha256"),
                "quality_score": rec.get("quality_score"), "corruption_related": bool(rec.get("corruption_related"))}})
            for e in rec.get("entities", ()):
                ents.append({**e, "passage_id": rec["passage_id"]})
                mentions.append({"passage_id": rec["passage_id"], "type": e["type"], "name": e["name"]})
            rels += [{**r, "passage_id": rec["passage_id"]} for r in rec.get("relations", ())]
        props = {"doc_id": did, "title": doc.get("title"), "url": doc.get("source_url"), "publisher": doc.get("publisher"),
                 "doc_type": doc["doc_type"], "published_date": doc.get("published_date"), "licence": doc.get("licence"),
                 "sha256": doc.get("sha256"), "raw_sha256": doc.get("raw_sha256"), "source": doc.get("source"),
                 "source_url": doc.get("source_url"), "created_at": created_at, "language": doc.get("language"),
                 "person_key": doc.get("person_key")}
        for k in ("case_name", "case_type", "case_number", "court", "bench", "year", "judge", "outcome", "case_ref", "subjects", "statutes", "case_stage", "petitions"):      # court judgments and case records
            if doc.get(k) not in (None, "", []):
                props[k] = doc[k]
        props = {k: v for k, v in props.items() if v is not None}
        full_doc = {**doc, "created_at": created_at}

        def work(tx):
            if props.get("sha256"):
                other = tx.run("MATCH (s:Source {sha256:$h}) WHERE s.id <> $id RETURN s.id AS id", h=props["sha256"], id=did).single()
                if other is not None:
                    return other["id"], False
            if props.get("url"):                                  # the same page loaded before in another form (another loader, another cleaning): not stored twice
                same_page = tx.run("MATCH (s:Source {url:$u}) WHERE s.id <> $id RETURN s.id AS id", u=props["url"], id=did).single()
                if same_page is not None:
                    return same_page["id"], False
            existed = tx.run("MATCH (s:Source {id:$id}) RETURN 1 AS x", id=did).single() is not None
            tx.run("MERGE (s:Source {id:$id}) ON CREATE SET s.retrieved_at=$now SET s:Document, s += $props", id=did, now=_now(), props=props)
            if prows:
                tx.run("MATCH (s:Source {id:$id}) UNWIND $rows AS r MERGE (p:Passage {id:r.id}) SET p += r.props "
                       "MERGE (p)-[:FROM]->(s) MERGE (s)-[:HAS_PASSAGE]->(p)", id=did, rows=prows)
            self._entities_tx(tx, ents, did, doc.get("source"), doc.get("source_url"), created_at)
            self._mentions_tx(tx, mentions)
            self._relations_tx(tx, rels, full_doc)
            return did, not existed
        return self._tx(work)

    def link_person_documents(self, pairs):
        """Write (:Entity {type:'Person'})-[:HAS_DOCUMENT]->(:Source) for (person_key, doc_id, url) triples whose person entity and document both exist.
        The document is found by doc_id, or by url (a page the knowledge base already held under another id). Idempotent. Returns the pairs linked."""
        done = []
        for person_key, doc_id, url in pairs:
            rows = self._run("MATCH (e:Entity {type:'Person', person_key:$pk}) MATCH (s:Source) WHERE s.doc_id=$d OR ($u IS NOT NULL AND s.url=$u) "
                             "MERGE (e)-[:HAS_DOCUMENT]->(s) RETURN count(s) AS n", pk=person_key, d=doc_id, u=url)
            if rows and rows[0]["n"]:
                done.append((person_key, doc_id))
        return done

    def label_documents(self):
        """One-off: give every existing Source the label Document (new loads already do). Returns how many were changed."""
        return self._run("MATCH (s:Source) WHERE NOT s:Document SET s:Document RETURN count(s) AS n")[0]["n"]

    # ---- review queue ----------------------------------------------
    def iter_passages(self, exclude_doc_types=()):
        """Every passage with its source type (used by the fact extractor)."""
        return self._read("MATCH (p:Passage)-[:FROM]->(s:Source) WHERE NOT s.doc_type IN $ex "
                          "RETURN p.id AS id, p.text AS text, s.doc_type AS doc_type", ex=list(exclude_doc_types))

    def add_candidate(self, passage_id, stage, status, event_date, sentence, level, sections=None, fir_no=None):
        """Store a proposed fact (pending). Returns True if new, False if the same proposal already exists."""
        uid = hashlib.sha1(f"{passage_id}|{stage}|{status}|{sentence}".encode("utf-8")).hexdigest()[:12]
        _one_of(level, LEVELS, "level")
        if self._read("MATCH (c:Candidate {uid:$uid}) RETURN 1 AS x", uid=uid):
            return False
        rows = self._run("MATCH (p:Passage {id:$pid}) "
                         "CREATE (c:Candidate {uid:$uid, stage:$stage, status:$status, event_date:$d, sentence:$sentence, "
                         "level:$level, sections:$sections, fir_no:$fir, state:'pending'})-[:FROM_PASSAGE]->(p) RETURN c.uid AS uid",
                         pid=passage_id, uid=uid, stage=stage, status=status, d=event_date, sentence=sentence, level=level,
                         sections=sections, fir=fir_no)
        return bool(rows)

    def pending_candidates(self, limit=20):
        return self._read("MATCH (c:Candidate {state:'pending'})-[:FROM_PASSAGE]->(p:Passage)-[:FROM]->(s:Source) "
                          "RETURN c{.*} AS c, s.title AS title, s.url AS url, s.doc_type AS doc_type, p.locator AS locator, p.text AS passage "
                          "ORDER BY c.uid LIMIT $n", n=limit)

    def get_candidate(self, uid):
        rows = self._read("MATCH (c:Candidate {uid:$uid})-[:FROM_PASSAGE]->(p:Passage)-[:FROM]->(s:Source) "
                          "RETURN c{.*} AS c, p.id AS passage_id, s.title AS title, s.url AS url, s.doc_type AS doc_type, "
                          "p.locator AS locator, p.text AS passage", uid=uid)
        return rows[0] if rows else None

    def reject_candidate(self, uid):
        self._run("MATCH (c:Candidate {uid:$uid, state:'pending'}) SET c.state='rejected'", uid=uid)

    def approve_candidate(self, uid, case, person=None, role="accused", stage=None, level=None):
        """Turn a pending candidate into a stored fact (event, and a party row if `person` is given) in ONE transaction."""
        def work(tx):
            row = tx.run("MATCH (c:Candidate {uid:$uid, state:'pending'})-[:FROM_PASSAGE]->(p:Passage) "
                         "RETURN c{.*} AS c, p.id AS pid", uid=uid).single()
            if row is None:
                raise ValueError(f"candidate {uid} is not pending")
            c, pid = row["c"], row["pid"]
            use_level = level or c["level"]
            case_id = self._entity_tx(tx, "Case", case)
            self._event_tx(tx, case_id, stage or c["stage"], c["sentence"], use_level, pid, c["event_date"])
            if person:
                person_id = self._entity_tx(tx, "Person", person)
                status = c["status"] if c["status"] in STATUSES else None
                self._party_tx(tx, case_id, person_id, role, use_level, pid, status, c["event_date"])
            tx.run("MATCH (c:Candidate {uid:$uid}) SET c.state='approved'", uid=uid)
        self._tx(work)

    # ---- lookup ----------------------------------------------------
    def find_entities(self, text, type_=None):
        """Entities whose name or alias appears in `text` as whole words (case-insensitive), as dicts with id/type/name."""
        rows = self._read("MATCH (e:Entity) WHERE $t IS NULL OR e.type=$t RETURN e.key AS id, e.type AS type, e.name AS name, e.aliases AS aliases",
                          t=type_)
        low, hits = text.lower(), {}
        for r in rows:
            for alias in r["aliases"]:
                if len(alias) >= 3 and re.search(r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)", low):
                    hits[r["id"]] = {"id": r["id"], "type": r["type"], "name": r["name"]}
                    break
        return list(hits.values())

    def cases_of_person(self, person_id):
        return self._read(
            "MATCH (cp:CaseParty)-[:PERSON]->(:Entity {key:$k}), (cp)-[:IN_CASE]->(c:Entity), "
            "(cp)-[:SUPPORTED_BY]->(p:Passage)-[:FROM]->(s:Source) "
            "RETURN cp.role AS role, cp.status AS status, cp.status_date AS status_date, cp.level AS level, c.name AS case_name, "
            "c.case_number AS case_number, c.year_registered AS year_registered, p.locator AS locator, s.id AS source_id, "
            "s.title AS source_title, s.url AS url, s.doc_type AS doc_type, s.published_date AS published_date "
            "ORDER BY coalesce(c.year_registered, 9999), c.name", k=person_id)

    def case_timeline(self, case_id):
        rows = self._read(
            "MATCH (ev:CaseEvent)-[:OF_CASE]->(:Entity {key:$k}), (ev)-[:SUPPORTED_BY]->(p:Passage)-[:FROM]->(s:Source) "
            "RETURN ev.stage AS stage, ev.event_date AS event_date, ev.description AS description, ev.level AS level, "
            "p.locator AS locator, s.id AS source_id, s.title AS source_title, s.url AS url, s.doc_type AS doc_type, "
            "s.published_date AS published_date", k=case_id)
        return sorted(rows, key=lambda r: (r["event_date"] or "9999", STAGE_ORDER[r["stage"]]))

    # ---- graph traversal (entities + relationships), used by graphrag ----
    _SRC = ("s.id AS source_id, s.title AS source_title, s.url AS url, s.doc_type AS doc_type, "
            "s.published_date AS published_date, s.licence AS licence")

    def case_parties(self, case_id):
        """Everyone linked to a case with role and status (co-accused, complainants, judges ...), each with its citation."""
        return self._read("MATCH (cp:CaseParty)-[:IN_CASE]->(:Entity {key:$k}), (cp)-[:PERSON]->(e:Entity), "
                          "(cp)-[:SUPPORTED_BY]->(p:Passage)-[:FROM]->(s:Source) "
                          "RETURN e.name AS person, e.key AS person_id, cp.role AS role, cp.status AS status, cp.status_date AS status_date, "
                          f"cp.level AS level, {self._SRC}", k=case_id)

    def case_info(self, case_id):
        rows = self._read("MATCH (c:Entity {key:$k}) RETURN c{.name, .case_number, .year_registered, .department, .location, .amount_inr, .fir_ecr_number} AS c", k=case_id)
        return rows[0]["c"] if rows else {}

    def entity_claims(self, entity_id):
        """Other cited facts about an entity: relationships to other entities (office held, agency, court ...) and plain values."""
        return self._read("MATCH (cl:Claim)-[:SUBJECT]->(:Entity {key:$k}), (cl)-[:SUPPORTED_BY]->(p:Passage)-[:FROM]->(s:Source) "
                          "OPTIONAL MATCH (cl)-[:OBJECT]->(o:Entity) "
                          "RETURN cl.predicate AS predicate, o.name AS object, cl.object_value AS value, cl.valid_from AS valid_from, "
                          f"cl.valid_to AS valid_to, cl.level AS level, {self._SRC}", k=entity_id)

    def passage_ids_present(self, ids):
        """Which of these passage ids exist in the graph (used to keep Chroma and Neo4j in step)."""
        return {r["id"] for r in self._read("MATCH (p:Passage) WHERE p.id IN $ids RETURN p.id AS id", ids=list(ids))}

    # ---- answers ---------------------------------------------------
    def answer(self, question):
        """Build a cited answer from stored facts only. Returns {'text', 'sources', 'found'}."""
        people = self.find_entities(question, "Person")
        cases = self.find_entities(question, "Case")
        refs, lines = {}, []

        def cite(row):
            if row["source_id"] not in refs:
                refs[row["source_id"]] = {"n": len(refs) + 1, "title": row["source_title"], "url": row["url"],
                                          "type": row["doc_type"], "date": row["published_date"]}
            return f"[{refs[row['source_id']]['n']}]"

        for person in people:
            rows = self.cases_of_person(person["id"])
            if not rows:
                continue
            lines.append(f"Cases linked to {person['name']} in my data:")
            for r in rows:
                status = (r["status"] or r["role"]).replace("_", " ")
                when = f" ({r['status_date']})" if r["status_date"] else ""
                num = f", {r['case_number']}" if r["case_number"] else ""
                lines.append(f"- {r['case_name']}{num}: {r['role'].replace('_', ' ')}, {status}{when} "
                             f"- {LEVEL_WORDS[r['level']]} {cite(r)}")
        for case in cases:
            events = self.case_timeline(case["id"])
            if not events:
                continue
            lines.append(f"Timeline of {case['name']}:")
            for r in events:
                lines.append(f"- {r['event_date'] or 'date unknown'}: {r['description']} "
                             f"({r['stage'].replace('_', ' ')}; {LEVEL_WORDS[r['level']]}) {cite(r)}")
        if not lines:
            return {"text": NOT_FOUND, "sources": [], "found": False}
        sources = sorted(refs.values(), key=lambda s: s["n"])
        lines.append("Allegations are not findings of guilt; only a 'court finding' is established by a court.")
        lines.append("Sources: " + "; ".join(f"[{s['n']}] {s['title']} ({s['type']}, {s['date'] or 'undated'})" for s in sources))
        return {"text": "\n".join(lines), "sources": sources, "found": True}

    def search_passages(self, query, k=5):
        """Full-text (BM25) search over passage text. `query` is used as given after removing Lucene operators
        (so several words mean "any of them"). Keyword fallback for tools; the chat's retrieval is graphrag.py (Chroma + graph)."""
        safe = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', " ", query).strip()
        if not safe:
            return []
        return self._read("CALL db.index.fulltext.queryNodes('passage_text', $q) YIELD node, score "
                          "MATCH (node)-[:FROM]->(s:Source) RETURN node.text AS text, node.locator AS locator, s.title AS title, "
                          "s.url AS url, s.doc_type AS doc_type, s.publisher AS publisher, s.published_date AS published_date, "
                          "s.licence AS licence, score ORDER BY score DESC LIMIT $k", q=safe, k=k)

    # ---- chat hook -------------------------------------------------
    def populated(self):
        """True once any document text is stored (so a corruption question nothing can answer is refused, not guessed)."""
        return bool(self._read("MATCH (p:Passage) RETURN count(p) > 0 AS any")[0]["any"])

    # ---- integrity and backup --------------------------------------
    def audit(self):
        """Re-check the rules over the whole graph; returns a list of problems (empty = healthy)."""
        problems = []
        for label in ("CaseParty", "CaseEvent", "Claim"):
            n = self._read(f"MATCH (f:{label}) WHERE NOT (f)-[:SUPPORTED_BY]->(:Passage)-[:FROM]->(:Source) RETURN count(f) AS n")[0]["n"]
            if n:
                problems.append(f"{n} {label} node(s) without a cited passage and source")
            n = self._read(f"MATCH (f:{label} {{level:'court_finding'}})-[:SUPPORTED_BY]->(:Passage)-[:FROM]->(s:Source) "
                           "WHERE NOT s.doc_type IN $ok RETURN count(f) AS n", ok=list(COURT_DOC_TYPES))[0]["n"]
            if n:
                problems.append(f"{n} {label} node(s) marked court_finding but not backed by a judgment/order")
        n = self._read("MATCH (p:Passage) WHERE NOT (p)-[:FROM]->(:Source) RETURN count(p) AS n")[0]["n"]
        if n:
            problems.append(f"{n} passage(s) without a source")
        return problems

    # Nodes that are NOT exported when reference documents are left out (see export_jsonl): a reference source nothing cites,
    # its passages, and any proposed facts attached to those passages.
    _KEEP = ("NOT ({x}:Source AND {x}.id IN $drop) AND NOT ({x}:Passage AND {x}.source_id IN $drop) "
             "AND NOT ({x}:Candidate AND EXISTS {{ ({x})-[:FROM_PASSAGE]->(q:Passage) WHERE q.source_id IN $drop }})")

    def export_jsonl(self, path, skip_reference=False):
        """Portable backup: every node and relationship as JSON lines (restore with import_jsonl). Returns counts.
        skip_reference=True leaves out reference documents (Wikipedia, project notes), news documents (copyrighted articles) and court judgments
        (hundreds of thousands of passages, no facts cite them) and their passages unless a fact cites them: the hosted graph then stays small,
        and those passages are served by the vector store alone."""
        drop = []
        if skip_reference:
            drop = [r["id"] for r in self._read(
                "MATCH (s:Source) WHERE s.doc_type IN ['reference', 'news', 'judgment'] AND NOT EXISTS { MATCH (s)<-[:FROM]-(:Passage)<-[:SUPPORTED_BY]-() } RETURN s.id AS id")]
        with self._session() as s:
            node_rows = [{"node": r["id"], "labels": r["labels"], "props": r["props"]} for r in s.run(
                f"MATCH (n) WHERE {self._KEEP.format(x='n')} RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props", drop=drop)]
            rel_rows = [{"rel": r["type"], "from": r["a"], "to": r["b"], "props": r["props"]} for r in s.run(
                f"MATCH (a)-[r]->(b) WHERE {self._KEEP.format(x='a')} AND {self._KEEP.format(x='b')} "
                "RETURN elementId(a) AS a, type(r) AS type, elementId(b) AS b, properties(r) AS props", drop=drop)]
        if skip_reference:                                   # judges, parties and dates read out of judgments: with their documents gone nothing links to them
            linked = {r["from"] for r in rel_rows} | {r["to"] for r in rel_rows}
            lone = [n for n in node_rows if n["node"] not in linked and "Entity" in n["labels"] and n["props"].get("first_passage_id")]
            from_judgments = set()
            for i in range(0, len(lone), 5000):
                from_judgments |= {r["id"] for r in self._read(
                    "MATCH (p:Passage)-[:FROM]->(:Source {doc_type: 'judgment'}) WHERE p.id IN $ids RETURN p.id AS id",
                    ids=[n["props"]["first_passage_id"] for n in lone[i:i + 5000]])}
            skip = {n["node"] for n in lone if n["props"]["first_passage_id"] in from_judgments}
            node_rows = [n for n in node_rows if n["node"] not in skip]
        with open(path, "w", encoding="utf-8") as f:
            for row in node_rows + rel_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"nodes": len(node_rows), "relationships": len(rel_rows)}

    def import_jsonl(self, path, batch=1000):
        """Restore an export_jsonl backup into THIS graph (used to move the data to a hosted Neo4j). The target must be EMPTY.
        Labels and relationship types in the file are checked against the schema before they are put into any query."""
        if self._read("MATCH (n) RETURN count(n) AS n")[0]["n"]:
            raise ValueError("the target graph is not empty; refusing to import into it")
        primary = ("Source", "Passage", "Entity", "CaseParty", "CaseEvent", "Claim", "Candidate")
        allowed = set(primary) | set(ENTITY_TYPES) | {"Document"}
        nodes, rels, main = {}, {}, {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "node" in r:
                    labels = tuple(sorted(r["labels"]))
                    if not labels or not set(labels) <= allowed:
                        raise ValueError(f"unexpected node labels in the backup: {labels}")
                    nodes.setdefault(labels, []).append({"id": r["node"], "props": r["props"]})
                    main[r["node"]] = next(l for l in primary if l in labels)
                else:
                    if not re.fullmatch(r"[A-Z_]+", r["rel"]):
                        raise ValueError(f"unexpected relationship type in the backup: {r['rel']!r}")
                    rels.setdefault(r["rel"], []).append(r)
        for labels, rows in nodes.items():                                       # 1) nodes, carrying their old id as _xid
            for i in range(0, len(rows), batch):
                self._run(f"UNWIND $rows AS r CREATE (n:{':'.join(labels)}) SET n = r.props, n._xid = r.id", rows=rows[i:i + batch])
        for label in primary:                                                    # 2) temporary indexes so relationship matching is fast
            self._run(f"CREATE INDEX xid_{label} IF NOT EXISTS FOR (n:{label}) ON (n._xid)")
        self._run("CALL db.awaitIndexes(600)")
        for rel, rows in rels.items():                                           # 3) relationships, grouped by type and end labels
            groups = {}
            for r in rows:
                groups.setdefault((main[r["from"]], main[r["to"]]), []).append(r)
            for (a, b), part in groups.items():
                for i in range(0, len(part), batch):
                    self._run(f"UNWIND $rows AS r MATCH (a:{a} {{_xid: r.from}}), (b:{b} {{_xid: r.to}}) CREATE (a)-[x:{rel}]->(b) SET x = r.props",
                              rows=[{"from": r["from"], "to": r["to"], "props": r["props"]} for r in part[i:i + batch]])
        self._run("MATCH (n) WHERE n._xid IS NOT NULL REMOVE n._xid")           # 4) clean up
        for label in primary:
            self._run(f"DROP INDEX xid_{label} IF EXISTS")
        return {"nodes": sum(len(v) for v in nodes.values()), "relationships": sum(len(v) for v in rels.values())}
