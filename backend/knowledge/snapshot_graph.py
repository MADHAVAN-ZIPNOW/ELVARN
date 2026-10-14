"""A read-only copy of the knowledge graph, served from memory (for the free hosted service).

Neo4j needs a Java process that does not fit in a free 512 MB container, so the hosted service loads the lean graph export
(export_jsonl(skip_reference=True), about 7,000 nodes) and answers the same questions from dictionaries. It offers exactly the
methods the retrieval pipeline (graphrag.py) calls on a KnowledgeBase, with the same row shapes and ordering, and the tests run
both implementations on the same data and compare. Neo4j on your own computer stays the source of truth: the snapshot is rebuilt from it.
"""
import gzip
import json
import re

STAGE_ORDER = {s: i for i, s in enumerate(("fir", "investigation", "arrest", "chargesheet", "trial", "high_court", "supreme_court", "final_judgment"))}
CASE_INFO = ("name", "case_number", "year_registered", "department", "location", "amount_inr", "fir_ecr_number")


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


class SnapshotGraph:
    def __init__(self, path):
        nodes, rels = {}, []
        with open_text(path) as f:
            for line in f:
                r = json.loads(line)
                if "node" in r:
                    nodes[r["node"]] = {"labels": set(r["labels"]), "props": r["props"]}
                else:
                    rels.append(r)
        self.nodes = nodes
        self.entities = {n["props"]["key"]: n["props"] for n in nodes.values() if "Entity" in n["labels"]}
        self.sources = {n["props"]["id"]: n["props"] for n in nodes.values() if "Source" in n["labels"]}
        self.passages = {n["props"]["id"]: n["props"] for n in nodes.values() if "Passage" in n["labels"]}
        out = {}                                           # (node id, relationship type) -> [target node id]
        for r in rels:
            out.setdefault((r["from"], r["rel"]), []).append(r["to"])
        self._out = out
        self._by_label = {}
        for xid, n in nodes.items():
            for label in n["labels"]:
                self._by_label.setdefault(label, []).append(xid)

    # ---- helpers --------------------------------------------------------------------------------
    def _first(self, xid, rel):
        targets = self._out.get((xid, rel), [])
        return self.nodes[targets[0]]["props"] if targets else None

    def _source_of(self, passage):
        return self.sources.get(passage["source_id"], {}) if passage else {}

    def _src_fields(self, passage, with_licence=True):
        s = self._source_of(passage)
        row = {"source_id": s.get("id"), "source_title": s.get("title"), "url": s.get("url"), "doc_type": s.get("doc_type"),
               "published_date": s.get("published_date")}
        if with_licence:
            row["licence"] = s.get("licence")
        return row

    # ---- the methods graphrag.py uses ------------------------------------------------------------
    def populated(self):
        return bool(self.passages)

    def find_entities(self, text, type_=None):
        """Entities whose name or alias appears in `text` as whole words (case-insensitive), as dicts with id/type/name."""
        low, hits = text.lower(), {}
        for e in self.entities.values():
            if type_ and e["type"] != type_:
                continue
            for alias in e.get("aliases", []):
                if len(alias) >= 3 and re.search(r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)", low):
                    hits[e["key"]] = {"id": e["key"], "type": e["type"], "name": e["name"]}
                    break
        return list(hits.values())

    def case_info(self, case_id):
        e = self.entities.get(case_id)
        return {k: e.get(k) for k in CASE_INFO} if e else {}

    def cases_of_person(self, person_id):
        rows = []
        for xid in self._by_label.get("CaseParty", []):
            cp = self.nodes[xid]["props"]
            person, case = self._first(xid, "PERSON"), self._first(xid, "IN_CASE")
            passage = self._first(xid, "SUPPORTED_BY")
            if not (person and case and passage and person["key"] == person_id):
                continue
            rows.append({"role": cp["role"], "status": cp.get("status"), "status_date": cp.get("status_date"), "level": cp["level"],
                         "case_name": case["name"], "case_number": case.get("case_number"), "year_registered": case.get("year_registered"),
                         "locator": passage.get("locator"), **self._src_fields(passage, with_licence=False)})
        rows.sort(key=lambda r: (r["year_registered"] if r["year_registered"] is not None else 9999, r["case_name"]))
        return rows

    def case_timeline(self, case_id):
        rows = []
        for xid in self._by_label.get("CaseEvent", []):
            ev = self.nodes[xid]["props"]
            case, passage = self._first(xid, "OF_CASE"), self._first(xid, "SUPPORTED_BY")
            if not (case and passage and case["key"] == case_id):
                continue
            rows.append({"stage": ev["stage"], "event_date": ev.get("event_date"), "description": ev["description"], "level": ev["level"],
                         "locator": passage.get("locator"), **self._src_fields(passage, with_licence=False)})
        return sorted(rows, key=lambda r: (r["event_date"] or "9999", STAGE_ORDER[r["stage"]]))

    def case_parties(self, case_id):
        rows = []
        for xid in self._by_label.get("CaseParty", []):
            cp = self.nodes[xid]["props"]
            case, person, passage = self._first(xid, "IN_CASE"), self._first(xid, "PERSON"), self._first(xid, "SUPPORTED_BY")
            if not (case and person and passage and case["key"] == case_id):
                continue
            rows.append({"person": person["name"], "person_id": person["key"], "role": cp["role"], "status": cp.get("status"),
                         "status_date": cp.get("status_date"), "level": cp["level"], **self._src_fields(passage)})
        return rows

    def entity_claims(self, entity_id):
        rows = []
        for xid in self._by_label.get("Claim", []):
            cl = self.nodes[xid]["props"]
            subject, passage, obj = self._first(xid, "SUBJECT"), self._first(xid, "SUPPORTED_BY"), self._first(xid, "OBJECT")
            if not (subject and passage and subject["key"] == entity_id):
                continue
            rows.append({"predicate": cl["predicate"], "object": obj["name"] if obj else None, "value": cl.get("object_value"),
                         "valid_from": cl.get("valid_from"), "valid_to": cl.get("valid_to"), "level": cl["level"], **self._src_fields(passage)})
        return rows
