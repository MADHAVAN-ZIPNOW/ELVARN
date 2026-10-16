"""One folder per politician, so a person's material is in one place and not scattered over source and date folders.

The archive keeps the layout the pipeline contract asks for, kb_raw/{source}/{date}/{doc_id}.{ext}: every file stored once, named by its bytes.
On top of it each person gets a view that holds the SAME files (NTFS hard links: no extra disk space; a copy where links are not possible):

    kb_raw/people/<name>__<wikidata id>/
        manifest.json                 who the person is, and every document with its topic, url, size and sha256
        person/        <doc_id>.html  the person's own articles (English and Tamil Wikipedia)
        structured/    <doc_id>.json  the Wikidata record
        corruption/    <doc_id>.html  articles about cases and scandals that concern the person
        court_record/  <doc_id>.pdf   Supreme Court and Madras High Court judgments
        agency_record/ <doc_id>.pdf   DVAC, CBI and ED releases

An article about a case that names several politicians appears in each of their folders (the same file, linked). Nothing here is read by Spark:
messages point at the files under {source}/{date}/. The monitor and the raw-file counters skip `people/`, so nothing is counted twice.
"""
import json
import os
import re
import shutil
import unicodedata
from pathlib import Path

from backend.pipeline.common import RAW_DIR

PEOPLE_DIR = "people"


def folder_name(person):
    """'Jayaram-Jayalalithaa__Q465018': readable, safe on Windows, unique through the Wikidata id (Tamil-only names fall back to the id)."""
    ascii_name = unicodedata.normalize("NFKD", person.get("name") or "").encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_name).strip("-")[:60]
    qid = (person.get("wikidata_id") or person["key"].split(":")[-1]).replace("/", "_")
    return f"{slug}__{qid}" if slug else qid


class PersonFolders:
    def __init__(self, registry, root=RAW_DIR):
        self.registry, self.root = registry, Path(root)

    def folder(self, person):
        return self.root / PEOPLE_DIR / folder_name(person)

    def add(self, person, doc, topic):
        """Make the document appear in the person's folder (link, or copy). `doc` needs doc_id and raw_path. Returns True if it is there afterwards."""
        if not doc.get("raw_path"):
            return False                                             # a page the knowledge base already held: there is no file to link
        src = self.root / doc["raw_path"].removeprefix("kb_raw/")
        if not src.is_file():
            return False
        dst = self.folder(person) / (topic or "other") / src.name
        if dst.exists():
            return True
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:                                              # another drive, or a file system without hard links
            shutil.copy2(src, dst)
        return True

    def write_manifest(self, person):
        """The person's summary and document list, written atomically (safe to read while the collector runs)."""
        row = self.registry.person(person["key"]) or person
        docs = self.registry.documents_of(person["key"], limit=100000)
        by_topic = {}
        for d in docs:
            by_topic[d["topic"]] = by_topic.get(d["topic"], 0) + 1
        manifest = {"person_key": row["key"], "name": row["name"], "wikidata_id": row.get("wikidata_id"), "party": row.get("party"),
                    "first_year": row.get("first_year"), "last_year": row.get("last_year"), "positions": row.get("positions", []),
                    "status": row.get("status"), "documents": len(docs), "documents_by_topic": by_topic, "files": [
                        {"doc_id": d["doc_id"], "topic": d["topic"], "title": d["title"], "url": d["url"], "bytes": d["bytes"], "collected_at": d["collected_at"],
                         "path": f"{d['topic']}/{Path(d['raw_path']).name}" if d.get("raw_path") else None,
                         "in_knowledge_base_before": d.get("status") == "in_kb"} for d in docs]}
        folder = self.folder(person)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, folder / "manifest.json")
        return manifest

    def sync_person(self, person):
        """Put every registered document of the person into the folder and refresh the manifest (also repairs a folder that was deleted)."""
        for d in self.registry.documents_of(person["key"], limit=100000):
            self.add(person, d, d["topic"])
        return self.write_manifest(person)
