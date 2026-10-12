"""Collect real, open-licensed text from Wikipedia for ELVARN's v2 training data.

Wikipedia text is CC BY-SA 4.0: every block keeps a Source URL so the origin and licence stay attached.
The official API is used with a descriptive User-Agent and a pause between requests.

Output (in --out-dir, default backend/data/raw_v2/):
    wikipedia_domain.txt    Tamil Nadu / Indian government, law, courts, audit, finance, welfare (DOMAIN_KNOWLEDGE)
    wikipedia_general.txt   broad general knowledge from the Wikipedia "vital articles" list   (GENERAL_ENGLISH)
Each article becomes (a) one Question/Answer block made from its opening paragraph and (b) passage blocks
of about 900-1400 characters, all in the block format the training pipeline already reads.
Pages are cached in backend/data/cache_wikipedia/ so an interrupted run resumes where it stopped.

Usage (project root):
    python backend/data_tools/collect_wikipedia.py                 # everything
    python backend/data_tools/collect_wikipedia.py --limit 20      # quick test
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
API = "https://en.wikipedia.org/w/api.php"
UA = "ELVARN-learning-project/1.0 (personal research project; madhavan.ai.research@gmail.com)"
PAUSE = 0.35

DOMAIN_CATEGORIES = [
    # Tamil Nadu: government, politics, history, administration
    "Government of Tamil Nadu", "Politics of Tamil Nadu", "History of Tamil Nadu", "Chief ministers of Tamil Nadu",
    "Governors of Tamil Nadu", "Tamil Nadu Legislative Assembly", "Political parties in Tamil Nadu", "Dravidian movement",
    "Districts of Tamil Nadu", "Economy of Tamil Nadu", "Agriculture in Tamil Nadu", "Education in Tamil Nadu",
    "Health in Tamil Nadu", "Transport in Tamil Nadu", "Law enforcement in Tamil Nadu", "Local government in Tamil Nadu",
    "Government schemes in Tamil Nadu", "Madras High Court", "Water in Tamil Nadu", "Tamil Nadu",
    # India: constitution, law, courts, audit, finance, welfare
    "Constitution of India", "Fundamental rights in India", "Judiciary of India", "Supreme Court of India",
    "Indian criminal law", "Law of India", "Right to Information in India", "Corruption in India", "Taxation in India",
    "Public finance in India", "Government of India", "Elections in India", "Government schemes in India",
    "Local government in India", "Indian Administrative Service", "Audit in India", "Anti-corruption agencies",
    "Federalism in India", "Indian welfare schemes", "Indian legislation", "Indian courts",
]
GENERAL_CATEGORIES = [
    "Physics", "Chemistry", "Biology", "Mathematics", "Astronomy", "Earth sciences", "Computer science", "Technology", "Energy",
    "Human anatomy", "Health", "Nutrition", "Diseases and disorders", "Economics", "Geography of India", "History of India",
    "Indian independence movement", "Languages of India", "Rivers of India", "States and union territories of India", "Indian culture",
    "English grammar", "English language", "Linguistics", "Weather", "Oceans", "Continents", "Agriculture", "Education", "Transport",
    "Mathematics education", "Geometry", "Algebra", "Statistics", "Inventions", "Environmental issues", "Climate", "Plants", "Animals",
    "Mammals", "Birds", "Philosophy", "Ethics", "Law", "Democracy", "Human rights", "Banking", "Finance", "Insurance",
]

SKIP_TITLE = re.compile(r"^(List of|Lists of|Index of|Outline of|Timeline of|Template:|Category:|Portal:|Draft:)|\(disambiguation\)|\b(film|album|song|TV series)\)", re.I)
CUT_SECTIONS = {"references", "external links", "see also", "notes", "further reading", "bibliography", "sources",
                "citations", "footnotes", "gallery", "awards", "filmography", "discography"}


def api(params, retries=4):
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            time.sleep(PAUSE)
            return data
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            wait = 2 ** attempt * 2
            print(f"  (retry {attempt + 1} after {wait}s: {exc})", flush=True)
            time.sleep(wait)
    return None


def category_titles_cached(cache_dir, category, per_category, depth=1):
    """category_titles() with an on-disk cache, so restarting a run does not repeat the slow category crawl."""
    f = Path(cache_dir) / "categories.json"
    cache = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    key = f"{category}|{per_category}|{depth}"
    if key not in cache:
        cache[key] = category_titles(category, per_category, depth)
        f.write_text(json.dumps(cache), encoding="utf-8")
    return cache[key]


def category_titles(category, per_category, depth=1):
    """Article titles in a category and (depth levels of) its subcategories."""
    found, queue, seen = [], [(f"Category:{category}", 0)], set()
    while queue and len(found) < per_category:
        cat, d = queue.pop(0)
        if cat in seen:
            continue
        seen.add(cat)
        cont = {}
        while True:
            data = api({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": "500", "cmtype": "page|subcat", **cont})
            if not data or "query" not in data:
                break
            for m in data["query"]["categorymembers"]:
                if m["ns"] == 0:
                    found.append(m["title"])
                elif m["ns"] == 14 and d < depth:
                    queue.append((m["title"], d + 1))
            if "continue" not in data or len(found) >= per_category * 2:
                break
            cont = {"cmcontinue": data["continue"]["cmcontinue"]}
    return found[:per_category * 2]


def vital_titles(page):
    titles, cont = [], {}
    while True:
        data = api({"action": "query", "prop": "links", "titles": page, "pllimit": "500", "plnamespace": "0", **cont})
        if not data or "query" not in data:
            break
        for p in data["query"]["pages"]:
            titles += [l["title"] for l in p.get("links", [])]
        if "continue" not in data:
            break
        cont = {"plcontinue": data["continue"]["plcontinue"]}
    return titles


def fetch_text(title, cache_dir):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", title)[:120]
    cache = cache_dir / f"{safe}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    data = api({"action": "query", "prop": "extracts", "explaintext": "1", "exsectionformat": "wiki", "titles": title, "redirects": "1"})
    if not data or "query" not in data or not data["query"].get("pages"):
        return None
    page = data["query"]["pages"][0]
    rec = {"title": page.get("title", title), "text": page.get("extract", "") or ""}
    cache.write_text(json.dumps(rec), encoding="utf-8")
    return rec


# ------------------------------------------------------------------ cleaning
def clean_text(raw, max_chars):
    """Keep readable English prose: drop reference-style sections, headings, non-Latin scripts and leftovers."""
    keep, lead = [], []
    in_lead = True
    for line in raw.splitlines():
        m = re.match(r"^=+\s*(.*?)\s*=+\s*$", line)
        if m:
            in_lead = False
            if m.group(1).strip().lower() in CUT_SECTIONS:
                break
            continue
        keep.append(line)
        if in_lead:
            lead.append(line)
    def tidy(t):
        t = re.sub(r"[^\x00-ɏ‘’“”–—•…₹ ]", "", t)   # drop Tamil/Hindi/other scripts
        t = re.sub(r"\(\s*[,;:/\-–]*\s*\)", "", t)                                            # parentheses emptied by the removal
        t = re.sub(r"\(\s*[;,]\s*", "(", t)
        t = re.sub(r"\((?:[^()]*?\b)?(?:IPA|pronounced|pronunciation|listen)\b[^()]*\)", "", t)    # pronunciation guides are noise
        t = re.sub(r"\s*\[[^\]]{0,40}\]", "", t)                                                   # leftover [..] fragments
        t = re.sub(r"\(\s*[A-Za-z]{2,12}\s*:\s*\)", "", t)                                                                 # "(Tamil: )" left after removing a script
        t = re.sub(r"\(\s*[,;:/\-–]*\s*\)", "", t)
        t = re.sub(r"[<>]", "", t)                                                                 # never create block tags
        t = re.sub(r"[ \t ]+", " ", t)
        return t.strip()
    paras = [tidy(p) for p in "\n".join(keep).split("\n")]
    paras = [p for p in paras if len(p) >= 60 and not re.match(r"^[\W\d_]+$", p)]
    leads = [tidy(p) for p in "\n".join(lead).split("\n")]
    leads = [p for p in leads if len(p) >= 60]
    text, total = [], 0
    for p in paras:
        if total + len(p) > max_chars:
            break
        text.append(p); total += len(p)
    return text, (leads[0] if leads else "")


def first_sentences(paragraph, max_chars=620, max_sentences=3):
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", paragraph)
    out, n = [], 0
    for s in sents[:max_sentences]:
        if n + len(s) > max_chars and out:
            break
        out.append(s); n += len(s)
    return " ".join(out).strip()


def question_for(title, lead, rng):
    person = bool(re.search(r"\((?:born|c\.\s*)?\s*\d{1,2}?\s*\w*\s*\d{4}|\bborn\b|\b\d{4}\s*[–-]\s*\d{4}\)", lead[:220]))
    forms = ([f"Who is {title}?", f"Who was {title}?", f"Tell me about {title}."] if person
             else [f"What is {title}?", f"Explain {title} briefly.", f"Tell me about {title}."])
    if person and not re.search(r"\b(is|are) an?\b", lead[:200]) and "was" in lead[:200]:
        forms = [f"Who was {title}?", f"Tell me about {title}."]
    return rng.choice(forms)


def chunks(paragraphs, lo=900, hi=1400):
    cur, out = "", []
    for p in paragraphs:
        if cur and len(cur) + len(p) > hi:
            out.append(cur); cur = p
        else:
            cur = (cur + "\n" + p).strip()
        if len(cur) >= lo and len(cur) <= hi:
            out.append(cur); cur = ""
    if len(cur) >= 300:
        out.append(cur)
    return out


def blocks_for(tag, title, paragraphs, lead, rng):
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
    status = "Unverified - Wikipedia text (CC BY-SA 4.0), not checked against official sources"
    out = []
    if lead:
        answer = first_sentences(lead)
        out.append(f"<{tag}>\nTopic: {title}\nQuestion: {question_for(title, lead, rng)}\nAnswer: {answer}\nSource: {url}\nVerification status: {status}\n</{tag}>")
    for c in chunks(paragraphs):
        out.append(f"<{tag}>\nTopic: {title}\nText: {c}\nSource: {url}\nVerification status: {status}\n</{tag}>")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=BACKEND_DIR / "data" / "raw_v2")
    ap.add_argument("--cache-dir", type=Path, default=BACKEND_DIR / "data" / "cache_wikipedia")
    ap.add_argument("--per-category", type=int, default=150, help="max articles taken from each category")
    ap.add_argument("--max-chars", type=int, default=14000, help="max characters kept per article")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many articles per group (testing)")
    ap.add_argument("--general-per-category", type=int, default=60, help="max general-knowledge articles per category")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (polite: keep this small)")
    ap.add_argument("--skip-general", action="store_true")
    ap.add_argument("--only-general", action="store_true", help="collect only the general-knowledge part (leaves wikipedia_domain.txt alone)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True); args.cache_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)

    groups = {"domain": ("DOMAIN_KNOWLEDGE", []), "general": ("GENERAL_ENGLISH", [])}
    for cat in ([] if args.only_general else DOMAIN_CATEGORIES):
        titles = category_titles_cached(args.cache_dir, cat, args.per_category)
        print(f"[domain] {cat}: {len(titles)} candidate pages", flush=True)
        groups["domain"][1].extend(titles)
    if not args.skip_general:
        for cat in GENERAL_CATEGORIES:
            t = category_titles_cached(args.cache_dir, cat, args.general_per_category, 0)
            print(f"[general] {cat}: {len(t)} candidate pages", flush=True)
            groups["general"][1].extend(t)

    seen_titles = set()
    for name, (tag, titles) in groups.items():
        if args.only_general and name == "domain":
            continue
        uniq = []
        for t in titles:
            if t not in seen_titles and not SKIP_TITLE.search(t):
                seen_titles.add(t); uniq.append(t)
        if args.limit:
            uniq = uniq[:args.limit]
        out_file = args.out_dir / f"wikipedia_{name}.txt"
        done = [0]
        def prefetch(t):
            fetch_text(t, args.cache_dir)
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  [{name}] downloaded {done[0]}/{len(uniq)}", flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(prefetch, uniq))
        n_art = n_blocks = n_chars = 0
        with open(out_file, "w", encoding="utf-8") as fh:
            for i, title in enumerate(uniq, 1):
                rec = fetch_text(title, args.cache_dir)
                if not rec or len(rec["text"]) < 1000:
                    continue
                paragraphs, lead = clean_text(rec["text"], args.max_chars)
                if sum(len(p) for p in paragraphs) < 700:
                    continue
                for b in blocks_for(tag, rec["title"], paragraphs, lead, rng):
                    fh.write(b + "\n\n"); n_blocks += 1; n_chars += len(b)
                n_art += 1
                if i % 50 == 0:
                    print(f"  [{name}] {i}/{len(uniq)} pages read, {n_art} kept, {n_blocks} blocks, {n_chars / 4.2 / 1e6:.2f}M tokens (est.)", flush=True)
        print(f"DONE {name}: {n_art} articles -> {n_blocks} blocks, about {n_chars / 4.2 / 1e6:.2f}M tokens (estimated) -> {out_file}", flush=True)


if __name__ == "__main__":
    main()
