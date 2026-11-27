#!/usr/bin/env python3
"""Pilot: find measured nulls ("no superconductivity down to X K") in full-text papers (plan: docs/null-mining.md).

    python scripts/null_mining.py search            # OpenAlex full-text phrase search, 2010 onwards
    python scripts/null_mining.py fetch --limit 200 # arXiv full texts for the first works with an arXiv version
    python scripts/null_mining.py extract           # the sentences that state a null and its floor

Stage 1 is a pattern match, so it costs no model reading. It finds the sentences and floors; which compound each
sentence is about, and whether the sample was bulk and at ambient pressure, are left for the readers (stage 2), whose
leads then go through the owner's correction rule. Nothing here changes a label.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/null-mining"           # git-ignored
SUMMARY = ROOT / "docs/null-mining.json"
PHRASES = ["no superconductivity down to", "no superconducting transition down to", "not superconducting down to",
           "no sign of superconductivity down to", "no evidence of superconductivity down to",
           "absence of superconductivity down to", "does not superconduct down to", "no superconductivity above",
           "no superconducting transition was observed down to", "remains normal down to"]
SINCE = "2010-01-01"
FLOOR = re.compile(r"(?:down to|above|to)\s*(?:T\s*=\s*)?~?\s*(\d+(?:\.\d+)?)\s*(m?K)\b", re.I)
NULL = re.compile(r"(no (?:sign|evidence|trace|indication)s? of (?:bulk )?superconductivity|no superconduct\w*|"
                  r"not superconduct\w*|does not superconduct|absence of superconductivity|"
                  r"no (?:\w+ )?superconducting transition|remains? (?:metallic|normal)(?: \w+)? down to)", re.I)


def curl_json(url: str) -> dict:
    for attempt in range(4):
        r = subprocess.run(["curl", "-sf", "--max-time", "90", url], capture_output=True)
        if r.returncode == 0:
            return json.loads(r.stdout)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(url)


def arxiv_id(work: dict) -> str | None:
    for loc in work.get("locations") or []:
        url = (loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or "")
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})", url)
        if m:
            return m.group(1)
    return None


def cmd_search() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seen: dict[str, dict] = {}
    for phrase in PHRASES:
        cursor, n = "*", 0
        while cursor:
            q = urllib.parse.quote(f'"{phrase}"')
            url = (f"https://api.openalex.org/works?filter=fulltext.search:{q},from_publication_date:{SINCE}"
                   f"&per_page=200&cursor={urllib.parse.quote(cursor)}"
                   "&select=id,doi,title,publication_year,locations,primary_location")
            d = curl_json(url)
            for w in d["results"]:
                entry = seen.setdefault(w["id"], {"id": w["id"], "doi": w.get("doi"), "title": w.get("title"),
                                                  "year": w.get("publication_year"), "arxiv": arxiv_id(w),
                                                  "phrases": []})
                entry["phrases"].append(phrase)
                n += 1
            cursor = d["meta"].get("next_cursor") if d["results"] else None
            time.sleep(0.2)
        print(f"{phrase!r}: {n}", flush=True)
    with open(OUT / "candidates.jsonl", "w", encoding="utf-8") as handle:
        for entry in sorted(seen.values(), key=lambda e: e["id"]):
            handle.write(json.dumps(entry) + "\n")
    print(f"{len(seen)} distinct works, {sum(1 for e in seen.values() if e['arxiv'])} with an arXiv version")


def cmd_fetch(limit: int) -> None:
    works = [json.loads(line) for line in open(OUT / "candidates.jsonl", encoding="utf-8")]
    todo = sorted((w for w in works if w["arxiv"]),                 # "down to" phrases state a floor: those first
                  key=lambda w: (not any("down to" in p for p in w["phrases"]), w["id"]))[:limit]
    (OUT / "text").mkdir(exist_ok=True)
    for i, w in enumerate(todo, 1):
        name = w["arxiv"].replace("/", "_")
        txt = OUT / "text" / f"{name}.txt"
        if txt.is_file():
            continue
        pdf = OUT / "text" / f"{name}.pdf"
        r = subprocess.run(["curl", "-sfL", "--max-time", "120", "-o", str(pdf), f"https://arxiv.org/pdf/{w['arxiv']}"])
        if r.returncode == 0:
            subprocess.run(["pdftotext", "-q", str(pdf), str(txt)])
            pdf.unlink(missing_ok=True)
        if i % 25 == 0:
            print(f"{i}/{len(todo)}", flush=True)
        time.sleep(3)                                   # arXiv asks for no more than one request every few seconds


def sentences(text: str) -> list[str]:
    text = re.sub(r"-\n(\w)", r"\1", text).replace("\n", " ")
    return re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)


def cmd_extract() -> dict:
    works = {json.loads(l)["arxiv"]: json.loads(l) for l in open(OUT / "candidates.jsonl", encoding="utf-8")}
    rows = []
    for txt in sorted((OUT / "text").glob("*.txt")):
        key = txt.stem.replace("_", "/", 1) if "/" not in txt.stem and not re.match(r"\d{4}\.", txt.stem) else txt.stem
        w = works.get(key) or works.get(txt.stem) or {}
        for s in sentences(txt.read_text(encoding="utf-8", errors="replace")):
            if len(s) > 600 or not NULL.search(s):
                continue
            m = FLOOR.search(s)
            if not m:
                continue
            floor = float(m.group(1)) / (1000 if m.group(2).lower() == "mk" else 1)
            rows.append({"arxiv": txt.stem, "openalex": w.get("id"), "doi": w.get("doi"), "year": w.get("year"),
                         "title": w.get("title"), "floor_k": floor, "sentence": s.strip()})
    with open(OUT / "snippets.jsonl", "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    summary = {"stage": "1, pattern match only", "phrases": PHRASES, "since": SINCE,
               "candidate_works": sum(1 for _ in open(OUT / "candidates.jsonl", encoding="utf-8")),
               "with_arxiv": sum(1 for w in works if w), "texts_fetched": len(list((OUT / "text").glob("*.txt"))),
               "null_sentences_with_a_floor": len(rows),
               "papers_with_one": len({r["arxiv"] for r in rows}),
               "floor_k_quartiles": sorted(r["floor_k"] for r in rows)[len(rows) // 4::max(1, len(rows) // 4)][:3]
               if rows else []}
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("search")
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--limit", type=int, default=200)
    sub.add_parser("extract")
    args = parser.parse_args(argv)
    {"search": cmd_search, "extract": cmd_extract}.get(args.command, lambda: cmd_fetch(args.limit))()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
