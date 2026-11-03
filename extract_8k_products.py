#!/usr/bin/env python3
"""
SEC 8-K New Product Extractor
=============================

Pipeline
--------
1. Map ticker -> CIK + company name      (https://www.sec.gov/files/company_tickers.json)
2. List recent 8-K filings per CIK       (EDGAR browse-edgar Atom feed)
3. Open each filing's index page and download the main 8-K document and
   any EX-99.x exhibits (press releases live in EX-99.1 most of the time)
4. Clean the HTML with BeautifulSoup and keep only paragraphs that look like
   product announcements (keyword filter = cheap pre-screen before the LLM)
5. Run spaCy Named Entity Recognition (ORG / PRODUCT) on those paragraphs
6. Ask a local LLM (Ollama: llama3.1 / deepseek-r1 / ...) to return JSON with
   the new products and a short description, using the NER entities as hints
7. Validate the answer (product name must appear in the filing, description
   < 180 characters, no generic names) and write CSV + pipe-delimited table

Usage
-----
    python extract_8k_products.py --tickers AAPL NVDA MSFT --per-company 10
    python extract_8k_products.py --sp500 --max-docs 100 --model llama3.1
    python extract_8k_products.py --sp500 --max-docs 100 --no-llm   # NER-only baseline

SEC fair-access rules: send a descriptive User-Agent with your e-mail and stay
under 10 requests/second. Set SEC_USER_AGENT or pass --user-agent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from io import StringIO
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEARCH_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
    "&CIK={cik}&type=8-K&dateb=&owner=include&count={count}&output=atom"
)
SEC_BASE = "https://www.sec.gov"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

DEFAULT_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "UCF FinTech student research your.email@example.com"
)
REQUEST_PAUSE = 0.15          # seconds between SEC requests (< 10 req/s)
MAX_DESC_CHARS = 180          # assignment: description must be < 180 chars
MAX_LLM_CONTEXT_CHARS = 6000  # keep prompts small so local models stay fast

CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("output")

# 8-K items where product launches are normally reported.
#   7.01 = Regulation FD disclosure, 8.01 = Other events, 9.01 = exhibits
# Filings that only report e.g. 5.02 (officer changes) or 5.07 (votes) are skipped.
PRODUCT_ITEMS = {"7.01", "8.01", "9.01"}

PRODUCT_KEYWORDS = re.compile(
    r"\b(launch(?:es|ed|ing)?|introduc(?:e|es|ed|ing)|unveil(?:s|ed|ing)?|"
    r"debut(?:s|ed)?|new (?:product|platform|service|device|drug|therapy|model|"
    r"vehicle|chip|processor|feature|solution|offering|line)|"
    r"(?:now|generally|commercially) available|FDA (?:approv|clear)\w*|"
    r"approval of|roll(?:s|ed)? out|next[- ]generation|announc(?:e|es|ed|ing) "
    r"(?:the )?(?:launch|release|availability|introduction))\b",
    re.IGNORECASE,
)

# Names the LLM sometimes returns that are not products.
GENERIC_NAMES = {
    "", "n/a", "na", "none", "null", "unknown", "product", "products", "the product",
    "new product", "service", "services", "platform", "company", "the company",
}

FIELDNAMES = ["company_name", "stock_name", "filing_time", "new_product", "product_description"]

# Fallback list (large S&P 500 constituents that frequently announce products)
# used when the live S&P 500 list cannot be downloaded.
SP500_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM",
    "ADBE", "AMD", "INTC", "QCOM", "CSCO", "IBM", "TXN", "MU", "AMAT", "NOW",
    "INTU", "PANW", "CRWD", "ANET", "SNPS", "CDNS", "FTNT", "ADSK", "HPQ", "DELL",
    "LLY", "JNJ", "MRK", "ABBV", "PFE", "AMGN", "GILD", "BMY", "REGN", "VRTX",
    "MDT", "ABT", "TMO", "DHR", "ISRG", "SYK", "BSX", "EW", "DXCM", "ZBH",
    "PG", "KO", "PEP", "MDLZ", "KHC", "GIS", "CL", "KMB", "HSY", "MKC",
    "NKE", "SBUX", "MCD", "CMG", "YUM", "DIS", "NFLX", "CMCSA", "T", "VZ",
    "TMUS", "F", "GM", "BA", "LMT", "RTX", "GE", "HON", "CAT", "DE",
    "V", "MA", "PYPL", "AXP", "JPM", "GS", "MS", "BLK", "SCHW", "FI",
    "WMT", "COST", "TGT", "HD", "LOW", "BBY", "UBER", "ABNB", "BKNG", "EXPE",
]

log = logging.getLogger("8k")


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------
@dataclass
class Company:
    ticker: str
    cik: str          # zero-padded to 10 digits
    name: str


@dataclass
class Filing:
    company: Company
    accession: str
    filing_date: str
    filing_time: str  # ISO timestamp from the feed's <updated> tag
    index_url: str
    items: list[str] = field(default_factory=list)


@dataclass
class ProductRecord:
    company_name: str
    stock_name: str
    filing_time: str
    new_product: str
    product_description: str


# ----------------------------------------------------------------------------
# HTTP helpers (polite + cached)
# ----------------------------------------------------------------------------
class SecClient:
    def __init__(self, user_agent: str, use_cache: bool = True):
        if "example.com" in user_agent:
            log.warning("Set a real e-mail in SEC_USER_AGENT / --user-agent; "
                        "the SEC may block generic agents.")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self.use_cache = use_cache
        self._last = 0.0
        CACHE_DIR.mkdir(exist_ok=True)

    def get(self, url: str, retries: int = 3) -> str:
        key = hashlib.sha1(url.encode()).hexdigest()
        cache_file = CACHE_DIR / f"{key}.txt"
        if self.use_cache and cache_file.exists():
            return cache_file.read_text(encoding="utf-8", errors="ignore")

        for attempt in range(1, retries + 1):
            wait = REQUEST_PAUSE - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.session.get(url, timeout=30)
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                text = r.text
                if self.use_cache:
                    cache_file.write_text(text, encoding="utf-8")
                return text
            except requests.RequestException as exc:
                log.warning("GET %s failed (%s), attempt %d/%d", url, exc, attempt, retries)
                time.sleep(2 * attempt)
        raise RuntimeError(f"Could not download {url}")


# ----------------------------------------------------------------------------
# Step 1 - ticker -> CIK
# ----------------------------------------------------------------------------
def load_company_map(client: SecClient) -> dict[str, Company]:
    data = json.loads(client.get(TICKERS_URL))
    out = {}
    for row in data.values():
        t = row["ticker"].upper()
        out[t] = Company(ticker=t, cik=str(row["cik_str"]).zfill(10), name=row["title"])
    log.info("Loaded %d tickers from SEC", len(out))
    return out


def sp500_tickers(user_agent: str) -> list[str]:
    """Current S&P 500 list from Wikipedia; fall back to a built-in list."""
    try:
        import pandas as pd
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": user_agent}, timeout=30,
        ).text
        table = pd.read_html(StringIO(html))[0]
        tickers = [s.replace(".", "-").upper() for s in table["Symbol"].astype(str)]
        log.info("Loaded %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as exc:  # noqa: BLE001
        log.warning("S&P 500 download failed (%s); using built-in list", exc)
        return SP500_FALLBACK


# ----------------------------------------------------------------------------
# Step 2 - list 8-K filings from the Atom feed
# ----------------------------------------------------------------------------
def parse_items(items_desc: str) -> list[str]:
    return re.findall(r"\d+\.\d+", items_desc or "")


def list_8k_filings(client: SecClient, company: Company, count: int) -> list[Filing]:
    xml = client.get(SEARCH_URL.format(cik=company.cik, count=count))
    soup = BeautifulSoup(xml, "xml")
    filings = []
    for entry in soup.find_all("entry"):
        ftype = entry.find("filing-type")
        if ftype and ftype.get_text(strip=True) not in {"8-K", "8-K/A"}:
            continue
        href = entry.find("filing-href")
        if not href:
            continue
        updated = entry.find("updated")
        fdate = entry.find("filing-date")
        acc = entry.find("accession-number")
        items = entry.find("items-desc")
        filings.append(Filing(
            company=company,
            accession=acc.get_text(strip=True) if acc else "",
            filing_date=fdate.get_text(strip=True) if fdate else "",
            filing_time=updated.get_text(strip=True) if updated else "",
            index_url=href.get_text(strip=True),
            items=parse_items(items.get_text() if items else ""),
        ))
    return filings


def is_candidate(filing: Filing) -> bool:
    """Skip filings whose items cannot contain a product announcement."""
    if not filing.items:          # unknown -> keep, let the text filter decide
        return True
    return bool(PRODUCT_ITEMS.intersection(filing.items))


# ----------------------------------------------------------------------------
# Step 3 - documents inside a filing
# ----------------------------------------------------------------------------
def filing_document_urls(client: SecClient, filing: Filing) -> list[str]:
    """Return URLs of the main 8-K document and EX-99.x exhibits."""
    soup = BeautifulSoup(client.get(filing.index_url), "lxml")
    table = soup.find("table", class_="tableFile")
    urls = []
    if not table:
        return urls
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        doc_type = tds[3].get_text(strip=True).upper()
        a = tds[2].find("a")
        if not a or not a.get("href"):
            continue
        href = a["href"]
        if href.startswith("/ix?doc="):          # inline-XBRL viewer link
            href = href[len("/ix?doc="):]
        if not re.search(r"\.(htm|html|txt)$", href, re.I):
            continue
        if doc_type.startswith("8-K") or doc_type.startswith("EX-99"):
            urls.append(SEC_BASE + href if href.startswith("/") else href)
    # Exhibits first: press releases carry the product language.
    urls.sort(key=lambda u: 0 if re.search(r"ex-?99|ex99|dex99", u, re.I) else 1)
    return urls


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "title", "meta"]):
        tag.decompose()
    for hidden in soup.select('[style*="display:none"], ix\\:header'):
        hidden.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# ----------------------------------------------------------------------------
# Step 4 - keep only product-related paragraphs
# ----------------------------------------------------------------------------
def split_paragraphs(text: str) -> list[str]:
    paras = []
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(block.split())
        if len(block) >= 40:
            paras.append(block)
    return paras


def relevant_sections(text: str, max_chars: int = MAX_LLM_CONTEXT_CHARS) -> str:
    """Paragraphs that mention a launch, plus the following paragraph for context."""
    paras = split_paragraphs(text)
    keep: list[int] = []
    for i, p in enumerate(paras):
        # Skip safe-harbor boilerplate: it often contains "new products" generically.
        if re.search(r"forward-looking statements|safe harbor", p, re.I):
            continue
        if PRODUCT_KEYWORDS.search(p):
            for j in (i, i + 1):
                if j < len(paras) and j not in keep:
                    keep.append(j)
    out, total = [], 0
    for i in keep:
        if total + len(paras[i]) > max_chars:
            break
        out.append(paras[i])
        total += len(paras[i])
    return "\n\n".join(out)


# ----------------------------------------------------------------------------
# Step 5 - Named Entity Recognition (spaCy)
# ----------------------------------------------------------------------------
_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            sys.exit("spaCy model missing: run  python -m spacy download en_core_web_sm")
    return _NLP


def ner_entities(text: str) -> dict[str, list[str]]:
    doc = get_nlp()(text[:100_000])
    ents: dict[str, list[str]] = {"ORG": [], "PRODUCT": []}
    for e in doc.ents:
        if e.label_ in ents:
            val = " ".join(e.text.split())
            if val not in ents[e.label_] and len(val) > 1:
                ents[e.label_].append(val)
    return ents


# ----------------------------------------------------------------------------
# Step 6 - LLM extraction (Ollama)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You extract NEW PRODUCT announcements from SEC Form 8-K filings.
Return ONLY valid JSON with this exact schema:
{"products": [{"name": "<product name>", "description": "<what it is, < 180 characters>"}]}

Rules:
- A product is a specific, named good, service, drug, device, software, chip, vehicle or platform
  that the filing says is being launched, introduced, released, approved or made available.
- Use the product name exactly as written in the text.
- Do NOT include: the company itself, earnings results, acquisitions, executives,
  financing deals, existing products only mentioned in passing, or generic phrases.
- Description: one factual sentence, under 180 characters, based only on the text.
- If there is no new product, return {"products": []}.
"""

FEW_SHOT_USER = """Company: Apple Inc.
NER hints - ORG: ['Apple Inc.'], PRODUCT: ['MacBook Pro']
Text:
Apple Inc. has announced the launch of its latest MacBook Pro, featuring a new M3 processor.
The company also reported quarterly revenue of $90 billion."""

FEW_SHOT_ASSISTANT = json.dumps({"products": [{
    "name": "MacBook Pro",
    "description": "Apple's latest MacBook Pro laptop, featuring the new M3 processor.",
}]})


def build_user_prompt(company: Company, text: str, ents: dict[str, list[str]]) -> str:
    return (
        f"Company: {company.name}\n"
        f"NER hints - ORG: {ents['ORG'][:10]}, PRODUCT: {ents['PRODUCT'][:15]}\n"
        f"Text:\n{text}"
    )


def parse_llm_json(raw: str) -> list[dict]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)  # deepseek-r1 reasoning
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    products = data.get("products", []) if isinstance(data, dict) else []
    return [p for p in products if isinstance(p, dict)]


def query_ollama(model: str, user_prompt: str, timeout: int = 300) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEW_SHOT_USER},
            {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def ner_only_products(text: str, ents: dict[str, list[str]]) -> list[dict]:
    """Baseline without an LLM: PRODUCT entities + the sentence they appear in."""
    out = []
    for name in ents["PRODUCT"]:
        sent = next((s for s in re.split(r"(?<=[.!?])\s+", text) if name in s), "")
        if sent and PRODUCT_KEYWORDS.search(sent):
            out.append({"name": name, "description": sent})
    return out


# ----------------------------------------------------------------------------
# Step 7 - validation / cleaning
# ----------------------------------------------------------------------------
def truncate(text: str, limit: int = MAX_DESC_CHARS) -> str:
    text = " ".join(str(text).split())
    if len(text) < limit:
        return text
    cut = text[: limit - 4].rsplit(" ", 1)[0].rstrip(",;:-")
    return cut + "..."


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def validate(products: list[dict], source_text: str, company: Company) -> list[dict]:
    src = _norm(source_text)
    company_norm = _norm(company.name)
    seen, clean = set(), []
    for p in products:
        name = " ".join(str(p.get("name", "")).split()).strip(" \"'")
        desc = truncate(p.get("description", ""))
        n = _norm(name)
        if n in GENERIC_NAMES or len(n) < 2 or n == company_norm:
            continue
        if n not in src:                 # hallucination guard
            log.debug("Dropped %r: not found in filing text", name)
            continue
        if n in seen:
            continue
        seen.add(n)
        clean.append({"name": name, "description": desc})
    return clean


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def process_filing(client: SecClient, filing: Filing, model: str | None) -> tuple[list[ProductRecord], dict]:
    status = {"ticker": filing.company.ticker, "accession": filing.accession,
              "filing_time": filing.filing_time, "items": " ".join(filing.items),
              "status": "", "n_products": 0}
    try:
        urls = filing_document_urls(client, filing)
    except Exception as exc:  # noqa: BLE001
        status["status"] = f"index_error: {exc}"
        return [], status

    text = "\n\n".join(html_to_text(client.get(u)) for u in urls[:4])
    section = relevant_sections(text)
    if not section:
        status["status"] = "no_product_language"
        return [], status

    ents = ner_entities(section)
    if model:
        try:
            raw = query_ollama(model, build_user_prompt(filing.company, section, ents))
            products = parse_llm_json(raw)
        except requests.RequestException as exc:
            status["status"] = f"llm_error: {exc}"
            return [], status
    else:
        products = ner_only_products(section, ents)

    products = validate(products, text, filing.company)
    status["status"] = "ok" if products else "no_product_found"
    status["n_products"] = len(products)
    records = [ProductRecord(
        company_name=filing.company.name,
        stock_name=filing.company.ticker,
        filing_time=filing.filing_time or filing.filing_date,
        new_product=p["name"],
        product_description=p["description"],
    ) for p in products]
    return records, status


def write_outputs(records: list[ProductRecord], log_rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "new_products.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))

    # Pipe-delimited table in the format shown in the assignment.
    rows = [FIELDNAMES] + [[getattr(r, k) for k in FIELDNAMES] for r in records]
    headers = ["Company Name", "Stock Name", "Filing Time", "New Product", "Product Description"]
    rows[0] = headers
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    widths[-1] = min(widths[-1], 180)
    with open(out_dir / "new_products.txt", "w", encoding="utf-8") as f:
        for k, row in enumerate(rows):
            f.write(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)).rstrip() + "\n")
            if k == 0:
                f.write("-" * (sum(widths) + 3 * (len(widths) - 1)) + "\n")

    with open(out_dir / "processing_log.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["ticker", "accession", "filing_time", "items", "status", "n_products"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(log_rows)


def run(tickers: Iterable[str], per_company: int, max_docs: int, model: str | None,
        user_agent: str, out_dir: Path, use_cache: bool = True) -> list[ProductRecord]:
    client = SecClient(user_agent, use_cache=use_cache)
    companies = load_company_map(client)
    records: list[ProductRecord] = []
    log_rows: list[dict] = []
    processed = 0

    for t in tickers:
        if processed >= max_docs:
            break
        company = companies.get(t.upper())
        if not company:
            log.warning("Ticker %s not in SEC list, skipped", t)
            continue
        try:
            filings = list_8k_filings(client, company, per_company)
        except Exception as exc:  # noqa: BLE001
            log.warning("Feed failed for %s: %s", t, exc)
            continue
        for filing in filings:
            if processed >= max_docs:
                break
            if not is_candidate(filing):
                log_rows.append({"ticker": t, "accession": filing.accession,
                                 "filing_time": filing.filing_time, "items": " ".join(filing.items),
                                 "status": "skipped_items", "n_products": 0})
                continue
            processed += 1
            recs, status = process_filing(client, filing, model)
            records.extend(recs)
            log_rows.append(status)
            log.info("[%d/%d] %s %s -> %s (%d)", processed, max_docs, t,
                     filing.filing_date, status["status"], status["n_products"])
        # checkpoint after every company so a crash never loses work
        write_outputs(records, log_rows, out_dir)

    write_outputs(records, log_rows, out_dir)
    log.info("Done: %d filings analysed, %d product rows -> %s",
             processed, len(records), out_dir / "new_products.csv")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract new product launches from SEC 8-K filings")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--tickers", nargs="+", help="tickers to analyse, e.g. AAPL NVDA")
    src.add_argument("--sp500", action="store_true", help="use the S&P 500 constituents")
    ap.add_argument("--per-company", type=int, default=10, help="8-Ks to list per company (max 100)")
    ap.add_argument("--max-docs", type=int, default=100, help="stop after N filings analysed")
    ap.add_argument("--model", default="llama3.1", help="Ollama model (llama3.1, deepseek-r1:8b, ...)")
    ap.add_argument("--no-llm", action="store_true", help="NER-only baseline, no LLM calls")
    ap.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    tickers = args.tickers or (sp500_tickers(args.user_agent) if args.sp500 else SP500_FALLBACK)
    run(tickers, args.per_company, args.max_docs, None if args.no_llm else args.model,
        args.user_agent, Path(args.out), use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
