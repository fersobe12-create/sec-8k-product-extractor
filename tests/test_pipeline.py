"""Offline tests: SEC responses and the LLM are replaced with fixtures.

Run:  python -m pytest tests -q
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_8k_products as ex  # noqa: E402

TICKERS = json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})

ATOM = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
 <content type="text/xml">
  <accession-number>0000320193-99-000001</accession-number>
  <filing-date>2025-02-19</filing-date>
  <filing-href>https://www.sec.gov/Archives/edgar/data/320193/000032019399000001/0000320193-99-000001-index.htm</filing-href>
  <filing-type>8-K</filing-type>
  <items-desc>items 7.01 and 9.01</items-desc>
 </content>
 <updated>2025-02-19T16:30:43-05:00</updated>
</entry>
<entry>
 <content type="text/xml">
  <accession-number>0000320193-99-000002</accession-number>
  <filing-date>2025-02-01</filing-date>
  <filing-href>https://www.sec.gov/Archives/edgar/data/320193/000032019399000002/0000320193-99-000002-index.htm</filing-href>
  <filing-type>8-K</filing-type>
  <items-desc>items 5.07</items-desc>
 </content>
 <updated>2025-02-01T09:00:00-05:00</updated>
</entry>
</feed>"""

INDEX = """<html><body><table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/ix?doc=/Archives/edgar/data/320193/x/aapl-8k.htm">aapl-8k.htm</a></td><td>8-K</td><td>30 KB</td></tr>
<tr><td>2</td><td>PRESS RELEASE</td><td><a href="/Archives/edgar/data/320193/x/ex99-1.htm">ex99-1.htm</a></td><td>EX-99.1</td><td>20 KB</td></tr>
<tr><td>3</td><td>GRAPHIC</td><td><a href="/Archives/edgar/data/320193/x/logo.jpg">logo.jpg</a></td><td>GRAPHIC</td><td>5 KB</td></tr>
</table></body></html>"""

MAIN_DOC = """<html><body><p>Item 7.01 Regulation FD Disclosure.</p>
<p>On February 19, 2025, Apple Inc. issued a press release, attached as Exhibit 99.1.</p></body></html>"""

EX99 = """<html><body>
<p>Apple Inc. today announced the launch of iPhone 16e, a new member of the iPhone 16 family
featuring the A18 chip and Apple Intelligence, available starting February 28.</p>
<p>The company also reported strong services revenue during the quarter, reflecting customer demand.</p>
<p>Forward-looking statements: this release contains statements about new products and future plans that involve risks.</p>
</body></html>"""

FIXTURES = {
    ex.TICKERS_URL: TICKERS,
    "browse-edgar": ATOM,
    "-index.htm": INDEX,
    "aapl-8k.htm": MAIN_DOC,
    "ex99-1.htm": EX99,
}


def fake_get(self, url, retries=3):
    for key, body in FIXTURES.items():
        if key in url:
            return body
    raise AssertionError(f"unexpected URL {url}")


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(ex.SecClient, "get", fake_get)
    monkeypatch.setattr(ex, "CACHE_DIR", tmp_path / "cache")

    llm_answer = json.dumps({"products": [
        {"name": "iPhone 16e", "description": "A new iPhone 16 family model with the A18 chip and "
                                              "Apple Intelligence, available starting February 28. " * 3},
        {"name": "Apple Vision Ultra", "description": "hallucinated product not in the filing"},
        {"name": "Apple Inc.", "description": "the company, not a product"},
    ]})
    prompts = []
    monkeypatch.setattr(ex, "query_ollama", lambda m, p, timeout=300: prompts.append(p) or llm_answer)

    records = ex.run(["AAPL"], per_company=10, max_docs=100, model="llama3.1",
                     user_agent="test test@ucf.edu", out_dir=tmp_path / "out")

    assert len(records) == 1
    r = records[0]
    assert (r.company_name, r.stock_name, r.new_product) == ("Apple Inc.", "AAPL", "iPhone 16e")
    assert r.filing_time == "2025-02-19T16:30:43-05:00"
    assert len(r.product_description) < 180
    # Safe-harbor paragraph must not be sent to the LLM
    assert "Forward-looking" not in prompts[0]

    rows = list(csv.DictReader(open(tmp_path / "out" / "new_products.csv")))
    assert rows[0]["new_product"] == "iPhone 16e"
    logrows = list(csv.DictReader(open(tmp_path / "out" / "processing_log.csv")))
    assert {r["status"] for r in logrows} == {"ok", "skipped_items"}
    assert "|" in (tmp_path / "out" / "new_products.txt").read_text()


def test_truncate_limit():
    s = ex.truncate("word " * 100)
    assert len(s) < 180 and s.endswith("...")


def test_parse_llm_json_deepseek():
    raw = '<think>reasoning...</think>```json\n{"products": [{"name": "X1", "description": "d"}]}\n```'
    assert ex.parse_llm_json(raw) == [{"name": "X1", "description": "d"}]


def test_ner_finds_org():
    ents = ex.ner_entities("Apple Inc. has announced the launch of its latest MacBook Pro.")
    assert "Apple Inc." in ents["ORG"]


def test_clean_records():
    R = ex.ProductRecord
    rows = [
        R("Abbott", "ABT", "2026-03-23T08:00", "YouTube", "Social media platform"),
        R("Accenture", "ACN", "2026-01-28T16:00", "2010 Share Incentive Plan", "plan"),
        R("Aflac", "AFL", "2026-02-04T16:00", "Miraito", "cancer insurance"),
        R("Aflac", "AFL", "2025-11-04T16:00", "Miraito", "cancer insurance"),
        R("Abbott", "ABT", "2026-10-21T09:00", "Similac", "formula"),
        R("Abbott", "ABT", "2026-10-21T07:00", "Similac 360 Total Care", "formula"),
    ]
    out = ex.clean_records(rows)
    assert [(r.stock_name, r.new_product, r.filing_time[:10]) for r in out] == [
        ("ABT", "Similac 360 Total Care", "2026-10-21"),
        ("AFL", "Miraito", "2025-11-04"),
    ]
