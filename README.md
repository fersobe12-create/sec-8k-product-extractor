# SEC 8-K New Product Extractor

Extracts **new product announcements** from SEC Form 8-K filings using
BeautifulSoup parsing, spaCy Named Entity Recognition (NER) and a local
large language model served by [Ollama](https://ollama.com) (LLaMA 3.1 or DeepSeek-R1).

Output (`output/new_products.csv`):

```
company_name, stock_name, filing_time, new_product, product_description
"Example Corp", "EXMPL", "2025-01-11", "Product X", "A revolutionary new AI-powered tool..."
```

## How it works

| Step | What happens | Source |
|------|--------------|--------|
| 1 | Ticker → CIK and official company name | `https://www.sec.gov/files/company_tickers.json` |
| 2 | List recent 8-K filings per company | `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&count={count}&output=atom` |
| 3 | Skip filings whose items can't hold a launch (e.g. only 5.02 or 5.07); keep 7.01 / 8.01 / 9.01 | Atom `<items-desc>` |
| 4 | Open the filing index and download the 8-K body plus EX-99.x press releases | `-index.htm` page |
| 5 | Clean the HTML with BeautifulSoup and keep only paragraphs with launch language (safe-harbor text removed) | regex pre-filter |
| 6 | spaCy NER tags `ORG` and `PRODUCT` entities | `en_core_web_sm` |
| 7 | Local LLM returns JSON `{"products":[{"name","description"}]}` with NER entities as hints | Ollama `/api/chat`, `temperature=0`, `format=json` |
| 8 | Validation: the product name must appear in the filing, generic names and the company name are rejected, descriptions are cut to < 180 characters | |
| 9 | Write the CSV, a pipe-delimited table, and a processing log | `output/` |

## Setup

```bash
git clone https://github.com/fersobe12-create/sec-8k-product-extractor.git
cd sec-8k-product-extractor
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Local LLM
ollama pull llama3.1          # or: ollama pull deepseek-r1:8b
ollama serve                  # if it is not already running
```

The SEC requires a User-Agent with your contact e-mail:

```bash
export SEC_USER_AGENT="Your Name your.email@ucf.edu"   # Windows: set SEC_USER_AGENT=...
```

## Run

```bash
# Quick test on a few companies
python extract_8k_products.py --tickers AAPL NVDA AMD LLY --per-company 10 --max-docs 20

# Assignment run: S&P 500 companies, stop after 100 analysed filings
python extract_8k_products.py --sp500 --per-company 10 --max-docs 100 --model llama3.1

# Use DeepSeek instead
python extract_8k_products.py --sp500 --max-docs 100 --model deepseek-r1:8b

# NER-only baseline (no LLM), useful for comparison in the report
python extract_8k_products.py --sp500 --max-docs 100 --no-llm --out output_ner_baseline
```

Files written to `output/`:

- `new_products.csv`: the deliverable (all values quoted)
- `new_products.txt`: the same rows as a `|`-separated table
- `processing_log.csv`: one row per filing with its status (`ok`, `no_product_language`, `no_product_found`, `skipped_items`, errors), for the report's statistics

Downloads are cached in `cache/`, so a rerun (for example with a different model) doesn't hit the SEC again. Results are checkpointed after each company.

## Tests

```bash
python -m pytest tests -q
```

The tests run offline. SEC responses and the LLM are replaced with fixtures, covering parsing, filtering, validation and CSV output.
