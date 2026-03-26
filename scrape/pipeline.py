"""
scrape/pipeline.py
End-to-end scraping pipeline: discover (sitemap) → fetch → clean → chunk → save.
"""

import json
import os
import time

from config import PROCESSED_DIR, RAW_DIR, REQUEST_DELAY
from scrape.chunker import chunk_text
from scrape.cleaner import clean_html
from scrape.scraper import scrape_url
from scrape.seed_urls import SEED_URLS


def _ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def run_scrape_pipeline():
    """Execute the full scrape → clean → chunk pipeline and persist results."""
    _ensure_dirs()

    # ── Step 1: discover URLs from sitemap ───────────────
    print("=" * 60)
    print("STEP 1 — Discovering URLs from sitemap …")
    print("=" * 60)
    urls = SEED_URLS
    print(f"\n→ Found {len(urls)} target seed URLs.\n")

    # ── Step 2: fetch & clean each page ──────────────────
    print("=" * 60)
    print("STEP 2 — Fetching & cleaning pages …")
    print("=" * 60)
    raw_pages: list[dict] = []
    errors: list[str] = []

    for i, url in enumerate(urls, 1):
        print(f"  [{i}/{len(urls)}] {url}")
        page = scrape_url(url)
        if page is None:
            errors.append(url)
            continue

        clean_text = page["content"] # text is already cleaned by scraper.py
        if not clean_text:
            print(f"    ↳ empty after cleaning, skipping")
            continue

        raw_pages.append({
            "url": page["url"],
            "title": page["title"],
            "content": clean_text,
        })
        time.sleep(REQUEST_DELAY)

    # Save raw cleaned pages
    raw_path = os.path.join(RAW_DIR, "pages.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_pages, f, indent=2, ensure_ascii=False)
    print(f"\n→ Saved {len(raw_pages)} raw pages to {raw_path}")
    if errors:
        print(f"→ {len(errors)} pages failed: {errors}")

    # ── Step 3: chunk the content ────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Chunking content …")
    print("=" * 60)
    chunks: list[dict] = []
    for page in raw_pages:
        page_chunks = chunk_text(page["content"])
        for idx, chunk in enumerate(page_chunks):
            chunks.append({
                "id": f"{page['url']}#chunk-{idx}",
                "url": page["url"],
                "title": page["title"],
                "chunk_index": idx,
                "text": chunk,
            })

    chunks_path = os.path.join(PROCESSED_DIR, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"→ Generated {len(chunks)} chunks from {len(raw_pages)} pages.")
    print(f"→ Saved to {chunks_path}")

    # ── Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  URLs discovered  : {len(urls)}")
    print(f"  Pages scraped    : {len(raw_pages)}")
    print(f"  Pages failed     : {len(errors)}")
    print(f"  Total chunks     : {len(chunks)}")
    print(f"  Raw output       : {raw_path}")
    print(f"  Processed output : {chunks_path}")


if __name__ == "__main__":
    run_scrape_pipeline()
