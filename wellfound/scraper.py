#!/usr/bin/env python3
import argparse, csv, hashlib, os, random, sys, time, json
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# --- Selenium setup ---
DRIVER = None
def ensure_selenium(headless=True):
    global DRIVER
    if DRIVER is not None:
        return DRIVER
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,900")
    DRIVER = webdriver.Chrome(ChromeDriverManager().install(), options=opts)
    return DRIVER

# --- Helpers ---
UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

def h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def build_url(tmpl: str, query: str, location: str, page: int) -> str:
    return (tmpl
            .replace("{{query}}", quote_plus(query))
            .replace("{{location}}", quote_plus(location))
            .replace("{{page}}", str(page)))

def allowed_by_robots(url: str, user_agent="*") -> bool:
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True

# --- Fetch functions ---
def fetch(url: str, use_selenium=False, headless=True):
    if use_selenium:
        driver = ensure_selenium(headless=headless)
        driver.get(url)
        time.sleep(random.uniform(2, 3))
        return driver.page_source
    headers = {"User-Agent": random.choice(UA_LIST)}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text

def fetch_wellfound(url, max_scrolls=5, pause=2):
    driver = ensure_selenium(headless=True)
    driver.get(url)
    time.sleep(3)

    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause + random.uniform(0, 1.5))
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    return driver.page_source

# --- Parsing ---
def parse_jobs(html: str, selectors: dict, site_name: str) -> list[dict]:
    # use built-in parser instead of lxml
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for card in soup.select(selectors["job_card"]):
        def sel_text(sel):
            el = card.select_one(sel) if sel else None
            return el.get_text(strip=True) if el else ""

        title = sel_text(selectors.get("title"))
        company = sel_text(selectors.get("company"))
        location = sel_text(selectors.get("location"))
        posted_at = sel_text(selectors.get("posted_at"))

        url = ""
        link_el = card.select_one(selectors.get("link")) if selectors.get("link") else None
        if link_el:
            url = link_el.get("href", "").strip()
            if url and url.startswith("/"):
                url = f"https://wellfound.com{url}"

        if not title:
            continue

        job_id = h(f"{site_name}|{title}|{company}|{url}")
        out.append({
            "job_id": job_id,
            "site": site_name,
            "title": title,
            "company": company,
            "location": location,
            "posted_at": posted_at,
            "url": url,
            "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    return out

# --- Output ---
def ensure_output_dirs():
    os.makedirs("output", exist_ok=True)

def export_csv(jobs: list[dict]):
    ensure_output_dirs()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"output/jobs_{ts}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "job_id","site","title","company","location","posted_at","url","scraped_at"
        ])
        writer.writeheader()
        writer.writerows(jobs)
    return path

# --- Main ---
def main():
    ap = argparse.ArgumentParser(description="Scrape job listings safely.")
    ap.add_argument("--site", required=True, help="Site key from sites.json (e.g., wellfound)")
    ap.add_argument("--query", default="data-analyst", help="Search keywords")
    ap.add_argument("--location", default="mumbai", help="Location filter")
    ap.add_argument("--pages", type=int, default=1, help="Unused for Wellfound (kept for API parity)")
    ap.add_argument("--headless", action="store_true", help="Run Selenium headless")
    args = ap.parse_args()

    with open("sites.json", "r", encoding="utf-8") as f:
        sites = json.load(f)

    if args.site not in sites:
        print(f"[ERR] Unknown site '{args.site}'. Available: {', '.join(sites.keys())}", file=sys.stderr)
        sys.exit(2)

    config = sites[args.site]
    tmpl = config["start_url_template"]
    selectors = config["selectors"]

    url = build_url(tmpl, args.query, args.location, 1)
    if not allowed_by_robots(url):
        print(f"[SKIP] Disallowed by robots.txt: {url}", file=sys.stderr)
        sys.exit(0)

    jobs = []
    try:
        if args.site == "wellfound":
            html = fetch_wellfound(url, max_scrolls=5)
        else:
            html = fetch(url, use_selenium=config.get("uses_selenium", False), headless=args.headless or True)
        jobs = parse_jobs(html, selectors, args.site)
        print(f"[OK] {args.site}: parsed {len(jobs)} jobs. URL={url}")
    except Exception as e:
        print(f"[WARN] Error on {url}: {e}", file=sys.stderr)

    if jobs:
        csv_path = export_csv(jobs)
        print(f"[DONE] Saved {len(jobs)} jobs to {csv_path}")
    else:
        print("[DONE] No jobs found.")

if __name__ == "__main__":
    main()
