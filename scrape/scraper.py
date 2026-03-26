import time
import random
import logging
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from config import USER_AGENTS, REQUEST_DELAY

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_page_content(url):
    """
    Fetches the HTML content of a URL using a headless browser (Playwright)
    to bypass Cloudflare and WAF protections.
    """
    logging.info(f"Playwright fetching URL: {url}")
    
    with sync_playwright() as p:
        # Launch Chromium with anti-bot configurations
        browser = p.chromium.launch(headless=True)
        user_agent = random.choice(USER_AGENTS)
        
        context = browser.new_context(
            user_agent=user_agent,
            viewport={"width": 1920, "height": 1080},
            java_script_enabled=True,
            bypass_csp=True
        )
        
        page = context.new_page()
        
        try:
            # Increase timeout and wait for network to be idle
            response = page.goto(url, timeout=60000, wait_until="networkidle")
            
            if response and response.status >= 400:
                logging.error(f"HTTP Error {response.status} for {url}")
                return None
            
            # Additional sleep to let any CF JS challenge complete
            time.sleep(5)
            
            html_content = page.content()
            return html_content
            
        except Exception as e:
            logging.error(f"Playwright error fetching {url}: {e}")
            return None
        finally:
            browser.close()

def parse_html(html, url):
    """
    Extracts the main text content and title from HTML using BeautifulSoup.
    """
    if not html:
        return None
        
    soup = BeautifulSoup(html, 'lxml')
    
    # Extract title
    title_tag = soup.find('title')
    title = title_tag.get_text(strip=True) if title_tag else url
    
    # Strip out unwanted elements that pollute the knowledge base
    unwanted_tags = [
        'script', 'style', 'nav', 'footer', 'header', 
        'aside', 'form', 'iframe', 'noscript', 'button',
        '.menu', '.sidebar', '#cookie-banner'
    ]
    
    for tag in unwanted_tags:
        if tag.startswith('.'):
            for el in soup.select(tag):
                el.decompose()
        elif tag.startswith('#'):
            for el in soup.select(tag):
                el.decompose()
        else:
            for el in soup.find_all(tag):
                el.decompose()
                
    # Try to find the main content area first
    main_content = soup.find('main') or soup.find('article') or soup.find(id='content')
    
    # Fallback to body if no main semantic tag is found
    if not main_content:
        main_content = soup.find('body')
        
    if not main_content:
        return None
        
    # Get text and clean up whitespace
    text = main_content.get_text(separator=' ', strip=True)
    
    # Remove excessive blank lines and spaces
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    
    return {
        "url": url,
        "title": title,
        "content": text
    }

def scrape_url(url):
    """
    Main pipeline function for a single URL using Playwright
    """
    html = get_page_content(url)
    if html:
        return parse_html(html, url)
    return None
