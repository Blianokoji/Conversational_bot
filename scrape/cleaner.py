"""
scrape/cleaner.py
Converts raw HTML into clean plain text suitable for embedding.
"""

from bs4 import BeautifulSoup, Comment


# Tags whose entire subtree is noise
_STRIP_TAGS = {
    "script", "style", "noscript", "iframe",
    "nav", "footer", "header",
    "form", "button", "svg", "img",
}


def clean_html(html: str) -> str:
    """
    Parse *html* and return the visible body text with boilerplate removed.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove comment nodes
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove noise tags
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Try to grab the main content area; fall back to <body>
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"id": "content"})
        or soup.find("div", {"class": "entry-content"})
        or soup.find("div", {"class": "page-content"})
        or soup.body
    )

    if main is None:
        return ""

    # Get text, collapse whitespace, keep paragraph breaks
    lines: list[str] = []
    for element in main.stripped_strings:
        lines.append(element)

    text = "\n".join(lines)

    # Collapse multiple blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()
