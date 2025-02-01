from __future__ import annotations

from typing import List, Optional

import httpx
from pydantic import BaseModel

from readability import Document
from markdownify import markdownify as md


class SearchParameters(BaseModel):
    q: str
    type: str
    engine: str


class Attributes(BaseModel):
    Missing: str


class OrganicItem(BaseModel):
    title: str
    link: str
    snippet: str
    position: int
    date: Optional[str] = None
    attributes: Optional[Attributes] = None


class PeopleAlsoAskItem(BaseModel):
    question: str
    snippet: str
    title: str
    link: str


class RelatedSearch(BaseModel):
    query: str


class SerperResponse(BaseModel):
    searchParameters: SearchParameters
    organic: List[OrganicItem]
    peopleAlsoAsk: List[PeopleAlsoAskItem]
    relatedSearches: List[RelatedSearch]
    credits: int




def fetch_and_convert_to_markdown(url: str) -> str:
    """
    Fetch the web page from the given URL, extract its main content,
    and convert it to Markdown using markdownify.
    
    Args:
        url (str): The URL of the web page.
    
    Returns:
        str: The main content of the page in Markdown format.
    """
    # Download the web page content
    response = httpx.get(url)
    response.raise_for_status()  # Raise exception for HTTP errors

    # Extract the main content using readability
    doc = Document(response.text)
    main_html = doc.summary()

    # Convert HTML to Markdown using markdownify
    markdown_content = md(main_html)
    
    return markdown_content