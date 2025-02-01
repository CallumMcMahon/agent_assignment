from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchParameters(BaseModel):
    q: str
    gl: str | None
    hl: str | None
    autocorrect: bool | None
    page: int | None
    type: str | None
    engine: str | None


class Attributes(BaseModel):
    Headquarters: str
    CEO: str
    Founded: str
    Sales: str
    Products: str
    Founders: str
    Subsidiaries: str
    Missing: str


class KnowledgeGraph(BaseModel):
    title: str
    type: str
    website: str
    imageUrl: str
    description: str
    descriptionSource: str
    descriptionLink: str
    attributes: Attributes


class Sitelink(BaseModel):
    title: str
    link: str


class Attributes1(BaseModel):
    Products: Optional[str] = None
    Founders: Optional[str] = None
    Founded: Optional[str] = None
    Industry: Optional[str] = None
    Related_People: Optional[str] = Field(None, alias='Related People')
    Date: Optional[str] = None
    Areas_Of_Involvement: Optional[str] = Field(None, alias='Areas Of Involvement')


class OrganicItem(BaseModel):
    title: str
    link: str
    snippet: str
    sitelinks: Optional[List[Sitelink]] = None
    position: int
    attributes: Optional[Attributes1] = None
    date: Optional[str] = None


class PeopleAlsoAskItem(BaseModel):
    question: str
    snippet: str
    title: str
    link: str


class RelatedSearch(BaseModel):
    query: str


class SerperResponse(BaseModel):
    searchParameters: SearchParameters
    knowledgeGraph: KnowledgeGraph | None
    organic: List[OrganicItem]
    peopleAlsoAsk: List[PeopleAlsoAskItem]
    relatedSearches: List[RelatedSearch]
