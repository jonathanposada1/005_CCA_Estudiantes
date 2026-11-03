"""
Herramientas extra para el agente investigador:

- optica_open_access_search: artículos open access de Optica Publishing Group
  (Optics Express, Optica, Biomedical Optics Express, etc.) vía la API de OpenAlex.
- site_search: búsqueda web restringida a los sitios que te interesan (vía Tavily).
- check_open_access: dado un DOI, dice si hay versión OA y dónde está el PDF (vía Unpaywall).

Variables de entorno (.env):
    OPENALEX_EMAIL   -> tu correo (recomendado; entra al "polite pool" de OpenAlex)
    UNPAYWALL_EMAIL  -> tu correo (obligatorio para Unpaywall; si falta usa OPENALEX_EMAIL)
    TAVILY_API_KEY   -> ya lo tienes si usas tavily_search_tool
    RESEARCH_SITES   -> lista separada por comas, p. ej. "opg.optica.org,arxiv.org,spie.org"
"""

import os
from functools import lru_cache
from typing import Optional

import requests

OPENALEX = "https://api.openalex.org"
MAILTO = os.getenv("OPENALEX_EMAIL", "")
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", MAILTO)
DEFAULT_SITES = [
    s.strip() for s in os.getenv("RESEARCH_SITES", "opg.optica.org").split(",") if s.strip()
]


# ---------------------------------------------------------------- helpers
def _params(extra: dict) -> dict:
    p = dict(extra)
    if MAILTO:
        p["mailto"] = MAILTO
    return p


@lru_cache(maxsize=32)
def _publisher_id(name: str) -> Optional[str]:
    """Busca el ID de OpenAlex de una editorial (se resuelve una vez y se cachea)."""
    r = requests.get(
        f"{OPENALEX}/publishers",
        params=_params({"search": name, "per-page": 1}),
        timeout=20,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    return results[0]["id"].rsplit("/", 1)[-1] if results else None


def _rebuild_abstract(inverted_index: Optional[dict]) -> str:
    """OpenAlex entrega el abstract como índice invertido; lo reconstruimos."""
    if not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


# ---------------------------------------------------------------- tools
def optica_open_access_search(
    query: str,
    max_results: int = 5,
    from_year: Optional[int] = None,
    publisher: str = "Optica Publishing Group",
) -> list[dict]:
    """
    Search open access articles published by Optica Publishing Group
    (Optics Express, Optica, Biomedical Optics Express, Optical Materials Express, ...).

    Args:
        query: Topic to search, in English works best (e.g. "metasurface holography").
        max_results: Number of articles to return (1-25).
        from_year: Only return articles published from this year on.
        publisher: Publisher name; change it to search another publisher's OA articles.

    Returns:
        List of articles with title, authors, journal, year, DOI, PDF/landing URL,
        license, citation count and abstract.
    """
    try:
        pid = _publisher_id(publisher)
        if not pid:
            return [{"error": f"Publisher not found in OpenAlex: {publisher}"}]

        filters = [f"primary_location.source.publisher_lineage:{pid}", "is_oa:true"]
        if from_year:
            filters.append(f"from_publication_date:{int(from_year)}-01-01")

        r = requests.get(
            f"{OPENALEX}/works",
            params=_params(
                {
                    "search": query,
                    "filter": ",".join(filters),
                    "per-page": max(1, min(int(max_results), 25)),
                }
            ),
            timeout=30,
        )
        r.raise_for_status()

        articles = []
        for w in r.json().get("results", []):
            oa_loc = w.get("best_oa_location") or {}
            source = (w.get("primary_location") or {}).get("source") or {}
            articles.append(
                {
                    "title": w.get("title"),
                    "authors": ", ".join(
                        a["author"]["display_name"] for a in w.get("authorships", [])[:6]
                    ),
                    "journal": source.get("display_name"),
                    "year": w.get("publication_year"),
                    "doi": w.get("doi"),
                    "oa_status": (w.get("open_access") or {}).get("oa_status"),
                    "pdf_url": oa_loc.get("pdf_url"),
                    "url": oa_loc.get("landing_page_url") or w.get("doi"),
                    "license": oa_loc.get("license"),
                    "cited_by": w.get("cited_by_count"),
                    "summary": _rebuild_abstract(w.get("abstract_inverted_index"))[:1500],
                }
            )
        return articles or [{"info": "No open access results for that query."}]
    except Exception as e:
        return [{"error": f"optica_open_access_search failed: {e}"}]


def site_search(query: str, domains: Optional[list[str]] = None, max_results: int = 5) -> list[dict]:
    """
    Web search restricted to specific websites of interest (default: RESEARCH_SITES env var,
    e.g. opg.optica.org).

    Args:
        query: What to search for.
        domains: Optional list of domains to restrict to, e.g. ["opg.optica.org", "spie.org"].
        max_results: Number of results (1-10).

    Returns:
        List of results with title, url and a content snippet.
    """
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        resp = client.search(
            query=query,
            include_domains=domains or DEFAULT_SITES,
            max_results=max(1, min(int(max_results), 10)),
        )
        return [
            {"title": r.get("title"), "url": r.get("url"), "content": r.get("content")}
            for r in resp.get("results", [])
        ] or [{"info": "No results on those sites."}]
    except Exception as e:
        return [{"error": f"site_search failed: {e}"}]


def check_open_access(doi: str) -> dict:
    """
    Check whether an article (by DOI) has a free, legal open access version and where to get it.

    Args:
        doi: Article DOI, e.g. "10.1364/OE.123456" (with or without https://doi.org/).

    Returns:
        Dict with is_oa, oa_status, pdf_url, landing_page and license.
    """
    if not UNPAYWALL_EMAIL:
        return {"error": "Set UNPAYWALL_EMAIL in .env to use check_open_access."}
    try:
        doi = doi.replace("https://doi.org/", "").strip()
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": UNPAYWALL_EMAIL},
            timeout=20,
        )
        r.raise_for_status()
        d = r.json()
        loc = d.get("best_oa_location") or {}
        return {
            "title": d.get("title"),
            "is_oa": d.get("is_oa"),
            "oa_status": d.get("oa_status"),
            "pdf_url": loc.get("url_for_pdf"),
            "landing_page": loc.get("url_for_landing_page"),
            "license": loc.get("license"),
        }
    except Exception as e:
        return {"error": f"check_open_access failed: {e}"}


# ---------------------------------------------------------------- registro
# Para conectar con un loop manual de tool calling (estilo OpenAI).
# Si usas aisuite con max_turns, basta con pasar las funciones directamente.
tool_mapping = {
    "optica_open_access_search": optica_open_access_search,
    "site_search": site_search,
    "check_open_access": check_open_access,
}

tool_defs = [
    {
        "type": "function",
        "function": {
            "name": "optica_open_access_search",
            "description": "Search open access articles from Optica Publishing Group journals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                    "from_year": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "site_search",
            "description": "Web search restricted to specific domains of interest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "domains": {"type": "array", "items": {"type": "string"}},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_open_access",
            "description": "Given a DOI, find a legal open access PDF via Unpaywall.",
            "parameters": {
                "type": "object",
                "properties": {"doi": {"type": "string"}},
                "required": ["doi"],
            },
        },
    },
]
