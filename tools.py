
from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
from rich import print
import requests as request
from bs4 import BeautifulSoup
import os


load_dotenv()

# Tavily API client
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY .env file mein nahi mili!")

tavily = TavilyClient(api_key=tavily_api_key)


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information.
    Return titles, URLs, and summaries of the top results.
    """
    result = tavily.search(
        query=query,
        max_results=5
    )

    out = []
    for r in result["results"]:
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n")

    return "\n-----\n".join(out)






@tool
def scrape_url(url: str) -> str:
    """
    Scrape and return clean text content from the given URL for deeper reading.
    """
    try:
        resp = request.get(url,timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style","nav","footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)[:3000]
    except Exception as e:
        return f"Error scraping the URL: {str(e)}"

