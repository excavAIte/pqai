from bs4 import BeautifulSoup
import requests
import structlog
from requests.exceptions import Timeout, RequestException


log = structlog.getLogger(__name__)
class GooglePatentScraper:
    """Scraper class for Google Patent data"""

    def __init__(self, patent_id: str):
        """
        Args:
            patent_id (str): Google patent id
        """
        self.patent_url = f"https://patents.google.com/patent/{patent_id}/en"

    def get_soup(self):
        """Get beautifulsoup object of google patent page"""
        try:
            response = requests.get(self.patent_url, timeout=10)
            if response.status_code != 200:
                log.error("Failed to retrieve the google patent page")

            return BeautifulSoup(response.text, "html.parser")
        except Timeout:
            log.error("Request timed out for URL: %s", self.patent_url)
            raise
        except RequestException as e:
            log.error("Failed to fetch data from URL: %s. Error: %s", self.patent_url, e)
            raise

    def get_patent_description(self):
        soup = self.get_soup()
        # Extract description of patent
        description_section = soup.find("section", {"itemprop": "description"})

        if description_section:
            return description_section.get_text().strip()
        log.error("Description information not found", patent_url=self.patent_url)
        return None

