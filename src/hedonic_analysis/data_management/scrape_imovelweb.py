"""One-time scraper for ImovelWeb property listings in Curitiba.

This script is NOT part of the pytask reproducible pipeline. It collects
raw property listing data that becomes the starting input for the pipeline.

The scraper works in two phases:
    Phase 1: Collect listing URLs from search result pages.
    Phase 2: Visit each listing URL and extract full property details.

Both phases support resume from previous partial runs via incremental CSV saves.

Usage:
    pixi run python -m hedonic_analysis.data_management.scrape_imovelweb
    pixi run python -m hedonic_analysis.data_management.scrape_imovelweb --phase 1
    pixi run python -m hedonic_analysis.data_management.scrape_imovelweb --phase 2
    pixi run python -m hedonic_analysis.data_management.scrape_imovelweb --headless
"""

import argparse
import json
import logging
import random
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait

# =====================================================================================
# Constants
# =====================================================================================

_SRC = Path(__file__).parent.parent.resolve()
_DATA_DIR = _SRC / "data"

URLS_CSV = _DATA_DIR / "imovelweb_urls.csv"
RAW_CSV = _DATA_DIR / "imovelweb_raw.csv"
LOG_FILE = _DATA_DIR / "scraper.log"

BASE_URL = "https://www.imovelweb.com.br"
PROPERTY_TYPES = ("apartamentos", "casas")
MAX_PAGES = 5
MIN_DELAY = 2.0
MAX_DELAY = 5.0
PAGE_LOAD_TIMEOUT = 30
MAX_RETRIES = 3
BATCH_SIZE = 50
_MIN_ADDRESS_PARTS = 2

OUTPUT_COLUMNS = (
    "URL",
    "Endereco",
    "Bairro",
    "Tipo",
    "Categoria",
    "Area_total_m2",
    "Area_util_m2",
    "N_quartos",
    "N_banheiros",
    "N_vagas",
    "Preco",
    "IPTU",
    "Idade_anos",
    "Adicionais",
    "Areas_comuns",
    "Areas_privativas",
    "Descricao",
    "Planta",
)

NEIGHBORHOODS = (
    "Abranches",
    "Água Verde",
    "Ahú",
    "Alto Boqueirão",
    "Alto da Glória",
    "Alto da Rua XV",
    "Atuba",
    "Augusta",
    "Bacacheri",
    "Bairro Alto",
    "Barreirinha",
    "Batel",
    "Bigorrilho",
    "Boa Vista",
    "Bom Retiro",
    "Boqueirão",
    "Butiatuvinha",
    "Cabral",
    "Cachoeira",
    "Cajuru",
    "Campina do Siqueira",
    "Campo Comprido",
    "Campo de Santana",
    "Capão da Imbuia",
    "Capão Raso",
    "Cascatinha",
    "Caximba",
    "Centro",
    "Centro Cívico",
    "Cidade Industrial de Curitiba",
    "Cristo Rei",
    "Fanny",
    "Fazendinha",
    "Ganchinho",
    "Guabirotuba",
    "Guaíra",
    "Hauer",
    "Hugo Lange",
    "Jardim Botânico",
    "Jardim das Américas",
    "Jardim Social",
    "Juvevê",
    "Lamenha Pequena",
    "Lindóia",
    "Mercês",
    "Mossunguê",
    "Novo Mundo",
    "Orleans",
    "Parolin",
    "Pilarzinho",
    "Pinheirinho",
    "Portão",
    "Prado Velho",
    "Rebouças",
    "Riviera",
    "Santa Cândida",
    "Santa Felicidade",
    "Santa Quitéria",
    "Santo Inácio",
    "São Braz",
    "São Francisco",
    "São João",
    "São Lourenço",
    "São Miguel",
    "Seminário",
    "Sítio Cercado",
    "Taboão",
    "Tarumã",
    "Tatuquara",
    "Tingui",
    "Uberaba",
    "Umbará",
    "Vila Izabel",
    "Vista Alegre",
    "Xaxim",
)

# =====================================================================================
# Logging
# =====================================================================================


def _setup_logging():
    """Configure logging to both file and console.

    Returns:
        The configured logger instance.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("imovelweb_scraper")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# =====================================================================================
# CLI
# =====================================================================================


def _parse_args():
    """Parse command-line arguments.

    Returns:
        Parsed arguments with ``headless`` and ``phase`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Scrape ImovelWeb property listings in Curitiba.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode (lower success rate against bot detection).",
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "both"],
        default="both",
        help="Run only phase 1 (URLs), phase 2 (details), or both (default).",
    )
    return parser.parse_args()


# =====================================================================================
# Utility helpers
# =====================================================================================


def _slugify_neighborhood(name):
    """Convert a Brazilian neighborhood name to an ImovelWeb URL slug.

    Args:
        name: The neighborhood name in Portuguese (e.g., "Água Verde").

    Returns:
        The URL slug (e.g., "agua-verde").
    """
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_text = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    slug = re.sub(r"\s+", "-", ascii_text.strip().lower())
    return re.sub(r"[^a-z0-9-]", "", slug)


def _rate_limit():
    """Sleep for a random duration between MIN_DELAY and MAX_DELAY."""
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))  # noqa: S311


# =====================================================================================
# Driver management
# =====================================================================================


def _create_driver(*, headless=False):
    """Create and configure an undetected Chrome WebDriver instance.

    Args:
        headless: Whether to run Chrome without a visible window.

    Returns:
        A configured undetected_chromedriver.Chrome instance.
    """
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=pt-BR")
    options.add_argument("--disable-blink-features=AutomationControlled")

    if headless:
        options.add_argument("--headless=new")

    driver = uc.Chrome(options=options, version_main=145)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


def _safe_get(driver, url, *, retries=MAX_RETRIES, logger=None):
    """Navigate to a URL with retry logic and DataDome handling.

    Args:
        driver: The Selenium WebDriver instance.
        url: The URL to navigate to.
        retries: Maximum number of retry attempts.
        logger: Logger instance for messages.

    Returns:
        True if the page loaded successfully, False if all retries exhausted.
    """
    log = logger or logging.getLogger("imovelweb_scraper")

    for attempt in range(1, retries + 1):
        try:
            driver.get(url)
            time.sleep(1)

            page_source = driver.page_source.lower()

            if "datadome" in page_source or "pardon our interruption" in page_source:
                wait_time = 15 * attempt
                log.warning(
                    "DataDome challenge detected on %s, waiting %ds (attempt %d/%d)",
                    url,
                    wait_time,
                    attempt,
                    retries,
                )
                time.sleep(wait_time)
                continue

            if "403" in driver.title.lower() or "access denied" in page_source:
                wait_time = 10 * attempt
                log.warning(
                    "403/Access denied on %s, waiting %ds (attempt %d/%d)",
                    url,
                    wait_time,
                    attempt,
                    retries,
                )
                time.sleep(wait_time)
                continue

        except TimeoutException:
            log.warning(
                "Timeout on %s (attempt %d/%d)",
                url,
                attempt,
                retries,
            )
            time.sleep(2 * attempt)

        except WebDriverException:
            log.exception(
                "WebDriver error on %s (attempt %d/%d)",
                url,
                attempt,
                retries,
            )
            time.sleep(2 * attempt)

        else:
            return True

    log.error("All %d retries exhausted for %s", retries, url)
    return False


# =====================================================================================
# Phase 1: URL collection
# =====================================================================================


def _build_search_url(property_type, neighborhood_slug, page):
    """Build an ImovelWeb search results URL.

    Args:
        property_type: Either "apartamentos" or "casas".
        neighborhood_slug: The URL-safe neighborhood name.
        page: Page number (1-5). Page 1 has no pagination suffix.

    Returns:
        The fully formed search URL.
    """
    base = f"{BASE_URL}/{property_type}-venda-{neighborhood_slug}-curitiba"
    if page == 1:
        return f"{base}.html"
    return f"{base}-pagina-{page}.html"


def _normalize_href(href):
    """Convert a relative href to an absolute URL if needed.

    Args:
        href: The raw href string from an anchor element.

    Returns:
        The absolute URL.
    """
    if href.startswith("/"):
        return BASE_URL + href
    return href


def _collect_urls_by_selector(soup, selector, *, filter_propriedades=False):
    """Collect unique listing URLs matching a CSS selector.

    Args:
        soup: A BeautifulSoup object of the search results page.
        selector: CSS selector string to find anchor elements.
        filter_propriedades: Whether to require '/propriedades/' in href.

    Returns:
        A list of unique absolute listing URLs.
    """
    urls = []
    for link in soup.select(selector):
        href = link.get("href", "")
        if not href:
            continue
        if filter_propriedades and "/propriedades/" not in href:
            continue
        href = _normalize_href(href)
        if href not in urls:
            urls.append(href)
    return urls


def _extract_listing_urls(page_source):
    """Extract individual listing URLs from a search results page.

    Tries three strategies in order: data-qa attributes, card layout
    selectors, and generic anchor tag matching.

    Args:
        page_source: The full HTML source of a search results page.

    Returns:
        A list of absolute listing URLs found on the page.
    """
    soup = BeautifulSoup(page_source, "html.parser")

    urls = _collect_urls_by_selector(
        soup, "a[data-qa='POSTING_CARD_DESCRIPTION']",
    )
    if urls:
        return urls

    urls = _collect_urls_by_selector(
        soup,
        "div[data-qa='POSTING_CARD_GALLERY'] a, "
        "div.postingCardLayout a[href*='/propriedades/']",
        filter_propriedades=True,
    )
    if urls:
        return urls

    return _collect_urls_by_selector(
        soup, "a[href*='/propriedades/']", filter_propriedades=True,
    )


def _has_next_page(page_source, current_page):
    """Check whether there are more search result pages to scrape.

    Args:
        page_source: The full HTML source of the search results page.
        current_page: The current page number (1-indexed).

    Returns:
        True if there is a next page that should be scraped.
    """
    if current_page >= MAX_PAGES:
        return False

    soup = BeautifulSoup(page_source, "html.parser")

    next_link = soup.select_one(
        "a[data-qa='PAGING_NEXT'], "
        "li.next a, "
        "a.paging-next"
    )
    return next_link is not None


def _load_existing_urls():
    """Load already-collected URL records from the existing URLs CSV.

    Returns:
        A DataFrame of existing records, or an empty DataFrame if the file
        does not exist.
    """
    if URLS_CSV.exists():
        return pd.read_csv(URLS_CSV, encoding="utf-8-sig")
    return pd.DataFrame(columns=["url", "property_type", "neighborhood", "source_page"])


def _deduplicate_urls(urls_df):
    """Remove duplicate URLs, keeping the first occurrence.

    Args:
        urls_df: DataFrame with a ``url`` column.

    Returns:
        Deduplicated DataFrame.
    """
    return urls_df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)


def _append_to_csv(df, path):
    """Append a DataFrame to a CSV file, creating it if needed.

    Args:
        df: The data to append.
        path: The CSV file path.
    """
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")


def _scrape_neighborhood_urls(driver, logger, prop_type, neighborhood):
    """Scrape listing URLs for one property type / neighborhood combo.

    Args:
        driver: The Selenium WebDriver instance.
        logger: Logger instance.
        prop_type: Either "apartamentos" or "casas".
        neighborhood: The neighborhood name in Portuguese.

    Returns:
        A list of record dicts with url, property_type, neighborhood,
        and source_page keys.
    """
    slug = _slugify_neighborhood(neighborhood)
    batch_records = []

    for page in range(1, MAX_PAGES + 1):
        url = _build_search_url(prop_type, slug, page)
        success = _safe_get(driver, url, logger=logger)

        if not success:
            logger.warning("Failed to load search page: %s", url)
            break

        listing_urls = _extract_listing_urls(driver.page_source)

        if not listing_urls:
            logger.debug(
                "No listings found on %s page %d for %s",
                prop_type,
                page,
                neighborhood,
            )
            break

        batch_records.extend(
            {
                "url": listing_url,
                "property_type": prop_type,
                "neighborhood": neighborhood,
                "source_page": page,
            }
            for listing_url in listing_urls
        )

        if not _has_next_page(driver.page_source, page):
            break

        _rate_limit()

    return batch_records


def collect_listing_urls(driver, logger):
    """Phase 1: Collect all unique listing URLs from search pages.

    Iterates over all combinations of property types and neighborhoods,
    paginating up to MAX_PAGES per combination. Saves incrementally and
    supports resume from a previous partial run.

    Args:
        driver: The Selenium WebDriver instance.
        logger: Logger instance.

    Returns:
        DataFrame with columns: url, property_type, neighborhood, source_page.
    """
    existing_df = _load_existing_urls()
    done_combos = set()
    if not existing_df.empty and "property_type" in existing_df.columns:
        for _, row in existing_df.iterrows():
            done_combos.add((row["property_type"], row["neighborhood"]))

    total_combos = len(PROPERTY_TYPES) * len(NEIGHBORHOODS)
    completed = 0

    for prop_type in PROPERTY_TYPES:
        for neighborhood in NEIGHBORHOODS:
            if (prop_type, neighborhood) in done_combos:
                completed += 1
                continue

            batch_records = _scrape_neighborhood_urls(
                driver, logger, prop_type, neighborhood,
            )

            if batch_records:
                _append_to_csv(pd.DataFrame(batch_records), URLS_CSV)

            completed += 1
            logger.info(
                "[Phase 1] %d/%d combos done | %s/%s: %d URLs",
                completed,
                total_combos,
                prop_type,
                neighborhood,
                len(batch_records),
            )

            _rate_limit()

    full_df = pd.read_csv(URLS_CSV, encoding="utf-8-sig")
    deduped = _deduplicate_urls(full_df)
    deduped.to_csv(URLS_CSV, index=False, encoding="utf-8-sig")

    logger.info(
        "Phase 1 complete: %d unique URLs collected (from %d total)",
        len(deduped),
        len(full_df),
    )
    return deduped


# =====================================================================================
# Phase 2: Detail extraction
# =====================================================================================


def _extract_json_ld(page_source):
    """Extract JSON-LD structured data from a listing detail page.

    Args:
        page_source: The full HTML source of a listing detail page.

    Returns:
        The parsed JSON-LD dict, or None if not found.
    """
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("@type") in (
                    "Product",
                    "RealEstateListing",
                    "Residence",
                    "Apartment",
                    "House",
                    "SingleFamilyResidence",
                ):
                    return item
        elif isinstance(data, dict):
            schema_type = data.get("@type", "")
            if any(
                t in str(schema_type)
                for t in ("Product", "RealEstate", "Residence", "Apartment", "House")
            ):
                return data

    return None


def _extract_from_json_ld(json_ld):
    """Map JSON-LD fields to the target schema.

    Args:
        json_ld: The parsed JSON-LD dictionary.

    Returns:
        A dict with keys matching output columns, values as strings or None.
    """
    record = {}

    record["Endereco"] = json_ld.get("name") or json_ld.get("url")

    address = json_ld.get("address", {})
    if isinstance(address, dict):
        record["Endereco"] = address.get("streetAddress", record.get("Endereco"))
        record["Bairro"] = address.get("addressLocality")

    offers = json_ld.get("offers", {})
    if isinstance(offers, dict):
        record["Preco"] = offers.get("price") or offers.get("lowPrice")
    elif isinstance(offers, list) and offers:
        record["Preco"] = offers[0].get("price")

    record["Descricao"] = json_ld.get("description")

    return {k: str(v) if v is not None else None for k, v in record.items()}


def _extract_age(soup):
    """Extract property age in years from the listing page.

    Uses two strategies:
    1. The ``<li class="icon-feature">`` containing an
       ``<i class="icon-antiguedad">`` icon, with text like ``" 14 anos "``.
    2. Full-page regex requiring qualifying keywords after "anos".

    Args:
        soup: A BeautifulSoup object of the listing page.

    Returns:
        The age as a string, or None if not found.
    """
    # Strategy 1: icon-antiguedad inside icon-feature li (most reliable).
    # Captures both numeric ("14 anos") and text ("Breve Lançamento").
    icon = soup.select_one("li.icon-feature i.icon-antiguedad")
    if icon and icon.parent:
        text = " ".join(icon.parent.stripped_strings)
        if text:
            return text

    # Strategy 2: regex on full page text with qualifying keywords.
    age_match = re.search(
        r"(\d+)\s*anos?\s*(?:de\s*)?(?:constru|idade|antiguid)",
        soup.get_text(),
        re.IGNORECASE,
    )
    if age_match:
        return age_match.group(1)

    return None


def _extract_from_html(soup):
    """Extract listing details from HTML as a fallback to JSON-LD.

    Args:
        soup: A BeautifulSoup object of the listing page.

    Returns:
        A dict with keys matching output columns.
    """
    record = {}

    price_el = soup.select_one(
        "[data-qa='POSTING_CARD_PRICE'], "
        ".price-value, "
        "span.price, "
        "div.price-container span"
    )
    if price_el:
        record["Preco"] = price_el.get_text(strip=True)
    else:
        price_match = re.search(r"R\$\s?[\d.,]+", soup.get_text())
        if price_match:
            record["Preco"] = price_match.group()

    address_el = soup.select_one(
        "[data-qa='POSTING_CARD_LOCATION'], "
        "h2.posting-location, "
        ".location-container, "
        "span.location"
    )
    if address_el:
        text = address_el.get_text(strip=True)
        parts = [p.strip() for p in text.split(",")]
        if len(parts) >= _MIN_ADDRESS_PARTS:
            record["Endereco"] = ", ".join(parts[:-1])
            record["Bairro"] = parts[-1]
        else:
            record["Endereco"] = text

    desc_el = soup.select_one(
        "[data-qa='POSTING_DESCRIPTION'], "
        ".section-description--content, "
        "#verDatosDescripcionAmpl, "
        "div.description-content"
    )
    if desc_el:
        record["Descricao"] = desc_el.get_text(separator=" ", strip=True)

    tipo_el = soup.select_one(
        "[data-qa='POSTING_CARD_PROPERTY_TYPE'], "
        "span.property-type, "
        "h1.title-type-sup-property"
    )
    if tipo_el:
        record["Tipo"] = tipo_el.get_text(strip=True)

    iptu_match = re.search(
        r"IPTU[:\s]*R?\$?\s*([\d.,]+)",
        soup.get_text(),
        re.IGNORECASE,
    )
    if iptu_match:
        record["IPTU"] = iptu_match.group(1)

    age = _extract_age(soup)
    if age is not None:
        record["Idade_anos"] = age

    return {k: str(v) if v is not None else None for k, v in record.items()}


_FEATURE_PATTERNS = {
    "N_quartos": r"(\d+)\s*(?:quartos?|dormit[oó]rios?|dorms?)",
    "N_banheiros": r"(\d+)\s*(?:banheiros?|ba[nñ]os?|wc)",
    "N_vagas": r"(\d+)\s*(?:vagas?|garagens?|estac)",
    "Area_total_m2": r"(\d+[\.,]?\d*)\s*m[²2]\s*tot(?:ais?|al|\.)?",
    "Area_util_m2": r"(\d+[\.,]?\d*)\s*m[²2]\s*[úu]t(?:eis|il)?",
}

# "Label BEFORE number" patterns (e.g. "Sup. total 120 m²").
_LABEL_BEFORE_PATTERNS = {
    "Area_total_m2": (
        r"(?:sup(?:erf[ií]cie)?\.?\s*total|[áa]rea\s*total)"
        r"\s*:?\s*(\d+[\.,]?\d*)\s*m[²2]"
    ),
    "Area_util_m2": (
        r"(?:sup(?:erf[ií]cie)?\.?\s*[úu]til|[áa]rea\s*[úu]til)"
        r"\s*:?\s*(\d+[\.,]?\d*)\s*m[²2]"
    ),
}


def _search_patterns(text, patterns, record):
    """Apply regex patterns to text and fill missing keys in record.

    Args:
        text: The text to search.
        patterns: Dict mapping column names to regex patterns.
        record: The mutable record dict to update.
    """
    for key, pattern in patterns.items():
        if key not in record:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                record[key] = match.group(1).replace(",", ".")


def _extract_area_from_icon(soup, icon_class):
    """Extract an area value from an icon-feature ``<li>``.

    Args:
        soup: A BeautifulSoup object of the listing page.
        icon_class: CSS class of the ``<i>`` icon (e.g. ``icon-stotal``).

    Returns:
        Area as a string (e.g. ``"244"``), or None if not found.
    """
    icon = soup.select_one(f"li.icon-feature i.{icon_class}")
    if icon and icon.parent:
        text = " ".join(icon.parent.stripped_strings)
        match = re.search(r"(\d+[\.,]?\d*)\s*m[²2]", text)
        if match:
            return match.group(1).replace(",", ".")
    return None


def _extract_property_features(soup):
    """Extract bedroom, bathroom, parking, and area counts.

    Args:
        soup: A BeautifulSoup object of the listing page.

    Returns:
        Dict with keys: N_quartos, N_banheiros, N_vagas,
        Area_total_m2, Area_util_m2.
    """
    record = {}

    total = _extract_area_from_icon(soup, "icon-stotal")
    if total:
        record["Area_total_m2"] = total

    useful = _extract_area_from_icon(soup, "icon-scubierta")
    if useful:
        record["Area_util_m2"] = useful

    full_text = soup.get_text(separator=" ")

    _search_patterns(full_text, _FEATURE_PATTERNS, record)
    _search_patterns(full_text, _LABEL_BEFORE_PATTERNS, record)

    if "Area_total_m2" not in record and "Area_util_m2" not in record:
        h2 = soup.select_one("h2.title-type-sup-property")
        if h2:
            h2_text = h2.get_text(strip=True)
            area_match = re.search(r"(\d+[\.,]?\d*)\s*m[²2]", h2_text)
            if area_match:
                record["Area_util_m2"] = area_match.group(1).replace(",", ".")

    return record


def _extract_additional_details(soup):
    """Extract the floor plan image URL.

    Args:
        soup: A BeautifulSoup object of the listing page.

    Returns:
        Dict with key ``Planta`` if found, otherwise empty.
    """
    record = {}

    planta_el = soup.select_one(
        "[data-qa='FLOOR_PLAN'], "
        ".section-floor-plan, "
        "img[alt*='planta'], "
        "img[alt*='floor']"
    )
    if planta_el:
        record["Planta"] = planta_el.get("src") or planta_el.get_text(strip=True)

    return record


def _scroll_full_page(driver):
    """Scroll the entire page to trigger lazy-loaded React sections."""
    try:
        page_height = driver.execute_script(
            "return document.body.scrollHeight",
        )
        for pos in range(0, page_height, 600):
            driver.execute_script(f"window.scrollTo(0, {pos});")
            time.sleep(0.15)
    except WebDriverException:
        pass


def _read_tab_items(container, label):
    """Click a tab button's content and return deduplicated item strings."""
    items = []
    for selector in (
        "span[class*='description-text']",
        "span[class*='description']",
        "li span",
        "li",
    ):
        spans = container.find_elements(By.CSS_SELECTOR, selector)
        items = [
            s.text.strip() for s in spans
            if s.is_displayed() and s.text.strip()
        ]
        if items:
            break

    seen = set()
    unique = []
    for item in items:
        if item not in seen and item != label:
            seen.add(item)
            unique.append(item)
    return unique


def _extract_general_features(driver):
    """Click each tab in "Saiba mais" and extract feature items.

    Scrolls the full page first to trigger React lazy-loading, then
    uses ``WebDriverWait`` to find ``#reactGeneralFeatures``.  Each
    tab button is clicked via JavaScript and the visible content
    items are collected using multiple CSS selector fallbacks.

    Args:
        driver: The Selenium WebDriver instance (already on the listing page).

    Returns:
        Dict with keys ``Areas_comuns``, ``Areas_privativas``, and
        ``Adicionais`` (semicolon-separated strings).
    """
    record = {}

    _scroll_full_page(driver)

    try:
        container = WebDriverWait(driver, 5).until(
            expected_conditions.presence_of_element_located(
                (By.ID, "reactGeneralFeatures"),
            ),
        )
    except (TimeoutException, WebDriverException):
        return record

    driver.execute_script(
        "arguments[0].scrollIntoView({block:'center'});", container,
    )
    time.sleep(0.8)

    buttons = container.find_elements(By.TAG_NAME, "button")
    if not buttons:
        return record

    all_features = []

    for btn in buttons:
        label = btn.text.strip()
        if not label:
            try:
                label = btn.find_element(
                    By.CSS_SELECTOR, "span",
                ).text.strip()
            except NoSuchElementException:
                label = ""

        driver.execute_script("arguments[0].click();", btn)
        time.sleep(0.8)

        items = _read_tab_items(container, label)
        if not items:
            continue

        joined = "; ".join(items)
        all_features.extend(items)

        label_lower = label.lower()
        if "privativa" in label_lower:
            record["Areas_privativas"] = joined
        elif "comun" in label_lower or "comum" in label_lower:
            record["Areas_comuns"] = joined

    if all_features:
        record["Adicionais"] = "; ".join(dict.fromkeys(all_features))

    return record


def _merge_extractor_result(record, name, result):
    """Merge an extractor's result into the record dict.

    JSON-LD results overwrite existing values; other extractors only fill
    in keys that are still None.

    Args:
        record: The mutable record dict to update.
        name: The extractor name ("json_ld" or other).
        result: The dict returned by the extractor.
    """
    if name == "json_ld":
        json_data = _extract_from_json_ld(result)
        for key, value in json_data.items():
            if value is not None:
                record[key] = value
    else:
        for key, value in result.items():
            if record.get(key) is None and value is not None:
                record[key] = value


def _parse_listing_page(page_source, url):
    """Parse a single listing detail page into a flat data record.

    Combines JSON-LD extraction (preferred) with HTML fallback. Produces a
    dict with all 18 target columns.

    Args:
        page_source: The full HTML source of the listing page.
        url: The listing URL (included in the output record).

    Returns:
        Dict with all output columns.
    """
    logger = logging.getLogger("imovelweb_scraper")
    record = dict.fromkeys(OUTPUT_COLUMNS)
    record["URL"] = url

    soup = BeautifulSoup(page_source, "html.parser")

    extractors = [
        ("json_ld", lambda: _extract_json_ld(page_source)),
        ("html", lambda: _extract_from_html(soup)),
        ("features", lambda: _extract_property_features(soup)),
        ("details", lambda: _extract_additional_details(soup)),
    ]

    for name, extractor in extractors:
        try:
            result = extractor()
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.debug(
                "Extractor '%s' failed for %s", name, url, exc_info=True,
            )
            continue
        if result is not None:
            _merge_extractor_result(record, name, result)

    if record.get("Categoria") is None and "/apartamentos" in url:
        record["Categoria"] = "apartamentos"
    elif record.get("Categoria") is None and "/casas" in url:
        record["Categoria"] = "casas"

    return record


def _load_existing_details():
    """Load already-scraped detail records from the existing raw CSV.

    Returns:
        A DataFrame of existing records, or an empty DataFrame if the file
        does not exist.
    """
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    return pd.DataFrame(columns=list(OUTPUT_COLUMNS))


def scrape_listing_details(driver, urls_df, logger):
    """Phase 2: Visit each listing URL and extract full property details.

    Implements resume capability, incremental saving, rate limiting, and
    error isolation.

    Args:
        driver: The Selenium WebDriver instance.
        urls_df: DataFrame with at least a ``url`` column.
        logger: Logger instance.

    Returns:
        DataFrame with all output columns.
    """
    existing_df = _load_existing_details()
    already_scraped = set()
    if not existing_df.empty and "URL" in existing_df.columns:
        already_scraped = set(existing_df["URL"].tolist())

    pending_urls = [u for u in urls_df["url"] if u not in already_scraped]
    logger.info(
        "Phase 2: %d URLs pending, %d already scraped",
        len(pending_urls),
        len(already_scraped),
    )

    batch_records = []

    for i, url in enumerate(pending_urls):
        success = _safe_get(driver, url, logger=logger)

        if not success:
            logger.warning("Failed to load listing: %s", url)
            record = dict.fromkeys(OUTPUT_COLUMNS)
            record["URL"] = url
        else:
            try:
                record = _parse_listing_page(driver.page_source, url)
            except Exception:
                logger.exception("Error parsing listing: %s", url)
                record = dict.fromkeys(OUTPUT_COLUMNS)
                record["URL"] = url

            try:
                features = _extract_general_features(driver)
                for key, val in features.items():
                    if val and not record.get(key):
                        record[key] = val
            except (
                WebDriverException,
                NoSuchElementException,
                TimeoutException,
            ):
                logger.debug("General features extraction failed: %s", url)

        batch_records.append(record)

        if len(batch_records) >= BATCH_SIZE:
            _append_to_csv(pd.DataFrame(batch_records), RAW_CSV)
            logger.info(
                "[Phase 2] Saved batch | %d/%d done (%.1f%%)",
                i + 1,
                len(pending_urls),
                100 * (i + 1) / len(pending_urls),
            )
            batch_records.clear()

        _rate_limit()

    if batch_records:
        _append_to_csv(pd.DataFrame(batch_records), RAW_CSV)

    full_df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    deduped = full_df.drop_duplicates(subset=["URL"], keep="first").reset_index(
        drop=True,
    )
    deduped.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

    logger.info("Phase 2 complete: %d total records", len(deduped))
    return deduped


# =====================================================================================
# Main entry point
# =====================================================================================


def main():
    """Execute the two-phase ImovelWeb scraping pipeline."""
    args = _parse_args()
    logger = _setup_logging()

    logger.info(
        "Starting ImovelWeb scraper (phase=%s, headless=%s)",
        args.phase,
        args.headless,
    )

    driver = None
    try:
        driver = _create_driver(headless=args.headless)

        if args.phase in ("1", "both"):
            logger.info("=== Phase 1: Collecting listing URLs ===")
            urls_df = collect_listing_urls(driver, logger)
        else:
            urls_df = _load_existing_urls()
            if urls_df.empty:
                logger.error(
                    "No URLs file found at %s. Run phase 1 first.",
                    URLS_CSV,
                )
                return
            logger.info("Loaded %d existing URLs from %s", len(urls_df), URLS_CSV)

        if args.phase in ("2", "both"):
            logger.info("=== Phase 2: Scraping listing details ===")
            scrape_listing_details(driver, urls_df, logger)

        logger.info("Scraping pipeline finished.")

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Progress has been saved.")
    finally:
        if driver is not None:
            driver.quit()
            logger.info("Chrome driver closed.")


if __name__ == "__main__":
    main()
