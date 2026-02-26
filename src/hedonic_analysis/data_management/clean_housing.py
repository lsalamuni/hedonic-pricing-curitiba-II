"""Helper functions for cleaning raw ImovelWeb housing data.

This module provides the ``clean_housing`` function that transforms the raw
scraped CSV into a tidy DataFrame ready for hedonic regression analysis.
"""

import re
import unicodedata

import pandas as pd

# =====================================================================
# Valid Curitiba neighborhoods (75 official bairros)
# =====================================================================

VALID_BAIRROS: tuple[str, ...] = (
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

# =====================================================================
# Bairro corrections (typos / aliases -> canonical name)
# =====================================================================

BAIRRO_CORRECTIONS: dict[str, str] = {
    "Barigui": "Santo Inácio",
    "Alto da Rua Xv": "Alto da Rua XV",
    "Alto da XV": "Alto da Rua XV",
    "Alto da Xv": "Alto da Rua XV",
    "Alto": "Alto da Rua XV",
    "Caiua": "Capão Raso",
    "Champagnat": "Bigorrilho",
    "Cidade Industrial": "Cidade Industrial de Curitiba",
    "Cic": "Cidade Industrial de Curitiba",
    "CIC": "Cidade Industrial de Curitiba",
    "Itatiaia": "Cidade Industrial de Curitiba",
    "Neoville": "Cidade Industrial de Curitiba",
    "Ecoville": "Mossunguê",
    "Jardim Schaffer": "Vista Alegre",
    "Vila Lindoia": "Lindóia",
    "Agua Verde": "Água Verde",
    "Ahu": "Ahú",
    "Alto Boqueirao": "Alto Boqueirão",
    "Alto da Gloria": "Alto da Glória",
    "Capao da Imbuia": "Capão da Imbuia",
    "Capao Raso": "Capão Raso",
    "Guaira": "Guaíra",
    "Jardim Botanico": "Jardim Botânico",
    "Jardim das Americas": "Jardim das Américas",
    "Juveve": "Juvevê",
    "Lindoia": "Lindóia",
    "Merces": "Mercês",
    "Mossungue": "Mossunguê",
    "Portao": "Portão",
    "Reboucas": "Rebouças",
    "Santa Candida": "Santa Cândida",
    "Santa Quiteria": "Santa Quitéria",
    "Santo Inacio": "Santo Inácio",
    "Sao Braz": "São Braz",
    "Sao Francisco": "São Francisco",
    "Sao Joao": "São João",
    "Sao Lourenco": "São Lourenço",
    "Sao Miguel": "São Miguel",
    "Sitio Cercado": "Sítio Cercado",
    "Taboao": "Taboão",
    "Taruma": "Tarumã",
    "Boqueirao": "Boqueirão",
    "Umbara": "Umbará",
    "Bigorrilho/": "Bigorrilho",
    "Cetro": "Centro",
    "Taboao Curitiba Pr": "Taboão",
    "Cabral - Curitiba/pr": "Cabral",
    "Parque Tangua": "Pilarzinho",
}

# =====================================================================
# Compiled regex patterns
# =====================================================================

_PRICE_PATTERN = re.compile(r"R\$\s*([\d.]+)")

_OFFPLAN_PATTERN = re.compile(
    r"breve\s*lan.amento"
    r"|unidades\s*dispon.veis"
    r"|em\s*constru..o"
    r"|na\s*planta"
    r"|pr.\s*venda"
    r"|lan.amento"
    r"|entrega\s*prevista"
    r"|previs.o\s*de\s*entrega",
    re.IGNORECASE,
)

_APT_DESC_PATTERN = re.compile(
    r"apto\b|apartamento|edif.cio|studio"
    r"|loft|cobertura|kitnet|flat\b"
    r"|residencial|\bdorm\b|dormit.rio",
    re.IGNORECASE,
)

_CASA_DESC_PATTERN = re.compile(
    r"\bcasa\b|sobrado"
    r"|resid.ncia\s+a\s+venda"
    r"|condom.nio\s+fechado|village",
    re.IGNORECASE,
)

_AMENITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "Party_room": re.compile(
        r"sal.o\s*(?:de\s*)?festas?|espa.o\s*festas?",
        re.IGNORECASE,
    ),
    "Game_room": re.compile(
        r"sal.o\s*(?:de\s*)?jogos?|sala\s*(?:de\s*)?jogos?",
        re.IGNORECASE,
    ),
    "Gym": re.compile(
        r"gin.stica|fitness|academia|muscula..o",
        re.IGNORECASE,
    ),
    "Pool": re.compile(r"piscina", re.IGNORECASE),
    "Sauna": re.compile(r"sauna", re.IGNORECASE),
    "BBQ": re.compile(
        r"churrasqueira|churras|parrilla",
        re.IGNORECASE,
    ),
    "Gourmet_space": re.compile(
        r"espa.o\s*gourmet|gourmet", re.IGNORECASE,
    ),
    "Sports_court": re.compile(
        r"quadra|quadras?\s*(?:de\s*)?"
        r"(?:esporte|t.nis|poliesportiva|squash)",
        re.IGNORECASE,
    ),
    "Guardhouse": re.compile(
        r"guarita|portaria\s*24|port.o\s*eletr.nico",
        re.IGNORECASE,
    ),
    "Cameras": re.compile(
        r"c.mera|cftv|circuito\s*fechado"
        r"|monitoramento|vigil.ncia",
        re.IGNORECASE,
    ),
    "Balcony": re.compile(
        r"varanda|sacada|terra.o|balc.o",
        re.IGNORECASE,
    ),
    "Playground": re.compile(
        r"playground|parquinho|brinquedoteca"
        r"|espa.o\s*kids|espa.o\s*crian.as",
        re.IGNORECASE,
    ),
    "Elevator": re.compile(r"elevador", re.IGNORECASE),
}

# =====================================================================
# Outlier bounds
# =====================================================================

# =====================================================================
# Address patterns
# =====================================================================

_STREET_PREPS = re.compile(r"\b(De|Da|Do|Das|Dos|E)\b")

_ROMAN_NUMERALS = re.compile(
    r"\b(Ii|Iii|Iv|Vi|Vii|Viii|Ix"
    r"|Xi|Xii|Xiii|Xiv|Xv|Xvi|Xvii|Xviii|Xix|Xx)\b",
)

_KNOWN_TYPES_PATTERN = re.compile(
    r"^(?:Rua|Avenida|Alameda|Travessa|Praça"
    r"|Rodovia|Estrada|Largo|Via|Linha) ",
    re.IGNORECASE,
)

# =====================================================================
# Column rename mapping (Portuguese → English lowercase)
# =====================================================================

_COLUMN_RENAME = {
    "URL": "url",
    "Endereco": "address",
    "Bairro": "neighborhood",
    "Area_total_m2": "total_area_m2",
    "Area_util_m2": "usable_area_m2",
    "Idade_anos": "age_years",
    "Preco": "price",
}

# =====================================================================
# Outlier bounds
# =====================================================================

_MIN_PRICE = 50_000
_MAX_PRICE = 20_000_000
_MIN_AREA = 15
_MAX_AREA = 2_000
_MIN_PRICE_PER_M2 = 500
_MAX_PRICE_PER_M2 = 100_000
_IQR_K = 3.0

# =====================================================================
# Columns to drop at the end of the pipeline
# =====================================================================

_COLUMNS_TO_DROP = (
    "Tipo",
    "Categoria",
    "Adicionais",
    "Areas_comuns",
    "Areas_privativas",
    "Descricao",
    "Planta",
    "IPTU",
    "N_quartos",
    "N_banheiros",
    "N_vagas",
)

# =====================================================================
# Text normalization and bairro lookup
# =====================================================================


def _normalize_text(text):
    """Strip accents and lowercase a string for fuzzy matching.

    Args:
        text: Raw string, possibly with Portuguese diacritics.

    Returns:
        ASCII-lowercased version of *text*.
    """
    nfkd = unicodedata.normalize("NFKD", str(text))
    ascii_text = "".join(
        c for c in nfkd if unicodedata.category(c) != "Mn"
    )
    return ascii_text.lower().strip()


def _build_bairro_lookup():
    """Build a mapping from normalized names to canonical bairros.

    Returns:
        Dict mapping normalized (ASCII, lowercase) bairro names to
        their canonical accented form.
    """
    lookup = {}
    for bairro in VALID_BAIRROS:
        lookup[_normalize_text(bairro)] = bairro
    for alias, canonical in BAIRRO_CORRECTIONS.items():
        lookup[_normalize_text(alias)] = canonical
    return lookup


_BAIRRO_LOOKUP = _build_bairro_lookup()
_BAIRRO_KEYS = sorted(_BAIRRO_LOOKUP, key=len, reverse=True)


# =====================================================================
# Private helper functions
# =====================================================================


def _drop_empty_rows(df):
    """Remove rows where all columns except URL are null.

    These originate from 403-blocked listing pages during scraping.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame without empty rows.
    """
    data_cols = [c for c in df.columns if c != "URL"]
    mask = df[data_cols].isna().all(axis=1)
    return df[~mask].copy()


def _extract_price(series):
    """Parse messy price strings into Float32 values.

    Handles formats like ``'vendaR$ 2.300.000Me avisar...'``
    and ``'R$ 450.000'``.

    Args:
        series: Series of raw price strings.

    Returns:
        Series of numeric prices as ``Float32Dtype``.
    """
    text = series.astype("string")
    extracted = text.str.extract(
        r"R\$\s*([\d.]+)", expand=False,
    )
    cleaned = extracted.str.replace(".", "", regex=False)
    return pd.to_numeric(
        cleaned, errors="coerce",
    ).astype(pd.Float32Dtype())


def _match_bairro_tail(text):
    """Match a known neighborhood at the end of a description.

    Args:
        text: Full description string from a listing.

    Returns:
        Canonical bairro name, or ``pd.NA`` if no match.
    """
    if pd.isna(text):
        return pd.NA
    text = str(text).strip()
    if not text:
        return pd.NA
    normalized = _normalize_text(text)
    for key in _BAIRRO_KEYS:
        if not normalized.endswith(key):
            continue
        pos = len(normalized) - len(key)
        if pos == 0 or not normalized[pos - 1].isalnum():
            return _BAIRRO_LOOKUP[key]
    return pd.NA


def _extract_bairro(df):
    """Extract neighborhood from the tail of each description.

    ImovelWeb appends the neighborhood name at the end of every
    listing description.  This function matches that tail text
    against the 75 official Curitiba bairros (plus known aliases).

    Args:
        df: DataFrame with a ``Descricao`` column.

    Returns:
        Categorical Series of canonical bairro names.
    """
    bairro = df["Descricao"].map(_match_bairro_tail)
    cat_type = pd.CategoricalDtype(sorted(VALID_BAIRROS))
    return bairro.astype(cat_type)


def _classify_category(df):
    """Classify each listing as Apartamento or Casa.

    Uses the listing URL as the primary signal (the scraper queried
    ``/apartamentos`` and ``/casas`` separately) and falls back to
    keyword matching in the description text.

    Args:
        df: DataFrame with ``URL`` and ``Descricao`` columns.

    Returns:
        Categorical Series with categories Apartamento and Casa.
    """
    url = df["URL"].fillna("").str.lower()
    desc = df["Descricao"].fillna("").str.lower()

    is_apt_url = url.str.contains("/apartamentos", regex=False)
    is_casa_url = url.str.contains("/casas", regex=False)

    is_apt_desc = desc.str.contains(
        _APT_DESC_PATTERN, na=False,
    )
    is_casa_desc = desc.str.contains(
        _CASA_DESC_PATTERN, na=False,
    )

    result = pd.Series(
        None, index=df.index, dtype="object",
    )
    result[is_apt_url] = "Apartamento"
    mask_casa_url = is_casa_url & result.isna()
    result[mask_casa_url] = "Casa"
    mask_apt_desc = is_apt_desc & result.isna()
    result[mask_apt_desc] = "Apartamento"
    mask_casa_desc = is_casa_desc & result.isna()
    result[mask_casa_desc] = "Casa"

    cat_type = pd.CategoricalDtype(["Apartamento", "Casa"])
    return result.astype(cat_type)


def _smart_title(text):
    """Title-case a string, keeping prepositions lowercase.

    Handles Portuguese prepositions (de, da, do, das, dos, e) and
    preserves Roman numerals (XV, II, etc.).

    Args:
        text: Raw address string.

    Returns:
        Title-cased string with correct prepositions.
    """
    if pd.isna(text) or not str(text).strip():
        return pd.NA
    titled = str(text).title()
    titled = _STREET_PREPS.sub(
        lambda m: m.group().lower(), titled,
    )
    titled = _ROMAN_NUMERALS.sub(
        lambda m: m.group().upper(), titled,
    )
    return titled


def _clean_endereco(series):
    """Standardize address strings.

    Expands abbreviations (R. → Rua, Av. → Avenida, Al. → Alameda,
    Tv. → Travessa), applies title case to ALL-CAPS addresses, adds
    'Rua' prefix when no street type is present, and inserts a comma
    before the house number.

    Args:
        series: Series of raw address strings.

    Returns:
        Cleaned address Series.
    """
    s = series.astype("string").str.strip()

    # Remove "Endereco:" prefix
    s = s.str.replace(r"^Endereco:\s*", "", regex=True)

    # Remove "Type:" form (colon after type name)
    s = s.str.replace(
        r"^(Rua|Avenida|Alameda|Travessa|Pra[cç]a"
        r"|Rodovia|Estrada):\s*",
        r"\1 ",
        regex=True,
        case=False,
    )

    # Expand abbreviations
    s = s.str.replace(r"^R\.\s*", "Rua ", regex=True)
    s = s.str.replace(r"^Av\.\s*", "Avenida ", regex=True)
    s = s.str.replace(r"^Al\.\s*", "Alameda ", regex=True)
    s = s.str.replace(r"^Tv\.\s*", "Travessa ", regex=True)
    s = s.str.replace(
        r"^P[çc]\.\s*", "Praça ", regex=True,
    )
    s = s.str.replace(r"^Rod\.\s*", "Rodovia ", regex=True)

    # Strip apartment identifiers
    s = s.str.replace(r"\s*Apto:.*$", "", regex=True)
    s = s.str.replace(
        r"\s*Apt\.?\s*\d+.*$", "", regex=True,
    )
    s = s.str.replace(
        r"\s*Apartamento\s*\d+.*$", "", regex=True,
    )

    # Remove Nº/nº/N°/n° and surrounding punctuation
    s = s.str.replace(
        r",?\s*[Nn][º°]\.?\s*,?", " ", regex=True,
    )
    # Remove standalone "n"/"N" used as número before digits
    s = s.str.replace(
        r",?\s*\b[Nn]\b\s*,?\s*(?=\d)", " ", regex=True,
    )

    # Remove stray dashes: "Colombo -, 21" → "Colombo, 21"
    s = s.str.replace(r"\s+-\s*,", ",", regex=True)
    s = s.str.replace(r"\s+-\s+(?=\d)", " ", regex=True)

    # Remove CEP patterns and everything after
    s = s.str.replace(
        r"\s*-?\s*[Cc][Ee][Pp]:?.*$", "", regex=True,
    )
    # Remove city name "Curitiba" and trailing info
    s = s.str.replace(
        r",\s*Curitiba\b.*$", "", regex=True,
    )

    # Fix extra spaces before commas
    s = s.str.replace(r"\s+,", ",", regex=True)

    # Normalize whitespace
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    # Title case (fixes ALL CAPS addresses)
    s = s.map(_smart_title)

    # Add 'Rua' to addresses without a known type prefix
    has_type = s.str.contains(
        _KNOWN_TYPES_PATTERN, na=True,
    )
    s = s.where(has_type, "Rua " + s)

    # Add comma before house number (last digits at end of string)
    s = s.str.replace(
        r"(?<!,)\s+(\d+)\s*$", r", \1", regex=True,
    )

    # Remove trailing text after house number (neighborhood, etc.)
    s = s.str.replace(
        r"(,\s*\d+)\s*,.*$", r"\1", regex=True,
    )

    return s


def _detect_offplan(series):
    """Detect off-plan properties from description text.

    Matches keywords such as *lançamento*, *na planta*, and
    *em construção*.

    Args:
        series: Series of description strings.

    Returns:
        Binary Int8 Series (1 = off-plan, 0 = not).
    """
    matches = series.fillna("").str.contains(
        _OFFPLAN_PATTERN, na=False,
    )
    return matches.astype(pd.Int8Dtype())


def _extract_amenities(series):
    """Extract 13 binary amenity columns from descriptions.

    Each column indicates whether the listing description mentions
    a specific amenity (1) or not (0).

    Args:
        series: Series of description strings.

    Returns:
        DataFrame with 13 Int8 amenity columns.
    """
    text = series.fillna("")
    frames = {}
    for name, pattern in _AMENITY_PATTERNS.items():
        frames[name] = text.str.contains(
            pattern, na=False,
        ).astype(pd.Int8Dtype())
    return pd.DataFrame(frames, index=series.index)


def _create_count_dummies(df, col, prefix, max_val):
    """Create dummy variables from a numeric count column.

    The reference category is value 1 (or 0 for parking).
    The last dummy captures *max_val + 1* or more.

    Args:
        df: DataFrame to modify.
        col: Name of the count column.
        prefix: Prefix for dummy column names (e.g. ``BEDROOM_``).
        max_val: Number of dummy columns to create.

    Returns:
        DataFrame with new dummy columns appended.
    """
    counts = pd.to_numeric(
        df[col], errors="coerce",
    ).fillna(0)
    for i in range(1, max_val + 1):
        if i == max_val:
            df[f"{prefix}{i}"] = (
                counts >= i + 1
            ).astype(pd.Int8Dtype())
        else:
            df[f"{prefix}{i}"] = (
                counts == i + 1
            ).astype(pd.Int8Dtype())
    return df


def _cast_numeric_columns(df):
    """Ensure area and price columns use nullable Float32.

    Args:
        df: DataFrame to modify.

    Returns:
        DataFrame with casted columns.
    """
    for col in ("Area_util_m2", "Preco"):
        df[col] = pd.to_numeric(
            df[col], errors="coerce",
        ).astype(pd.Float32Dtype())
    return df


def _iqr_outliers(series):
    """Flag outliers using the IQR method with k=3.

    Args:
        series: Numeric Series.

    Returns:
        Boolean Series (True = outlier).
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - _IQR_K * iqr
    upper = q3 + _IQR_K * iqr
    return (series < lower) | (series > upper)


def _detect_outliers(df):
    """Flag outlier observations using IQR and domain bounds.

    Outliers are flagged but **not** dropped, so they can be used
    for sensitivity analysis in the regression stage.

    Args:
        df: DataFrame with ``Preco`` and ``Area_util_m2`` columns.

    Returns:
        Int8 Series (1 = outlier, 0 = not).
    """
    price = df["Preco"]
    area = df["Area_util_m2"]

    price_bounds = (price < _MIN_PRICE) | (price > _MAX_PRICE)
    area_bounds = (area < _MIN_AREA) | (area > _MAX_AREA)

    price_iqr = _iqr_outliers(price)
    area_iqr = _iqr_outliers(area)

    price_per_m2 = price / area
    pm2_bounds = (
        (price_per_m2 < _MIN_PRICE_PER_M2)
        | (price_per_m2 > _MAX_PRICE_PER_M2)
    )

    outlier = (
        price_bounds
        | area_bounds
        | price_iqr
        | area_iqr
        | pm2_bounds
    )
    return outlier.fillna(value=False).astype(pd.Int8Dtype())


def _fail_if_bairro_not_valid(series):
    """Raise if any bairro value is not in the official set.

    Args:
        series: Categorical bairro Series.

    Raises:
        ValueError: If invalid neighborhood names are found.
    """
    valid = set(VALID_BAIRROS)
    present = set(series.dropna().unique())
    bad = present - valid
    if bad:
        msg = f"Invalid bairros found: {sorted(bad)}"
        raise ValueError(msg)


def _fail_if_price_negative(series):
    """Raise if any price value is negative.

    Args:
        series: Numeric price Series.

    Raises:
        ValueError: If negative prices are found.
    """
    negative = (series.dropna() < 0).sum()
    if negative:
        msg = f"Negative prices found: {negative} rows"
        raise ValueError(msg)


# =====================================================================
# Public API
# =====================================================================


def clean_housing(raw_df):
    """Clean raw ImovelWeb housing data for hedonic analysis.

    Applies a 17-step pipeline: drop empty / duplicate rows, parse
    prices, extract neighborhoods, classify property type, standardize
    addresses, drop invalid rows, detect off-plan status, extract
    amenity features, create count dummies, cast dtypes, flag outliers,
    drop intermediate columns, and rename to English lowercase.

    Args:
        raw_df: Raw DataFrame loaded from ``imovelweb_raw.csv``.

    Returns:
        Cleaned DataFrame with ~30 English lowercase columns.
    """
    df = raw_df.copy()

    # 1. Drop empty rows (URL-only, from 403 blocks)
    df = _drop_empty_rows(df)

    # 2. Drop duplicate URLs
    df = df.drop_duplicates(subset=["URL"])

    # 3. Parse price
    df["Preco"] = _extract_price(df["Preco"])

    # 4. Extract bairro from description tail
    df["Bairro"] = _extract_bairro(df)

    # 5. Drop non-Curitiba (unmatched bairro)
    df = df.dropna(subset=["Bairro"])

    # 6. Classify property category
    df["Category"] = _classify_category(df)

    # 7. Clean addresses
    df["Endereco"] = _clean_endereco(df["Endereco"])

    # 8. Drop rows without address or without house number
    df = df.dropna(subset=["Endereco"])
    df = df[df["Endereco"].str.strip().ne("")]
    df = df[df["Endereco"].str.contains(r"\d", na=False)]

    # 9. Drop rows without usable area or without price
    df = df.dropna(subset=["Area_util_m2", "Preco"])

    # 10. Detect off-plan properties
    df["Offplan"] = _detect_offplan(df["Descricao"])

    # 11. Extract 13 amenity features
    amenities = _extract_amenities(df["Descricao"])
    df = pd.concat([df, amenities], axis=1)

    # 12. Create count dummies
    df = _create_count_dummies(df, "N_quartos", "BEDROOM_", 4)
    df = _create_count_dummies(df, "N_banheiros", "BATHROOM_", 4)
    df = _create_count_dummies(df, "N_vagas", "PARKING_", 2)

    # 13. Cast numeric columns
    df = _cast_numeric_columns(df)

    # 14. Flag outliers (kept, not dropped)
    df["Outlier"] = _detect_outliers(df)

    # 15. Drop text / useless columns
    df = df.drop(columns=list(_COLUMNS_TO_DROP), errors="ignore")

    # 16. Validate
    _fail_if_bairro_not_valid(df["Bairro"])
    _fail_if_price_negative(df["Preco"])

    # 17. Rename columns to English lowercase
    df = df.rename(columns=_COLUMN_RENAME)
    df.columns = df.columns.str.lower()

    # 18. Move URL to last column
    if "url" in df.columns:
        cols = [c for c in df.columns if c != "url"] + ["url"]
        df = df[cols]

    return df.reset_index(drop=True)
