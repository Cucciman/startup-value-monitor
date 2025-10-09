# --- Normalize sector taxonomy -------------------------------------------------
def _normalize_sector_names(df: pd.DataFrame, col: str = "sector") -> pd.DataFrame:
    """
    Map assorted sector labels into a canonical Sifted-style taxonomy:
    {Fintech, B2B SaaS, Climate, Deeptech, Healthtech, Consumer, AI-native}
    This helps keep charts consistent even when source labels vary
    across platforms (Crowdcube, Seedrs, BME Growth, etc.).
    """
    if col not in df.columns:
        return df

    # Clean up whitespace and casing
    raw = (
        df[col].astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Direct mapping for known variations
    mapping = {
        # SaaS
        "saas": "B2B SaaS", "SaaS": "B2B SaaS", "SaAS": "B2B SaaS", "B2B SaaS": "B2B SaaS",
        "software": "B2B SaaS", "enterprise software": "B2B SaaS",

        # Climate / Cleantech
        "climate": "Climate", "Climate": "Climate", "Climate Tech": "Climate",
        "climatetech": "Climate", "greentech": "Climate", "cleantech": "Climate",
        "energy": "Climate", "renewables": "Climate",

        # Fintech
        "fintech": "Fintech", "FinTech": "Fintech", "Fin tech": "Fintech", "payments": "Fintech",

        # Deeptech
        "deeptech": "Deeptech", "DeepTech": "Deeptech", "Deep tech": "Deeptech",
        "semiconductors": "Deeptech", "robotics": "Deeptech", "space": "Deeptech",

        # Healthtech
        "healthtech": "Healthtech", "HealthTech": "Healthtech", "digital health": "Healthtech",
        "medtech": "Healthtech", "biotech": "Healthtech",

        # Consumer
        "consumer": "Consumer", "ConsumerTech": "Consumer", "ecommerce": "Consumer",
        "marketplace": "Consumer", "D2C": "Consumer",

        # AI-native
        "ai": "AI-native", "AI": "AI-native", "AI native": "AI-native", "GenAI": "AI-native",
        "machine learning": "AI-native",
    }

    # Apply the direct replacements
    normalized = raw.replace(mapping)

    # Fallbacks for messy labels using keyword inference
    def infer(v: str) -> str:
        s = v.lower()
        if "fintech" in s or "payment" in s or "bank" in s:
            return "Fintech"
        if "saas" in s or "software" in s or "b2b" in s:
            return "B2B SaaS"
        if any(k in s for k in ["climate", "green", "clean", "energy", "renew"]):
            return "Climate"
        if any(k in s for k in ["deep", "semicon", "robot", "space", "quantum"]):
            return "Deeptech"
        if any(k in s for k in ["health", "med", "bio"]):
            return "Healthtech"
        if any(k in s for k in ["consumer", "ecom", "marketplace"]):
            return "Consumer"
        if any(k in s for k in ["ai", "genai", "ml", "artificial intelligence"]):
            return "AI-native"
        return v  # leave unchanged if uncertain

    normalized = normalized.apply(infer)

    allowed = {"Fintech", "B2B SaaS", "Climate", "Deeptech", "Healthtech", "Consumer", "AI-native"}
    df[col] = normalized.where(normalized.isin(allowed), normalized)
    return df
# -------------------------------------------------------------------------------

