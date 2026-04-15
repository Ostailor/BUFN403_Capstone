from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "ai_corpus"
DEFAULT_MANUAL_SOURCE_DIR = ROOT_DIR / "manual_sources"
DEFAULT_ROSTER_CSV = ROOT_DIR / "AI_Bank_Classification.csv"
DEFAULT_SEC_ZIP = ROOT_DIR / "10K_10Q_8K_DEF14A_combined_data.zip"
DEFAULT_TRANSCRIPT_ZIP = ROOT_DIR / "transcripts_final-20260304T030232Z-1-001.zip"
DEFAULT_AS_OF_DATE = date(2026, 3, 11)
DEFAULT_TRANSCRIPT_PERIODS = [
    (2024, 1),
    (2024, 2),
    (2024, 3),
    (2024, 4),
    (2025, 1),
    (2025, 2),
    (2025, 3),
    (2025, 4),
]
DEFAULT_CALL_REPORT_PERIODS = list(DEFAULT_TRANSCRIPT_PERIODS)
DEFAULT_10K_PERIODS = [(2024, 4), (2025, 4)]
DEFAULT_10Q_PERIODS = [
    (2024, 1),
    (2024, 2),
    (2024, 3),
    (2025, 1),
    (2025, 2),
    (2025, 3),
]
DEFAULT_PROXY_PERIODS = [(2024, 2), (2025, 2)]
QWEN_MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]
EMBEDDING_MODEL_CANDIDATES = [
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
]
CALL_REPORT_FIELDS = [
    "CERT",
    "REPDTE",
    "NAME",
    "ASSET",
    "DEP",
    "NETINC",
    "ROA",
    "ROE",
    "EQ",
    "NIMY",
    "OFFDOM",
]
STRUCTURED_METRIC_ALIASES = {
    "assets": ["asset", "assets", "balance sheet", "size"],
    "deposits": ["deposit", "deposits", "funding"],
    "net_income": ["net income", "earnings", "profit", "profits"],
    "roa": ["roa", "return on assets"],
    "roe": ["roe", "return on equity"],
    "equity": ["equity", "capital"],
    "nim": ["nim", "net interest margin"],
}
THEME_KEYWORDS = {
    "ai_strategy": [
        "artificial intelligence strategy",
        "ai strategy",
        "strategic ai",
        "genai strategy",
        "enterprise ai",
    ],
    "use_cases": [
        "fraud",
        "underwriting",
        "risk model",
        "contact center",
        "customer service",
        "chatbot",
        "copilot",
        "compliance",
        "operations",
        "trading",
    ],
    "investment_spend": [
        "investment",
        "spend",
        "budget",
        "capital allocation",
        "funding",
        "roadmap",
    ],
    "risk_controls": [
        "model risk",
        "controls",
        "governance",
        "guardrail",
        "responsible ai",
        "validation",
        "compliance",
    ],
    "governance": [
        "board",
        "oversight",
        "policy",
        "committee",
        "governance",
    ],
    "customer_facing_ai": [
        "customer",
        "advisor",
        "chatbot",
        "personalization",
        "digital assistant",
        "copilot",
    ],
    "operations_efficiency": [
        "efficiency",
        "productivity",
        "automation",
        "workflow",
        "back office",
        "operations",
    ],
    "vendors_partnerships": [
        "vendor",
        "partner",
        "partnership",
        "third party",
        "microsoft",
        "openai",
        "google",
        "anthropic",
    ],
    "measurable_outcomes": [
        "revenue",
        "cost save",
        "cost savings",
        "roi",
        "metric",
        "results",
        "improvement",
    ],
}

# ── Intent Classification ────────────────────────────────────────────
INTENT_LEVELS = {
    1: "Exploring",
    2: "Committing",
    3: "Deploying",
    4: "Scaling",
}

INTENT_SIGNAL_WORDS = {
    4: ["expanding", "enterprise-wide", "firm-wide", "increasing", "doubling down", "across all", "scaled", "scaling"],
    3: ["launched", "implemented", "using", "rolled out", "live", "operational", "deployed", "in production"],
    2: ["investing", "budgeting", "building", "planning", "allocating", "developing", "committed", "funding"],
    1: ["investigating", "considering", "evaluating", "studying", "piloting", "researching", "exploring", "assessing"],
}

APP_CATEGORIES = {
    "GenAI / LLMs": ["chatbot", "copilot", "generative ai", "large language model", "llm", "gpt", "prompt", "genai"],
    "Predictive ML": ["credit scoring", "forecasting", "propensity model", "regression", "prediction", "predictive"],
    "NLP / Text": ["document processing", "sentiment analysis", "entity extraction", "text mining", "natural language", "nlp"],
    "Computer Vision": ["check imaging", "id verification", "facial recognition", "ocr", "computer vision", "image"],
    "RPA / Automation": ["process automation", "straight-through processing", "workflow automation", "robotic", "rpa", "automation"],
    "Fraud / Risk Models": ["fraud detection", "aml", "transaction monitoring", "anomaly detection", "risk model", "anti-money"],
}

CLASSIFIER_BATCH_SIZE = 10
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.5


@dataclass(slots=True)
class CorpusPaths:
    root_dir: Path = ROOT_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    manual_source_dir: Path = DEFAULT_MANUAL_SOURCE_DIR
    roster_csv: Path = DEFAULT_ROSTER_CSV
    sec_zip: Path = DEFAULT_SEC_ZIP
    transcript_zip: Path = DEFAULT_TRANSCRIPT_ZIP

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def normalized_dir(self) -> Path:
        return self.output_dir / "normalized"

    @property
    def documents_dir(self) -> Path:
        return self.normalized_dir / "documents"

    @property
    def tables_dir(self) -> Path:
        return self.normalized_dir / "tables"

    @property
    def index_dir(self) -> Path:
        return self.output_dir / "index"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def benchmarks_dir(self) -> Path:
        return self.output_dir / "benchmarks"

    @property
    def manifest_csv(self) -> Path:
        return self.output_dir / "document_manifest.csv"

    @property
    def acquisition_log_csv(self) -> Path:
        return self.output_dir / "acquisition_log.csv"

    @property
    def chunks_jsonl(self) -> Path:
        return self.output_dir / "chunks.jsonl"

    @property
    def classifications_jsonl(self) -> Path:
        return self.output_dir / "classifications.jsonl"

    @property
    def classification_progress_json(self) -> Path:
        return self.output_dir / "classification_progress.json"

    @property
    def corpus_db(self) -> Path:
        return self.output_dir / "corpus.duckdb"

    @property
    def topic_findings_csv(self) -> Path:
        return self.output_dir / "topic_findings.csv"
