#config.py
@dataclass
class Config:
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost::11434"
    gemini_api_key: str = ""
    db_path: Path("~/.local/share/bantz/store.db")
    shell_confirm_destructive: bool = True
    language: str = "tr" #MarianMT active