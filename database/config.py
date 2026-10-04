import os
import json

CONFIG_FILE = "db_config.json"

class DatabaseConfig:
    """
    Handles loading database configuration from a persistent file or env vars.
    """
    
    @staticmethod
    def _read_config() -> dict:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @staticmethod
    def set_config(db_type: str, db_url: str):
        with open(CONFIG_FILE, "w") as f:
            json.dump({
                "DATABASE_TYPE": db_type,
                "DATABASE_URL": db_url
            }, f, indent=4)

    @staticmethod
    def get_database_type() -> str:
        """Get the configured database type. Environment variables take priority."""
        env_val = os.environ.get("DATABASE_TYPE")
        if env_val:
            return env_val.lower()
        cfg = DatabaseConfig._read_config()
        if "DATABASE_TYPE" in cfg:
            return cfg["DATABASE_TYPE"].lower()
        return "json"

    @staticmethod
    def get_database_url() -> str:
        """Get the configured database URL. Environment variables take priority."""
        env_val = os.environ.get("DATABASE_URL")
        if env_val:
            return env_val
        cfg = DatabaseConfig._read_config()
        if "DATABASE_URL" in cfg:
            return cfg["DATABASE_URL"]
        return "local"
