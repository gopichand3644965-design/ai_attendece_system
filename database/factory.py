from typing import Optional
from database.base import DatabaseAdapter
import importlib

class DatabaseFactory:
    """
    Instantiates the correct DatabaseAdapter based on the database_type.
    """

    @staticmethod
    def create_adapter(database_type: str, connection_url: str) -> DatabaseAdapter:
        db_type = database_type.lower()
        
        if db_type == "json":
            from database.adapters.json_adapter import JSONAdapter
            return JSONAdapter(connection_url)
        
        elif db_type == "sqlite":
            from database.adapters.sqlite_adapter import SQLiteAdapter
            return SQLiteAdapter(connection_url)
            
        elif db_type == "postgresql":
            from database.adapters.postgres_adapter import PostgresAdapter
            return PostgresAdapter(connection_url)
            
        elif db_type == "mongodb":
            from database.adapters.mongodb_adapter import MongoDBAdapter
            return MongoDBAdapter(connection_url)
            
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
