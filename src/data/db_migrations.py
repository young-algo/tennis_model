#!/usr/bin/env python
"""
Database Migration System

This script manages the evolution of the database schema through migrations.
It supports creating, applying, reverting, and listing migrations.
"""

import os
import sys
import json
import sqlite3
import datetime
import logging
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('db_migrations')

# Path constants
DATA_DIR = Path('data')
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
DB_PATH = PROCESSED_DATA_DIR / 'tennis.db'
MIGRATIONS_DIR = Path('src') / 'data' / 'migrations'


class MigrationManager:
    """Manages database migrations"""
    
    def __init__(self, db_path=DB_PATH):
        """Initialize the migration manager"""
        self.db_path = db_path
        self.conn = None
        self.setup_migrations_table()
    
    def setup_migrations_table(self):
        """Set up the migrations tracking table if it doesn't exist"""
        # Ensure the database directory exists
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create migrations table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_at TEXT,
            description TEXT
        )
        ''')
        
        self.conn.commit()
    
    def get_applied_migrations(self):
        """Get a list of already applied migrations"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT migration_id FROM schema_migrations ORDER BY migration_id")
        return [row[0] for row in cursor.fetchall()]
    
    def get_available_migrations(self):
        """Get a list of all available migrations in the migrations directory"""
        MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
        migration_files = sorted([f.stem for f in MIGRATIONS_DIR.glob("*.json")])
        return migration_files
    
    def get_pending_migrations(self):
        """Get a list of migrations that haven't been applied yet"""
        applied = self.get_applied_migrations()
        available = self.get_available_migrations()
        return [m for m in available if m not in applied]
    
    def create_migration(self, description):
        """Create a new migration file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize description for filename
        safe_desc = "".join([c if c.isalnum() else "_" for c in description]).lower()
        migration_id = f"{timestamp}_{safe_desc}"
        
        # Create migration file
        migration_file = MIGRATIONS_DIR / f"{migration_id}.json"
        MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Template for migration
        migration_data = {
            "migration_id": migration_id,
            "description": description,
            "up": {
                "operations": [
                    # Example operations
                    # {"type": "add_column", "table": "matches", "column": "new_column", "definition": "TEXT"}
                    # {"type": "create_index", "name": "idx_new_column", "table": "matches", "column": "new_column"}
                    # {"type": "raw_sql", "sql": "ALTER TABLE matches ADD COLUMN new_column TEXT;"}
                ]
            },
            "down": {
                "operations": [
                    # Reverse operations for rollback
                    # {"type": "drop_column", "table": "matches", "column": "new_column"}
                    # {"type": "drop_index", "name": "idx_new_column"}
                    # {"type": "raw_sql", "sql": "ALTER TABLE matches DROP COLUMN new_column;"}
                ]
            }
        }
        
        with open(migration_file, 'w') as f:
            json.dump(migration_data, f, indent=2)
        
        logger.info(f"Created migration: {migration_file}")
        return migration_id
    
    def apply_migration(self, migration_id):
        """Apply a specific migration"""
        migration_file = MIGRATIONS_DIR / f"{migration_id}.json"
        
        if not migration_file.exists():
            logger.error(f"Migration file not found: {migration_file}")
            return False
        
        try:
            # Load migration data
            with open(migration_file, 'r') as f:
                migration = json.load(f)
            
            # Execute operations
            cursor = self.conn.cursor()
            
            for op in migration["up"]["operations"]:
                if op["type"] == "add_column":
                    cursor.execute(f"ALTER TABLE {op['table']} ADD COLUMN {op['column']} {op['definition']}")
                
                elif op["type"] == "create_index":
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS {op['name']} ON {op['table']} ({op['column']})")
                
                elif op["type"] == "drop_column":
                    # SQLite doesn't directly support dropping columns, would need more complex solution
                    logger.warning("Drop column operation not directly supported in SQLite")
                
                elif op["type"] == "drop_index":
                    cursor.execute(f"DROP INDEX IF EXISTS {op['name']}")
                
                elif op["type"] == "raw_sql":
                    cursor.execute(op["sql"])
                
                else:
                    logger.warning(f"Unknown operation type: {op['type']}")
            
            # Record migration as applied
            cursor.execute(
                "INSERT INTO schema_migrations (migration_id, applied_at, description) VALUES (?, ?, ?)",
                (migration_id, datetime.datetime.now().isoformat(), migration["description"])
            )
            
            self.conn.commit()
            logger.info(f"Applied migration: {migration_id}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error applying migration {migration_id}: {e}")
            return False
    
    def revert_migration(self, migration_id):
        """Revert a specific migration"""
        migration_file = MIGRATIONS_DIR / f"{migration_id}.json"
        
        if not migration_file.exists():
            logger.error(f"Migration file not found: {migration_file}")
            return False
        
        # Check if migration was applied
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM schema_migrations WHERE migration_id = ?", (migration_id,))
        if not cursor.fetchone():
            logger.error(f"Migration {migration_id} has not been applied")
            return False
        
        try:
            # Load migration data
            with open(migration_file, 'r') as f:
                migration = json.load(f)
            
            # Execute down operations
            for op in migration["down"]["operations"]:
                if op["type"] == "drop_column":
                    # SQLite doesn't directly support dropping columns, would need more complex solution
                    logger.warning("Drop column operation not directly supported in SQLite")
                
                elif op["type"] == "drop_index":
                    cursor.execute(f"DROP INDEX IF EXISTS {op['name']}")
                
                elif op["type"] == "add_column":
                    # For rollback of an add_column, we'd need to recreate the table without the column
                    logger.warning("Rollback of add_column requires table recreation, not directly supported")
                
                elif op["type"] == "create_index":
                    cursor.execute(f"DROP INDEX IF EXISTS {op['name']}")
                
                elif op["type"] == "raw_sql":
                    cursor.execute(op["sql"])
                
                else:
                    logger.warning(f"Unknown operation type: {op['type']}")
            
            # Remove migration record
            cursor.execute("DELETE FROM schema_migrations WHERE migration_id = ?", (migration_id,))
            
            self.conn.commit()
            logger.info(f"Reverted migration: {migration_id}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error reverting migration {migration_id}: {e}")
            return False
    
    def apply_pending_migrations(self):
        """Apply all pending migrations"""
        pending = self.get_pending_migrations()
        
        if not pending:
            logger.info("No pending migrations to apply")
            return True
        
        success = True
        for migration_id in pending:
            if not self.apply_migration(migration_id):
                success = False
                break
        
        return success
    
    def list_migrations(self):
        """List all migrations and their status"""
        applied = self.get_applied_migrations()
        available = self.get_available_migrations()
        all_migrations = sorted(set(applied + available))
        
        logger.info("\nMigration Status:")
        for migration_id in all_migrations:
            status = "APPLIED" if migration_id in applied else "PENDING"
            logger.info(f"{migration_id}: {status}")
        
        return all_migrations


def main():
    """Main entry point for the migration script"""
    parser = argparse.ArgumentParser(description='Database migration manager')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Create migration
    create_parser = subparsers.add_parser('create', help='Create a new migration')
    create_parser.add_argument('description', help='Description of the migration')
    
    # Apply migrations
    apply_parser = subparsers.add_parser('apply', help='Apply pending migrations')
    apply_parser.add_argument('--migration', help='Specific migration to apply')
    
    # Revert migration
    revert_parser = subparsers.add_parser('revert', help='Revert a migration')
    revert_parser.add_argument('migration', help='Migration ID to revert')
    
    # List migrations
    list_parser = subparsers.add_parser('list', help='List all migrations and their status')
    
    args = parser.parse_args()
    
    manager = MigrationManager()
    
    if args.command == 'create':
        manager.create_migration(args.description)
    
    elif args.command == 'apply':
        if args.migration:
            manager.apply_migration(args.migration)
        else:
            manager.apply_pending_migrations()
    
    elif args.command == 'revert':
        manager.revert_migration(args.migration)
    
    elif args.command == 'list':
        manager.list_migrations()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()