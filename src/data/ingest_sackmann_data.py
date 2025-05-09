#!/usr/bin/env python
"""
Tennis Model Data Ingestion Script

This script ingests tennis match data from Jeff Sackmann's GitHub repositories
and stores it in an SQLite database with a unified schema.
"""

import os
import csv
import sqlite3
import glob
import logging
import datetime
import hashlib
import time
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/ingest_log.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('tennis_data_ingestor')

# Path constants
DATA_DIR = Path('data')
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Ensure processed directory exists
PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)

# Database file path
DB_PATH = PROCESSED_DATA_DIR / 'tennis.db'

# Data source paths
ATP_DIR = RAW_DATA_DIR / 'tennis_atp'
WTA_DIR = RAW_DATA_DIR / 'tennis_wta'
MCP_DIR = RAW_DATA_DIR / 'tennis_mcp'
PBP_DIR = RAW_DATA_DIR / 'tennis_pbp'


class TennisDataIngestor:
    """Class to handle ingestion of tennis data into SQLite database."""
    
    def __init__(self, db_path=DB_PATH, reset_db=False):
        """Initialize the ingestor with the path to the SQLite database."""
        self.db_path = db_path
        self.conn = None
        self.reset_db = reset_db
        self.file_checksums = {}
        self.setup_database()
        
    def setup_database(self):
        """Set up the SQLite database with the required tables."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        if self.reset_db:
            # Drop tables if they exist
            cursor.execute("DROP TABLE IF EXISTS matches")
            cursor.execute("DROP TABLE IF EXISTS players")
            cursor.execute("DROP TABLE IF EXISTS tournaments")
            cursor.execute("DROP TABLE IF EXISTS rankings")
            cursor.execute("DROP TABLE IF EXISTS file_metadata")
            self.conn.commit()
        
        # Create tables with appropriate schema
        
        # File metadata table to track processed files and their checksums
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_metadata (
            file_path TEXT PRIMARY KEY,
            last_modified TEXT,
            checksum TEXT,
            last_processed TEXT
        )
        ''')
        
        # Players table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            player_id TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            hand TEXT,
            birth_date TEXT,
            country_code TEXT,
            height INTEGER,
            tour TEXT
        )
        ''')
        
        # Tournaments table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tournaments (
            tournament_id TEXT PRIMARY KEY,
            tournament_name TEXT,
            surface TEXT,
            draw_size INTEGER,
            tournament_level TEXT,
            court_type TEXT,
            tour TEXT
        )
        ''')
        
        # Matches table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            match_id TEXT PRIMARY KEY,
            tournament_id TEXT,
            tournament_name TEXT,
            surface TEXT,
            tour TEXT,
            match_date TEXT,
            match_round TEXT,
            best_of INTEGER,
            winner_id TEXT,
            winner_name TEXT,
            winner_hand TEXT,
            winner_ht INTEGER,
            winner_ioc TEXT,
            winner_age REAL,
            winner_rank INTEGER,
            winner_rank_points INTEGER,
            loser_id TEXT,
            loser_name TEXT,
            loser_hand TEXT,
            loser_ht INTEGER,
            loser_ioc TEXT,
            loser_age REAL,
            loser_rank INTEGER,
            loser_rank_points INTEGER,
            score TEXT,
            w_sets_won INTEGER,
            l_sets_won INTEGER,
            w_games_won INTEGER,
            l_games_won INTEGER,
            w_ace INTEGER,
            w_df INTEGER,
            w_svpt INTEGER,
            w_1stIn INTEGER,
            w_1stWon INTEGER,
            w_2ndWon INTEGER,
            w_SvGms INTEGER,
            w_bpSaved INTEGER,
            w_bpFaced INTEGER,
            l_ace INTEGER,
            l_df INTEGER,
            l_svpt INTEGER,
            l_1stIn INTEGER,
            l_1stWon INTEGER,
            l_2ndWon INTEGER,
            l_SvGms INTEGER,
            l_bpSaved INTEGER,
            l_bpFaced INTEGER,
            minutes INTEGER,
            tournament_level TEXT,
            FOREIGN KEY (winner_id) REFERENCES players (player_id),
            FOREIGN KEY (loser_id) REFERENCES players (player_id),
            FOREIGN KEY (tournament_id) REFERENCES tournaments (tournament_id)
        )
        ''')
        
        # Rankings table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS rankings (
            ranking_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ranking_date TEXT,
            player_id TEXT,
            ranking INTEGER,
            ranking_points INTEGER,
            tour TEXT,
            FOREIGN KEY (player_id) REFERENCES players (player_id)
        )
        ''')
        
        # Create indexes for faster querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_winner_id ON matches (winner_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_loser_id ON matches (loser_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_tournament_id ON matches (tournament_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_match_date ON matches (match_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_surface ON matches (surface)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_tour ON matches (tour)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rankings_player_id ON rankings (player_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rankings_date ON rankings (ranking_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_players_country ON players (country_code)')
        
        self.conn.commit()
        
        # Load existing checksums
        cursor.execute("SELECT file_path, checksum FROM file_metadata")
        for file_path, checksum in cursor.fetchall():
            self.file_checksums[file_path] = checksum
    
    def calculate_file_checksum(self, file_path):
        """Calculate SHA-256 checksum for a file to detect changes."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def is_file_updated(self, file_path):
        """Check if a file has been updated since last processing."""
        file_path_str = str(file_path)
        
        # If file doesn't exist, return False
        if not os.path.exists(file_path):
            return False
            
        # Calculate current checksum
        current_checksum = self.calculate_file_checksum(file_path)
        
        # If file hasn't been processed before or checksum has changed
        if file_path_str not in self.file_checksums or current_checksum != self.file_checksums[file_path_str]:
            self.file_checksums[file_path_str] = current_checksum
            return True
        
        return False
    
    def update_file_metadata(self, file_path):
        """Update the file metadata in the database."""
        file_path_str = str(file_path)
        
        cursor = self.conn.cursor()
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        checksum = self.file_checksums[file_path_str]
        now = datetime.datetime.now().isoformat()
        
        cursor.execute('''
        INSERT OR REPLACE INTO file_metadata 
        (file_path, last_modified, checksum, last_processed) 
        VALUES (?, ?, ?, ?)
        ''', (file_path_str, last_modified, checksum, now))
        
        self.conn.commit()
    
    def parse_score(self, score):
        """Parse the score string and calculate sets and games won by each player."""
        if not score or score == '':
            return 0, 0, 0, 0
        
        # Handle retirements, walkovers, defaults
        if 'RET' in score or 'W/O' in score or 'DEF' in score or 'ABD' in score:
            # Extract the partial score before retirement
            score = score.split(' ')[0]
        
        # Initialize counters
        w_sets_won = 0
        l_sets_won = 0
        w_games_won = 0
        l_games_won = 0
        
        try:
            # Split the score into sets
            sets = score.replace('(', ' ').replace(')', ' ').strip().split(' ')
            sets = [s for s in sets if s and s[0].isdigit()]
            
            for set_score in sets:
                if not set_score or '-' not in set_score:
                    continue
                
                # Handle tiebreak sets
                if '(' in set_score:
                    set_score = set_score.split('(')[0]
                    
                # Extract games for each player
                parts = set_score.split('-')
                if len(parts) != 2:
                    continue
                    
                try:
                    w_games = int(parts[0])
                    l_games = int(parts[1])
                    
                    # Count games
                    w_games_won += w_games
                    l_games_won += l_games
                    
                    # Count sets
                    if w_games > l_games:
                        w_sets_won += 1
                    elif l_games > w_games:
                        l_sets_won += 1
                        
                except ValueError:
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing score '{score}': {e}")
            
        return w_sets_won, l_sets_won, w_games_won, l_games_won
    
    def ingest_all_data(self):
        """Run the full data ingestion process."""
        start_time = time.time()
        logger.info("Starting tennis data ingestion process...")
        
        # Check for repository updates if requested
        if hasattr(self, 'check_updates') and self.check_updates:
            self.check_for_repository_updates()
        
        # Ingest player data
        self.ingest_atp_players()
        self.ingest_wta_players()
        
        # Ingest match data
        self.ingest_atp_matches()
        self.ingest_wta_matches()
        
        # Ingest ranking data
        self.ingest_atp_rankings()
        self.ingest_wta_rankings()
        
        # Close database connection
        if self.conn:
            self.conn.close()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Data ingestion completed in {elapsed_time:.2f} seconds")


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Ingest tennis match data into SQLite database')
    parser.add_argument('--reset-db', action='store_true', help='Reset the database before ingestion')
    parser.add_argument('--db-path', type=str, default=None, help='Custom path to the SQLite database file')
    parser.add_argument('--check-updates', action='store_true', help='Check for updates in the source repositories')
    parser.add_argument('--atp-only', action='store_true', help='Only process ATP data')
    parser.add_argument('--wta-only', action='store_true', help='Only process WTA data')
    
    args = parser.parse_args()
    
    # Create and configure the ingestor
    ingestor = TennisDataIngestor(
        db_path=args.db_path if args.db_path else DB_PATH,
        reset_db=args.reset_db
    )
    
    # Set additional options
    ingestor.check_updates = args.check_updates
    
    # Process data based on options
    if args.atp_only:
        ingestor.ingest_atp_players()
        ingestor.ingest_atp_matches()
        ingestor.ingest_atp_rankings()
    elif args.wta_only:
        ingestor.ingest_wta_players()
        ingestor.ingest_wta_matches()
        ingestor.ingest_wta_rankings()
    else:
        ingestor.ingest_all_data()
    
    logger.info("Data ingestion completed successfully")


if __name__ == "__main__":
    main()