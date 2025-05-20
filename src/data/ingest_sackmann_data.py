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
    
    def __init__(self, db_path=DB_PATH, reset_db=False, incremental=False, since_date=None):
        """Initialize the ingestor with the path to the SQLite database."""
        self.db_path = db_path
        self.conn = None
        self.reset_db = reset_db
        self.incremental = incremental
        
        # Parse since_date string to datetime object if provided
        if since_date:
            try:
                self.since_date = datetime.datetime.strptime(since_date, '%Y-%m-%d').date()
                logger.info(f"Only processing matches since {self.since_date}")
            except ValueError:
                logger.warning(f"Invalid date format: {since_date}. Expected YYYY-MM-DD. Ignoring date filter.")
                self.since_date = None
        else:
            self.since_date = None
            
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
    
    def should_process_file(self, file_path):
        """Determine if a file should be processed based on incremental setting and checksum."""
        if not os.path.exists(file_path):
            return False
            
        # Always process if not in incremental mode
        if not self.incremental:
            return True
            
        # In incremental mode, only process files that have changed
        return self.is_file_updated(file_path)
        
    def should_process_match(self, match_date_str):
        """Determine if a match should be processed based on date filter."""
        if not self.incremental or not self.since_date:
            return True
            
        try:
            match_date = datetime.datetime.strptime(match_date_str, '%Y%m%d').date()
            return match_date >= self.since_date
        except (ValueError, TypeError):
            # If date parsing fails, include the match to be safe
            return True
    
    def ingest_atp_players(self):
        """Ingest ATP player data."""
        logger.info("Ingesting ATP player data...")
        
        cursor = self.conn.cursor()
        player_files = glob.glob(str(ATP_DIR / "atp_players_*.csv"))
        
        processed_count = 0
        for file_path in player_files:
            if not self.should_process_file(file_path):
                continue
                
            # Process player file
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    player_id = row[0]
                    
                    # Prepare player data
                    cursor.execute('''
                    INSERT OR REPLACE INTO players 
                    (player_id, first_name, last_name, hand, birth_date, country_code, height, tour)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (player_id, row[1], row[2], row[3], row[4], row[5], 
                          int(row[6]) if row[6] and row[6].isdigit() else None, 'ATP'))
                    
                    processed_count += 1
            
            # Update file metadata
            self.update_file_metadata(file_path)
            
        self.conn.commit()
        logger.info(f"Processed {processed_count} ATP players")
        
    def ingest_wta_players(self):
        """Ingest WTA player data."""
        logger.info("Ingesting WTA player data...")
        
        cursor = self.conn.cursor()
        player_files = glob.glob(str(WTA_DIR / "wta_players_*.csv"))
        
        processed_count = 0
        for file_path in player_files:
            if not self.should_process_file(file_path):
                continue
                
            # Process player file
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    player_id = row[0]
                    
                    # Prepare player data
                    cursor.execute('''
                    INSERT OR REPLACE INTO players 
                    (player_id, first_name, last_name, hand, birth_date, country_code, height, tour)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (player_id, row[1], row[2], row[3], row[4], row[5], 
                          int(row[6]) if row[6] and row[6].isdigit() else None, 'WTA'))
                    
                    processed_count += 1
            
            # Update file metadata
            self.update_file_metadata(file_path)
            
        self.conn.commit()
        logger.info(f"Processed {processed_count} WTA players")
        
    def ingest_atp_matches(self):
        """Ingest ATP match data."""
        logger.info("Ingesting ATP match data...")
        
        cursor = self.conn.cursor()
        match_files = glob.glob(str(ATP_DIR / "atp_matches_????.csv"))
        
        processed_count = 0
        for file_path in match_files:
            if not self.should_process_file(file_path):
                continue
                
            # Process match file
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                for row in reader:
                    try:
                        # Extract match date for filtering
                        match_date_str = row[5]  # Assuming date is in column 5
                        
                        # Skip matches that don't meet date criteria in incremental mode
                        if not self.should_process_match(match_date_str):
                            continue
                            
                        # Generate a unique match ID
                        match_id = f"ATP_{row[0]}_{row[5]}_{row[7]}_{row[10]}"
                        
                        # Parse score
                        w_sets, l_sets, w_games, l_games = self.parse_score(row[23])
                        
                        # Insert tournament if needed
                        tournament_id = f"ATP_{row[0]}_{row[1]}"
                        cursor.execute('''
                        INSERT OR IGNORE INTO tournaments
                        (tournament_id, tournament_name, surface, draw_size, tournament_level, court_type, tour)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (tournament_id, row[1], row[2], 
                              int(row[4]) if row[4] and row[4].isdigit() else None, 
                              row[6], None, 'ATP'))
                        
                        # Insert match data
                        cursor.execute('''
                        INSERT OR REPLACE INTO matches
                        (match_id, tournament_id, tournament_name, surface, tour, match_date,
                         match_round, best_of, winner_id, winner_name, winner_hand,
                         winner_ht, winner_ioc, winner_age, winner_rank, winner_rank_points,
                         loser_id, loser_name, loser_hand, loser_ht, loser_ioc,
                         loser_age, loser_rank, loser_rank_points, score,
                         w_sets_won, l_sets_won, w_games_won, l_games_won,
                         w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon,
                         w_SvGms, w_bpSaved, w_bpFaced, l_ace, l_df,
                         l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms,
                         l_bpSaved, l_bpFaced, minutes, tournament_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            match_id, tournament_id, row[1], row[2], 'ATP',
                            f"{match_date_str[:4]}-{match_date_str[4:6]}-{match_date_str[6:8]}" if len(match_date_str) == 8 else None,
                            row[3], row[29], row[7], row[10], row[11], 
                            int(row[12]) if row[12] and row[12].isdigit() else None, row[13],
                            float(row[14]) if row[14] and row[14].replace('.', '', 1).isdigit() else None,
                            int(row[15]) if row[15] and row[15].isdigit() else None,
                            int(row[16]) if row[16] and row[16].isdigit() else None,
                            row[17], row[20], row[21],
                            int(row[22]) if row[22] and row[22].isdigit() else None, row[23],
                            float(row[24]) if row[24] and row[24].replace('.', '', 1).isdigit() else None,
                            int(row[25]) if row[25] and row[25].isdigit() else None,
                            int(row[26]) if row[26] and row[26].isdigit() else None,
                            row[27], w_sets, l_sets, w_games, l_games,
                            int(row[30]) if row[30] and row[30].isdigit() else None,
                            int(row[31]) if row[31] and row[31].isdigit() else None,
                            int(row[32]) if row[32] and row[32].isdigit() else None,
                            int(row[33]) if row[33] and row[33].isdigit() else None,
                            int(row[34]) if row[34] and row[34].isdigit() else None,
                            int(row[35]) if row[35] and row[35].isdigit() else None,
                            int(row[36]) if row[36] and row[36].isdigit() else None,
                            int(row[37]) if row[37] and row[37].isdigit() else None,
                            int(row[38]) if row[38] and row[38].isdigit() else None,
                            int(row[39]) if row[39] and row[39].isdigit() else None,
                            int(row[40]) if row[40] and row[40].isdigit() else None,
                            int(row[41]) if row[41] and row[41].isdigit() else None,
                            int(row[42]) if row[42] and row[42].isdigit() else None,
                            int(row[43]) if row[43] and row[43].isdigit() else None,
                            int(row[44]) if row[44] and row[44].isdigit() else None,
                            int(row[45]) if row[45] and row[45].isdigit() else None,
                            int(row[46]) if row[46] and row[46].isdigit() else None,
                            int(row[47]) if row[47] and row[47].isdigit() else None,
                            int(row[48]) if row[48] and row[48].isdigit() else None,
                            row[6]
                        ))
                        
                        processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing ATP match in {file_path}: {e}")
                        continue
            
            # Update file metadata
            self.update_file_metadata(file_path)
            
        self.conn.commit()
        logger.info(f"Processed {processed_count} ATP matches")
        
    def ingest_wta_matches(self):
        """Ingest WTA match data."""
        logger.info("Ingesting WTA match data...")
        
        cursor = self.conn.cursor()
        match_files = glob.glob(str(WTA_DIR / "wta_matches_????.csv"))
        
        processed_count = 0
        for file_path in match_files:
            if not self.should_process_file(file_path):
                continue
                
            # Process match file
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                for row in reader:
                    try:
                        # Extract match date for filtering
                        match_date_str = row[5]  # Assuming date is in column 5
                        
                        # Skip matches that don't meet date criteria in incremental mode
                        if not self.should_process_match(match_date_str):
                            continue
                            
                        # Generate a unique match ID
                        match_id = f"WTA_{row[0]}_{row[5]}_{row[7]}_{row[10]}"
                        
                        # Parse score
                        w_sets, l_sets, w_games, l_games = self.parse_score(row[23])
                        
                        # Insert tournament if needed
                        tournament_id = f"WTA_{row[0]}_{row[1]}"
                        cursor.execute('''
                        INSERT OR IGNORE INTO tournaments
                        (tournament_id, tournament_name, surface, draw_size, tournament_level, court_type, tour)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (tournament_id, row[1], row[2], 
                              int(row[4]) if row[4] and row[4].isdigit() else None, 
                              row[6], None, 'WTA'))
                        
                        # Insert match data
                        cursor.execute('''
                        INSERT OR REPLACE INTO matches
                        (match_id, tournament_id, tournament_name, surface, tour, match_date,
                         match_round, best_of, winner_id, winner_name, winner_hand,
                         winner_ht, winner_ioc, winner_age, winner_rank, winner_rank_points,
                         loser_id, loser_name, loser_hand, loser_ht, loser_ioc,
                         loser_age, loser_rank, loser_rank_points, score,
                         w_sets_won, l_sets_won, w_games_won, l_games_won,
                         w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon,
                         w_SvGms, w_bpSaved, w_bpFaced, l_ace, l_df,
                         l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms,
                         l_bpSaved, l_bpFaced, minutes, tournament_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            match_id, tournament_id, row[1], row[2], 'WTA',
                            f"{match_date_str[:4]}-{match_date_str[4:6]}-{match_date_str[6:8]}" if len(match_date_str) == 8 else None,
                            row[3], 3, row[7], row[10], row[11], 
                            int(row[12]) if row[12] and row[12].isdigit() else None, row[13],
                            float(row[14]) if row[14] and row[14].replace('.', '', 1).isdigit() else None,
                            int(row[15]) if row[15] and row[15].isdigit() else None,
                            int(row[16]) if row[16] and row[16].isdigit() else None,
                            row[17], row[20], row[21],
                            int(row[22]) if row[22] and row[22].isdigit() else None, row[23],
                            float(row[24]) if row[24] and row[24].replace('.', '', 1).isdigit() else None,
                            int(row[25]) if row[25] and row[25].isdigit() else None,
                            int(row[26]) if row[26] and row[26].isdigit() else None,
                            row[27], w_sets, l_sets, w_games, l_games,
                            int(row[30]) if len(row) > 30 and row[30] and row[30].isdigit() else None,
                            int(row[31]) if len(row) > 31 and row[31] and row[31].isdigit() else None,
                            int(row[32]) if len(row) > 32 and row[32] and row[32].isdigit() else None,
                            int(row[33]) if len(row) > 33 and row[33] and row[33].isdigit() else None,
                            int(row[34]) if len(row) > 34 and row[34] and row[34].isdigit() else None,
                            int(row[35]) if len(row) > 35 and row[35] and row[35].isdigit() else None,
                            int(row[36]) if len(row) > 36 and row[36] and row[36].isdigit() else None,
                            int(row[37]) if len(row) > 37 and row[37] and row[37].isdigit() else None,
                            int(row[38]) if len(row) > 38 and row[38] and row[38].isdigit() else None,
                            int(row[39]) if len(row) > 39 and row[39] and row[39].isdigit() else None,
                            int(row[40]) if len(row) > 40 and row[40] and row[40].isdigit() else None,
                            int(row[41]) if len(row) > 41 and row[41] and row[41].isdigit() else None,
                            int(row[42]) if len(row) > 42 and row[42] and row[42].isdigit() else None,
                            int(row[43]) if len(row) > 43 and row[43] and row[43].isdigit() else None,
                            int(row[44]) if len(row) > 44 and row[44] and row[44].isdigit() else None,
                            int(row[45]) if len(row) > 45 and row[45] and row[45].isdigit() else None,
                            int(row[46]) if len(row) > 46 and row[46] and row[46].isdigit() else None,
                            row[6]
                        ))
                        
                        processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing WTA match in {file_path}: {e}")
                        continue
            
            # Update file metadata
            self.update_file_metadata(file_path)
            
        self.conn.commit()
        logger.info(f"Processed {processed_count} WTA matches")
        
    def ingest_atp_rankings(self):
        """Ingest ATP ranking data."""
        logger.info("Ingesting ATP ranking data...")
        
        cursor = self.conn.cursor()
        ranking_files = glob.glob(str(ATP_DIR / "atp_rankings_*.csv"))
        
        processed_count = 0
        for file_path in ranking_files:
            if not self.should_process_file(file_path):
                continue
                
            # Process ranking file
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    try:
                        # Skip rankings that don't meet date criteria in incremental mode
                        ranking_date_str = row[0]
                        if self.incremental and self.since_date:
                            try:
                                ranking_date = datetime.datetime.strptime(ranking_date_str, '%Y%m%d').date()
                                if ranking_date < self.since_date:
                                    continue
                            except ValueError:
                                pass  # If date parsing fails, include the ranking
                        
                        # Insert ranking data
                        cursor.execute('''
                        INSERT OR REPLACE INTO rankings
                        (ranking_date, player_id, ranking, ranking_points, tour)
                        VALUES (?, ?, ?, ?, ?)
                        ''', (
                            f"{ranking_date_str[:4]}-{ranking_date_str[4:6]}-{ranking_date_str[6:8]}" if len(ranking_date_str) == 8 else None,
                            row[2],
                            int(row[1]) if row[1] and row[1].isdigit() else None,
                            int(row[3]) if row[3] and row[3].isdigit() else None,
                            'ATP'
                        ))
                        
                        processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing ATP ranking in {file_path}: {e}")
                        continue
            
            # Update file metadata
            self.update_file_metadata(file_path)
            
        self.conn.commit()
        logger.info(f"Processed {processed_count} ATP rankings")
        
    def ingest_wta_rankings(self):
        """Ingest WTA ranking data."""
        logger.info("Ingesting WTA ranking data...")
        
        cursor = self.conn.cursor()
        ranking_files = glob.glob(str(WTA_DIR / "wta_rankings_*.csv"))
        
        processed_count = 0
        for file_path in ranking_files:
            if not self.should_process_file(file_path):
                continue
                
            # Process ranking file
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    try:
                        # Skip rankings that don't meet date criteria in incremental mode
                        ranking_date_str = row[0]
                        if self.incremental and self.since_date:
                            try:
                                ranking_date = datetime.datetime.strptime(ranking_date_str, '%Y%m%d').date()
                                if ranking_date < self.since_date:
                                    continue
                            except ValueError:
                                pass  # If date parsing fails, include the ranking
                        
                        # Insert ranking data
                        cursor.execute('''
                        INSERT OR REPLACE INTO rankings
                        (ranking_date, player_id, ranking, ranking_points, tour)
                        VALUES (?, ?, ?, ?, ?)
                        ''', (
                            f"{ranking_date_str[:4]}-{ranking_date_str[4:6]}-{ranking_date_str[6:8]}" if len(ranking_date_str) == 8 else None,
                            row[2],
                            int(row[1]) if row[1] and row[1].isdigit() else None,
                            int(row[3]) if row[3] and row[3].isdigit() else None,
                            'WTA'
                        ))
                        
                        processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing WTA ranking in {file_path}: {e}")
                        continue
            
            # Update file metadata
            self.update_file_metadata(file_path)
            
        self.conn.commit()
        logger.info(f"Processed {processed_count} WTA rankings")
    
    def ingest_all_data(self):
        """Run the full data ingestion process."""
        start_time = time.time()
        
        if self.incremental:
            logger.info(f"Starting incremental tennis data ingestion process{' since ' + self.since_date.strftime('%Y-%m-%d') if self.since_date else ''}...")
        else:
            logger.info("Starting full tennis data ingestion process...")
        
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
    parser.add_argument('--incremental', action='store_true', 
                        help='Process only new or updated files (incremental update)')
    parser.add_argument('--since-date', type=str, default=None,
                        help='Only process matches since this date (format: YYYY-MM-DD). Used with --incremental')
    
    args = parser.parse_args()
    
    # Create and configure the ingestor
    ingestor = TennisDataIngestor(
        db_path=args.db_path if args.db_path else DB_PATH,
        reset_db=args.reset_db,
        incremental=args.incremental,
        since_date=args.since_date
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
