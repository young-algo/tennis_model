#!/usr/bin/env python
"""
Feature Engineering Script

This script generates features from the tennis match data for model training.
It creates player, matchup, and tournament features from the raw data.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('feature_engineering')

# Path constants
DATA_DIR = Path('data')
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FEATURES_DIR = PROCESSED_DATA_DIR / 'features'
DB_PATH = PROCESSED_DATA_DIR / 'tennis.db'


class FeatureBuilder:
    """Class to build features from tennis match data"""

    def __init__(self, db_path=DB_PATH, rank_cutoff=None, years_limit=None, incremental=False, since_date=None):
        """Initialize the feature builder

        Parameters
        ----------
        db_path : str or Path
            Path to the SQLite database file.
        rank_cutoff : int or None
            Optional ranking cutoff. If provided, player features will only be
            generated for players whose most recent ranking is better than or
            equal to this value.
        years_limit : int or None
            Optional limit for matchup features to only include matches from the last
            specified number of years. If None, all matches are included.
        incremental : bool
            Whether to perform an incremental update of features or rebuild from scratch.
        since_date : str or None
            If provided with incremental=True, only process matches since this date.
            Format: 'YYYY-MM-DD'
        """
        self.db_path = db_path
        self.rank_cutoff = rank_cutoff
        self.years_limit = years_limit
        self.incremental = incremental
        self.since_date = pd.Timestamp(since_date) if since_date else None
        self.conn = None
        self.matches_df = None
        self.players_df = None
        self.rankings_df = None
        self.tournaments_df = None
        
        # Existing feature data (for incremental updates)
        self.existing_player_features = None
        self.existing_matchup_features = None
        self.existing_tournament_features = None
        
        # Ensure output directory exists
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_existing_features(self):
        """Load existing feature data for incremental updates"""
        logger.info("Loading existing feature data for incremental update")
        
        # Player features
        player_features_path = FEATURES_DIR / 'player_features.parquet'
        if player_features_path.exists():
            self.existing_player_features = pd.read_parquet(player_features_path)
            logger.info(f"Loaded {len(self.existing_player_features)} existing player features")
        else:
            self.existing_player_features = pd.DataFrame()
            logger.warning("No existing player features found, will create new")
        
        # Matchup features
        matchup_features_path = FEATURES_DIR / 'matchup_features.parquet'
        if matchup_features_path.exists():
            self.existing_matchup_features = pd.read_parquet(matchup_features_path)
            logger.info(f"Loaded {len(self.existing_matchup_features)} existing matchup features")
        else:
            self.existing_matchup_features = pd.DataFrame()
            logger.warning("No existing matchup features found, will create new")
        
        # Tournament features
        tournament_features_path = FEATURES_DIR / 'tournament_features.parquet'
        if tournament_features_path.exists():
            self.existing_tournament_features = pd.read_parquet(tournament_features_path)
            logger.info(f"Loaded {len(self.existing_tournament_features)} existing tournament features")
        else:
            self.existing_tournament_features = pd.DataFrame()
            logger.warning("No existing tournament features found, will create new")
    
    def load_data(self):
        """Load data from SQLite database"""
        logger.info("Loading data from database...")
        self.conn = sqlite3.connect(self.db_path)
        
        # SQL query for matches - filter by date if incremental
        if self.incremental and self.since_date:
            match_query = f"SELECT * FROM matches WHERE match_date >= '{self.since_date.strftime('%Y-%m-%d')}'"
            logger.info(f"Incremental update: Loading matches since {self.since_date.strftime('%Y-%m-%d')}")
        else:
            match_query = "SELECT * FROM matches"
        
        # Load matches
        self.matches_df = pd.read_sql_query(
            match_query, 
            self.conn, 
            parse_dates=['match_date']
        )
        logger.info(f"Loaded {len(self.matches_df)} matches")
        
        # Load players
        self.players_df = pd.read_sql_query(
            "SELECT * FROM players", 
            self.conn, 
            parse_dates=['birth_date']
        )
        logger.info(f"Loaded {len(self.players_df)} players")
        
        # Load rankings
        ranking_query = f"SELECT * FROM rankings"
        if self.incremental and self.since_date:
            # For rankings, get a bit more history to ensure we have context
            history_date = self.since_date - pd.DateOffset(months=3)
            ranking_query += f" WHERE ranking_date >= '{history_date.strftime('%Y-%m-%d')}'"
            
        self.rankings_df = pd.read_sql_query(
            ranking_query, 
            self.conn, 
            parse_dates=['ranking_date']
        )
        logger.info(f"Loaded {len(self.rankings_df)} ranking entries")
        
        # Load tournaments
        self.tournaments_df = pd.read_sql_query(
            "SELECT * FROM tournaments", 
            self.conn
        )
        logger.info(f"Loaded {len(self.tournaments_df)} tournaments")
        
        # If incremental update, load existing feature data
        if self.incremental:
            self._load_existing_features()
    
    def build_player_features(self):
        """Build player-level features"""
        logger.info("Building player features...")
        
        # Initialize features DataFrame
        player_features = []
        
        # Get unique players from match data
        unique_players = set(self.matches_df['winner_id']).union(set(self.matches_df['loser_id']))
        
        # For incremental updates, we only need to process players in recent matches
        # plus any players whose existing features would be affected
        if self.incremental and not self.existing_player_features.empty:
            logger.info(f"Incremental update: Starting with {len(unique_players)} players from recent matches")
            
            # Also add players from existing features whose stats need updating
            # (based on new matches in the dataset)
            known_player_ids = set(self.existing_player_features['player_id'])
            
            # Keep track of players we'll update
            players_to_update = unique_players.copy()
            # Players to keep unchanged
            players_to_keep = known_player_ids - unique_players
            
            logger.info(f"Incremental update: Updating {len(players_to_update)} players, keeping {len(players_to_keep)} unchanged")
        else:
            # When not doing incremental updates, players_to_keep is empty
            players_to_keep = set()

        # Apply ranking cutoff if provided
        if self.rank_cutoff is not None:
            latest_rankings = (
                self.rankings_df.sort_values('ranking_date')
                .dropna(subset=['ranking'])
                .groupby('player_id')
                .tail(1)
            )
            eligible_players = set(
                latest_rankings[latest_rankings['ranking'] <= self.rank_cutoff]['player_id']
            )
            unique_players = unique_players.intersection(eligible_players)
            
            # For incremental updates, also filter players_to_keep by ranking
            if self.incremental and not self.existing_player_features.empty:
                players_to_keep = players_to_keep.intersection(eligible_players)
                
            logger.info(
                f"Applying rank cutoff {self.rank_cutoff}: building features for {len(unique_players)} players"
            )
        else:
            logger.info(f"Building features for {len(unique_players)} players")
        
        for player_id in unique_players:
            # Basic player information
            player_info = self.players_df[self.players_df['player_id'] == player_id]
            if player_info.empty:
                continue
            
            player_data = {
                'player_id': player_id,
                'name': f"{player_info['first_name'].iloc[0]} {player_info['last_name'].iloc[0]}",
                'hand': player_info['hand'].iloc[0],
                'country': player_info['country_code'].iloc[0],
                'tour': player_info['tour'].iloc[0]
            }
            
            # Matches as winner
            won_matches = self.matches_df[self.matches_df['winner_id'] == player_id]
            # Matches as loser
            lost_matches = self.matches_df[self.matches_df['loser_id'] == player_id]
            
            # Total matches played
            total_matches = len(won_matches) + len(lost_matches)
            if total_matches == 0:
                continue  # Skip players with no matches
            
            player_data['total_matches'] = total_matches
            player_data['wins'] = len(won_matches)
            player_data['losses'] = len(lost_matches)
            player_data['win_rate'] = len(won_matches) / total_matches if total_matches > 0 else 0
            
            # Surface performance
            for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
                surface_wins = len(won_matches[won_matches['surface'] == surface])
                surface_losses = len(lost_matches[lost_matches['surface'] == surface])
                surface_matches = surface_wins + surface_losses
                
                player_data[f'{surface.lower()}_matches'] = surface_matches
                player_data[f'{surface.lower()}_wins'] = surface_wins
                player_data[f'{surface.lower()}_losses'] = surface_losses
                player_data[f'{surface.lower()}_win_rate'] = surface_wins / surface_matches if surface_matches > 0 else 0
            
            # Tournament level performance
            for level in ['G', 'M', 'A', 'D', 'F', 'C']:
                level_wins = len(won_matches[won_matches['tournament_level'] == level])
                level_losses = len(lost_matches[lost_matches['tournament_level'] == level])
                level_matches = level_wins + level_losses
                
                player_data[f'level_{level}_matches'] = level_matches
                player_data[f'level_{level}_wins'] = level_wins
                player_data[f'level_{level}_losses'] = level_losses
                player_data[f'level_{level}_win_rate'] = level_wins / level_matches if level_matches > 0 else 0
            
            # Recent form (last 20 matches)
            all_matches = pd.concat([
                won_matches[['match_date', 'surface']].assign(result=1),  # 1 for win
                lost_matches[['match_date', 'surface']].assign(result=0)   # 0 for loss
            ])
            all_matches = all_matches.sort_values('match_date', ascending=False)
            
            player_data['last_5_matches_win_rate'] = all_matches['result'].head(5).mean() if len(all_matches) >= 5 else np.nan
            player_data['last_10_matches_win_rate'] = all_matches['result'].head(10).mean() if len(all_matches) >= 10 else np.nan
            player_data['last_20_matches_win_rate'] = all_matches['result'].head(20).mean() if len(all_matches) >= 20 else np.nan
            
            # Service stats
            w_ace = won_matches['w_ace'].dropna()
            w_df = won_matches['w_df'].dropna()
            l_ace = lost_matches['l_ace'].dropna()
            l_df = lost_matches['l_df'].dropna()
            
            player_data['avg_ace'] = pd.concat([w_ace, l_ace]).mean() if not (w_ace.empty and l_ace.empty) else np.nan
            player_data['avg_df'] = pd.concat([w_df, l_df]).mean() if not (w_df.empty and l_df.empty) else np.nan
            
            # First serve stats
            w_1stIn = won_matches['w_1stIn'].dropna()
            w_svpt = won_matches['w_svpt'].dropna()
            l_1stIn = lost_matches['l_1stIn'].dropna()
            l_svpt = lost_matches['l_svpt'].dropna()
            
            # Calculate percentages where we have both stats
            w_mask = (~w_1stIn.isna()) & (~w_svpt.isna()) & (w_svpt > 0)
            l_mask = (~l_1stIn.isna()) & (~l_svpt.isna()) & (l_svpt > 0)
            
            w_1st_pct = (w_1stIn[w_mask] / w_svpt[w_mask]).mean() if w_mask.any() else np.nan
            l_1st_pct = (l_1stIn[l_mask] / l_svpt[l_mask]).mean() if l_mask.any() else np.nan
            
            # Combine winner and loser stats
            first_serve_pcts = []
            if not np.isnan(w_1st_pct):
                first_serve_pcts.append(w_1st_pct)
            if not np.isnan(l_1st_pct):
                first_serve_pcts.append(l_1st_pct)
            
            player_data['first_serve_pct'] = np.mean(first_serve_pcts) if first_serve_pcts else np.nan
            
            # Ranking metrics
            player_rankings = self.rankings_df[(self.rankings_df['player_id'] == player_id) & 
                                             (self.rankings_df['tour'] == player_data['tour'])]
            player_rankings = player_rankings.sort_values('ranking_date', ascending=False)
            
            player_data['current_ranking'] = player_rankings['ranking'].iloc[0] if not player_rankings.empty else np.nan
            player_data['current_points'] = player_rankings['ranking_points'].iloc[0] if not player_rankings.empty else np.nan
            
            player_features.append(player_data)
        
        # Convert to DataFrame
        player_features_df = pd.DataFrame(player_features)
        
        # For incremental updates, merge with existing features for players that aren't being updated
        if self.incremental and not self.existing_player_features.empty and players_to_keep:
            logger.info("Merging new player features with existing features")
            
            # Filter existing features to only include players we want to keep
            keep_features = self.existing_player_features[
                self.existing_player_features['player_id'].isin(players_to_keep)
            ]
            
            # Combine the new and existing features
            if not keep_features.empty:
                player_features_df = pd.concat([player_features_df, keep_features])
                logger.info(f"Added {len(keep_features)} players from existing features")
        
        # Save to parquet
        output_file = FEATURES_DIR / 'player_features.parquet'
        player_features_df.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(player_features_df)} player features to {output_file}")
        
        return player_features_df
    
    @staticmethod
    def _process_matchup(player_pair, tour_matches, players_df):
        """Process a single matchup (helper function for parallelization)"""
        p1, p2 = player_pair
        
        # Get matches between these players
        h2h_matches = tour_matches[
            ((tour_matches['winner_id'] == p1) & (tour_matches['loser_id'] == p2)) |
            ((tour_matches['winner_id'] == p2) & (tour_matches['loser_id'] == p1))
        ].copy()  # Create a copy to avoid any shared memory issues
        
        total_matches = len(h2h_matches)
        if total_matches < 2:  # Only include matchups with at least 2 matches
            return None
        
        # Count wins for each player
        p1_wins = len(h2h_matches[h2h_matches['winner_id'] == p1])
        p2_wins = len(h2h_matches[h2h_matches['winner_id'] == p2])
        
        matchup_data = {
            'player1_id': p1,
            'player2_id': p2,
            'total_matches': total_matches,
            'player1_wins': p1_wins,
            'player2_wins': p2_wins,
            'player1_win_rate': p1_wins / total_matches,
            'player2_win_rate': p2_wins / total_matches,
            'tour': tour_matches['tour'].iloc[0]
        }
        
        # Surface-specific H2H
        for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
            surface_matches = h2h_matches[h2h_matches['surface'] == surface]
            surface_total = len(surface_matches)
            
            if surface_total > 0:
                surface_p1_wins = len(surface_matches[surface_matches['winner_id'] == p1])
                surface_p2_wins = len(surface_matches[surface_matches['winner_id'] == p2])
                
                matchup_data[f'{surface.lower()}_matches'] = surface_total
                matchup_data[f'{surface.lower()}_player1_wins'] = surface_p1_wins
                matchup_data[f'{surface.lower()}_player2_wins'] = surface_p2_wins
                matchup_data[f'{surface.lower()}_player1_win_rate'] = surface_p1_wins / surface_total
                matchup_data[f'{surface.lower()}_player2_win_rate'] = surface_p2_wins / surface_total
        
        # Recent matchup results (last 3 matches)
        recent_matches = h2h_matches.sort_values('match_date', ascending=False).head(3)
        if len(recent_matches) > 0:
            recent_p1_wins = len(recent_matches[recent_matches['winner_id'] == p1])
            recent_p2_wins = len(recent_matches[recent_matches['winner_id'] == p2])
            
            matchup_data['recent_matches'] = len(recent_matches)
            matchup_data['recent_player1_win_rate'] = recent_p1_wins / len(recent_matches)
            matchup_data['recent_player2_win_rate'] = recent_p2_wins / len(recent_matches)
        
        # Physical matchup
        p1_info = players_df[players_df['player_id'] == p1]
        p2_info = players_df[players_df['player_id'] == p2]
        
        if not p1_info.empty and not p2_info.empty:
            # Height difference
            p1_height = p1_info['height'].iloc[0]
            p2_height = p2_info['height'].iloc[0]
            
            if not pd.isna(p1_height) and not pd.isna(p2_height):
                matchup_data['height_diff'] = p1_height - p2_height
            
            # Dominant hand matchup
            p1_hand = p1_info['hand'].iloc[0]
            p2_hand = p2_info['hand'].iloc[0]
            
            if not pd.isna(p1_hand) and not pd.isna(p2_hand):
                matchup_data['dominant_hand_matchup'] = f"{p1_hand}v{p2_hand}"
        
        return matchup_data

    def build_matchup_features(self):
        """Build matchup-level features between pairs of players"""
        logger.info("Building matchup features...")
        start_time = time.time()
        checkpoint_interval = 1000  # Save progress every 1000 pairs
        
        # Get unique combinations of players who've faced each other
        matchups = []
        checkpoint_path = FEATURES_DIR / 'matchup_features_checkpoint.parquet'
        
        # Resume from checkpoint if it exists
        if checkpoint_path.exists():
            logger.info(f"Found checkpoint file at {checkpoint_path}. Resuming from checkpoint...")
            checkpoint_df = pd.read_parquet(checkpoint_path)
            matchups = checkpoint_df.to_dict('records')
            # Create a set of already processed pairs to avoid duplicates
            processed_pairs = {(row['player1_id'], row['player2_id'], row['tour']) for row in matchups}
            logger.info(f"Loaded {len(matchups)} matchups from checkpoint")
        else:
            processed_pairs = set()
            
        # For incremental updates, track matchups to be updated
        matchups_to_update = set()
        if self.incremental and not self.existing_matchup_features.empty:
            logger.info("Incremental update: Determining which matchups need updating")
            
            # Identify player pairs in new matches
            for tour in ['ATP', 'WTA']:
                tour_matches = self.matches_df[self.matches_df['tour'] == tour]
                
                # Get player pairs from new matches
                for _, match in tour_matches.iterrows():
                    w_id, l_id = match['winner_id'], match['loser_id']
                    p1, p2 = (w_id, l_id) if w_id < l_id else (l_id, w_id)
                    matchups_to_update.add((p1, p2, tour))
            
            logger.info(f"Incremental update: {len(matchups_to_update)} matchups to update")
        
        # Determine number of workers (leave 1 core free for system)
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using {num_workers} workers for parallel processing")
        
        for tour in ['ATP', 'WTA']:
            tour_matches = self.matches_df[self.matches_df['tour'] == tour]
            
            # Filter matches by date if years_limit is specified
            if self.years_limit is not None:
                current_date = pd.Timestamp.now()
                cutoff_date = current_date - pd.DateOffset(years=self.years_limit)
                tour_matches = tour_matches[tour_matches['match_date'] >= cutoff_date]
                logger.info(f"Filtered to {len(tour_matches)} {tour} matches from the last {self.years_limit} years")
            
            # Create a unique set of player pairs who have played against each other
            player_pairs = set()
            
            # More efficient way to extract player pairs
            winners = tour_matches[['winner_id', 'loser_id']].values
            for w_id, l_id in winners:
                p1, p2 = (w_id, l_id) if w_id < l_id else (l_id, w_id)
                player_pairs.add((p1, p2))
            
            # Filter out already processed pairs
            player_pairs = [pair for pair in player_pairs if (pair[0], pair[1], tour) not in processed_pairs]
            
            logger.info(f"Building features for {len(player_pairs)} {tour} matchups")
            
            # Use process pool to parallelize computation
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Create a partial function with fixed arguments
                process_func = partial(self._process_matchup, 
                                      tour_matches=tour_matches, 
                                      players_df=self.players_df)
                
                # Process in batches with progress monitoring
                batch_size = 5000  # Adjust based on available memory
                for i in range(0, len(player_pairs), batch_size):
                    batch = player_pairs[i:i+batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{len(player_pairs)//batch_size + 1} ({len(batch)} pairs)")
                    
                    # Execute in parallel with progress bar
                    results = list(tqdm(
                        executor.map(process_func, batch),
                        total=len(batch),
                        desc=f"{tour} Matchups"
                    ))
                    
                    # Filter out None results and append to matchups
                    batch_results = [r for r in results if r is not None]
                    matchups.extend(batch_results)
                    
                    # Save checkpoint periodically
                    if len(matchups) % checkpoint_interval == 0:
                        checkpoint_df = pd.DataFrame(matchups)
                        checkpoint_df.to_parquet(checkpoint_path, index=False)
                        elapsed = time.time() - start_time
                        logger.info(f"Checkpoint saved: {len(matchups)} matchups processed in {elapsed:.2f}s")
        
        # Convert to DataFrame
        matchup_features_df = pd.DataFrame(matchups)
        
        # For incremental updates, merge with existing features
        if self.incremental and not self.existing_matchup_features.empty:
            # Create identifier for each matchup
            if not matchup_features_df.empty:
                matchup_features_df['matchup_id'] = matchup_features_df.apply(
                    lambda x: f"{x['player1_id']}_{x['player2_id']}_{x['tour']}", axis=1)
            
            # Add same identifier to existing features
            self.existing_matchup_features['matchup_id'] = self.existing_matchup_features.apply(
                lambda x: f"{x['player1_id']}_{x['player2_id']}_{x['tour']}", axis=1)
            
            # Get matchups that were updated
            updated_ids = set(matchup_features_df['matchup_id']) if not matchup_features_df.empty else set()
            
            # Filter out matchups that were updated
            keep_features = self.existing_matchup_features[
                ~self.existing_matchup_features['matchup_id'].isin(updated_ids)
            ].drop(columns=['matchup_id'])
            
            logger.info(f"Keeping {len(keep_features)} unchanged matchups from existing features")
            
            # Combine the new and existing features
            if not matchup_features_df.empty:
                matchup_features_df = matchup_features_df.drop(columns=['matchup_id'])
                matchup_features_df = pd.concat([matchup_features_df, keep_features])
                logger.info(f"Combined features: {len(matchup_features_df)} total matchups")
            else:
                matchup_features_df = keep_features
                logger.info(f"Using only existing features: {len(matchup_features_df)} matchups")
        
        # Remove checkpoint file
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        # Save to parquet if we have data
        if not matchup_features_df.empty:
            output_file = FEATURES_DIR / 'matchup_features.parquet'
            matchup_features_df.to_parquet(output_file, index=False)
            elapsed = time.time() - start_time
            logger.info(f"Saved {len(matchup_features_df)} matchup features to {output_file} in {elapsed:.2f}s")
        else:
            logger.warning("No matchup features generated")
        
        return matchup_features_df
    
    def build_tournament_features(self):
        """Build tournament-level features"""
        logger.info("Building tournament features...")
        
        tournament_features = []
        
        # For incremental updates, identify which tournaments need updating
        tournaments_to_update = set()
        if self.incremental and not self.existing_tournament_features.empty:
            # Get tournaments from new matches
            tournaments_to_update = set(self.matches_df['tournament_id'].dropna())
            logger.info(f"Incremental update: {len(tournaments_to_update)} tournaments to update")
        
        for tour in ['ATP', 'WTA']:
            # Get unique tournaments
            tournaments = self.tournaments_df[self.tournaments_df['tour'] == tour]
            
            for _, tournament in tournaments.iterrows():
                tournament_id = tournament['tournament_id']
                
                # In incremental mode, skip tournaments that aren't in new matches
                if self.incremental and tournament_id not in tournaments_to_update:
                    continue
                
                # Get all matches for this tournament
                tournament_matches = self.matches_df[self.matches_df['tournament_id'] == tournament_id]
                
                if len(tournament_matches) == 0:
                    continue  # Skip tournaments with no matches
                
                # Basic statistics
                tournament_data = {
                    'tournament_id': tournament_id,
                    'tournament_name': tournament['tournament_name'],
                    'surface': tournament['surface'],
                    'tournament_level': tournament['tournament_level'],
                    'draw_size': tournament['draw_size'],
                    'tour': tour,
                    'total_matches': len(tournament_matches)
                }
                
                # Years active
                years = tournament_matches['match_date'].dt.year.unique()
                tournament_data['years_active'] = len(years)
                tournament_data['first_year'] = min(years)
                tournament_data['last_year'] = max(years)
                
                # Player information
                unique_players = set(tournament_matches['winner_id']).union(set(tournament_matches['loser_id']))
                tournament_data['unique_players'] = len(unique_players)
                
                # Winner/loser rankings
                winner_ranks = tournament_matches['winner_rank'].dropna()
                loser_ranks = tournament_matches['loser_rank'].dropna()
                
                tournament_data['avg_winner_rank'] = winner_ranks.mean() if not winner_ranks.empty else np.nan
                tournament_data['avg_loser_rank'] = loser_ranks.mean() if not loser_ranks.empty else np.nan
                
                # Match duration
                match_duration = tournament_matches['minutes'].dropna()
                tournament_data['avg_match_duration'] = match_duration.mean() if not match_duration.empty else np.nan
                
                tournament_features.append(tournament_data)
        
        # Convert to DataFrame
        tournament_features_df = pd.DataFrame(tournament_features)
        
        # For incremental updates, merge with existing features
        if self.incremental and not self.existing_tournament_features.empty:
            # Add identifier column
            if not tournament_features_df.empty:
                # Create unique identifier for each tournament
                tournament_features_df['tournament_key'] = tournament_features_df.apply(
                    lambda x: f"{x['tournament_id']}_{x['tour']}", axis=1)
            
            # Add same identifier to existing features
            self.existing_tournament_features['tournament_key'] = self.existing_tournament_features.apply(
                lambda x: f"{x['tournament_id']}_{x['tour']}", axis=1)
            
            # Get tournaments that were updated
            updated_ids = set(tournament_features_df['tournament_key']) if not tournament_features_df.empty else set()
            
            # Filter out tournaments that were updated
            keep_features = self.existing_tournament_features[
                ~self.existing_tournament_features['tournament_key'].isin(updated_ids)
            ].drop(columns=['tournament_key'])
            
            logger.info(f"Keeping {len(keep_features)} unchanged tournaments from existing features")
            
            # Combine the new and existing features
            if not tournament_features_df.empty:
                tournament_features_df = tournament_features_df.drop(columns=['tournament_key'])
                tournament_features_df = pd.concat([tournament_features_df, keep_features])
                logger.info(f"Combined features: {len(tournament_features_df)} total tournaments")
            else:
                tournament_features_df = keep_features
                logger.info(f"Using only existing features: {len(tournament_features_df)} tournaments")
        
        # Save to parquet
        output_file = FEATURES_DIR / 'tournament_features.parquet'
        tournament_features_df.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(tournament_features_df)} tournament features to {output_file}")

        return tournament_features_df

    def build_all_features(self):
        """Build all feature sets"""
        self.load_data()

        player_features = self.build_player_features()
        
        # Close connection before multiprocessing to avoid pickling errors
        if self.conn:
            self.conn.close()
            self.conn = None
            
        matchup_features = self.build_matchup_features()
        
        # Reopen connection for tournament features if needed
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            
        tournament_features = self.build_tournament_features()
        
        # Close connection when done
        if self.conn:
            self.conn.close()

        return {
            'player_features': player_features,
            'matchup_features': matchup_features,
            'tournament_features': tournament_features
        }


def main():
    """Main entry point for the feature building script"""
    parser = argparse.ArgumentParser(description='Build features from tennis match data')
    parser.add_argument('--db-path', type=str, default=None, help='Custom path to the SQLite database file')
    parser.add_argument('--player-only', action='store_true', help='Only build player features')
    parser.add_argument('--matchup-only', action='store_true', help='Only build matchup features')
    parser.add_argument('--tournament-only', action='store_true', help='Only build tournament features')
    parser.add_argument('--rank-cutoff', type=int, default=None,
                        help='Only generate player features for players with a ranking better than or equal to this value')
    parser.add_argument('--years-limit', type=int, default=3,
                        help='Limit matchup features to only include matches from the last N years (default: 3)')
    parser.add_argument('--incremental', action='store_true', 
                        help='Incrementally update features with new data instead of rebuilding everything')
    parser.add_argument('--since-date', type=str, default=None,
                        help='Only process matches since this date (format: YYYY-MM-DD). Used with --incremental')

    args = parser.parse_args()

    # Set up feature builder
    builder = FeatureBuilder(
        db_path=args.db_path if args.db_path else DB_PATH,
        rank_cutoff=args.rank_cutoff,
        years_limit=args.years_limit,
        incremental=args.incremental,
        since_date=args.since_date,
    )
    builder.load_data()

    # Build features based on arguments
    if args.player_only:
        builder.build_player_features()
        # Close connection when done
        if builder.conn:
            builder.conn.close()
    elif args.matchup_only:
        # Close connection before multiprocessing to avoid pickling errors
        if builder.conn:
            builder.conn.close()
            builder.conn = None
        builder.build_matchup_features()
    elif args.tournament_only:
        builder.build_tournament_features()
        # Close connection when done
        if builder.conn:
            builder.conn.close()
    else:
        # Build all features by default
        builder.build_all_features()

    logger.info("Feature building completed successfully")


if __name__ == "__main__":
    main()
