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
from pathlib import Path
import argparse

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
    
    def __init__(self, db_path=DB_PATH):
        """Initialize the feature builder"""
        self.db_path = db_path
        self.conn = None
        self.matches_df = None
        self.players_df = None
        self.rankings_df = None
        self.tournaments_df = None
        
        # Ensure output directory exists
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load data from SQLite database"""
        logger.info("Loading data from database...")
        self.conn = sqlite3.connect(self.db_path)
        
        # Load matches
        self.matches_df = pd.read_sql_query(
            "SELECT * FROM matches", 
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
        self.rankings_df = pd.read_sql_query(
            "SELECT * FROM rankings", 
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
    
    def build_player_features(self):
        """Build player-level features"""
        logger.info("Building player features...")
        
        # Initialize features DataFrame
        player_features = []
        
        # Get unique players
        unique_players = set(self.matches_df['winner_id']).union(set(self.matches_df['loser_id']))
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
        
        # Save to parquet
        output_file = FEATURES_DIR / 'player_features.parquet'
        player_features_df.to_parquet(output_file, index=False)
        logger.info(f"Saved player features to {output_file}")
        
        return player_features_df
    
    def build_matchup_features(self):
        """Build matchup-level features between pairs of players"""
        logger.info("Building matchup features...")
        
        # Get unique combinations of players who've faced each other
        matchups = []
        
        for tour in ['ATP', 'WTA']:
            tour_matches = self.matches_df[self.matches_df['tour'] == tour]
            
            # Create a unique set of player pairs who have played against each other
            player_pairs = set()
            for _, match in tour_matches.iterrows():
                p1, p2 = match['winner_id'], match['loser_id']
                if p1 > p2:  # Ensure consistent ordering
                    p1, p2 = p2, p1
                player_pairs.add((p1, p2))
            
            logger.info(f"Building features for {len(player_pairs)} {tour} matchups")
            
            for p1, p2 in player_pairs:
                # Get matches between these players
                h2h_matches = tour_matches[
                    ((tour_matches['winner_id'] == p1) & (tour_matches['loser_id'] == p2)) |
                    ((tour_matches['winner_id'] == p2) & (tour_matches['loser_id'] == p1))
                ]
                
                total_matches = len(h2h_matches)
                if total_matches < 2:  # Only include matchups with at least 2 matches
                    continue
                
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
                    'tour': tour
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
                p1_info = self.players_df[self.players_df['player_id'] == p1]
                p2_info = self.players_df[self.players_df['player_id'] == p2]
                
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
                
                matchups.append(matchup_data)
        
        # Convert to DataFrame
        matchup_features_df = pd.DataFrame(matchups)
        
        # Save to parquet if we have data
        if not matchup_features_df.empty:
            output_file = FEATURES_DIR / 'matchup_features.parquet'
            matchup_features_df.to_parquet(output_file, index=False)
            logger.info(f"Saved matchup features to {output_file}")
        else:
            logger.warning("No matchup features generated")
        
        return matchup_features_df
    
    def build_tournament_features(self):
        """Build tournament-level features"""
        logger.info("Building tournament features...")
        
        tournament_features = []
        
        for tour in ['ATP', 'WTA']:
            # Get unique tournaments
            tournaments = self.tournaments_df[self.tournaments_df['tour'] == tour]
            
            for _, tournament in tournaments.iterrows():
                tournament_id = tournament['tournament_id']
                
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
        
        # Save to parquet
        output_file = FEATURES_DIR / 'tournament_features.parquet'
        tournament_features_df.to_parquet(output_file, index=False)
        logger.info(f"Saved tournament features to {output_file}")
        
        return tournament_features_df
    
    def build_all_features(self):
        """Build all feature sets"""
        self.load_data()
        
        player_features = self.build_player_features()
        matchup_features = self.build_matchup_features()
        tournament_features = self.build_tournament_features()
        
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
    
    args = parser.parse_args()
    
    # Set up feature builder
    builder = FeatureBuilder(db_path=args.db_path if args.db_path else DB_PATH)
    builder.load_data()
    
    # Build features based on arguments
    if args.player_only:
        builder.build_player_features()
    elif args.matchup_only:
        builder.build_matchup_features()
    elif args.tournament_only:
        builder.build_tournament_features()
    else:
        # Build all features by default
        builder.build_all_features()
    
    logger.info("Feature building completed successfully")


if __name__ == "__main__":
    main()