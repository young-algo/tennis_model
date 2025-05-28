import sys
import os
import unittest
import pandas as pd
from pathlib import Path
import tempfile

# Adjust PYTHONPATH to import FeatureBuilder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.build_features import FeatureBuilder
import src.features.build_features as bf # To modify FEATURES_DIR

class TestFeatureBuilder(unittest.TestCase):

    def setUp(self):
        self.current_year = pd.Timestamp.now().year
        
        # Create a temporary directory for feature outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.features_dir_path = Path(self.temp_dir.name) / 'features'
        self.features_dir_path.mkdir(parents=True, exist_ok=True)

        # Monkey patch FEATURES_DIR in the build_features module
        self.original_features_dir = bf.FEATURES_DIR
        bf.FEATURES_DIR = self.features_dir_path

        # Initialize FeatureBuilder
        # Using db_path=':memory:' to avoid actual DB file creation/connection issues for this unit test
        # Setting incremental to False and since_date to None as per typical non-incremental run
        self.builder = FeatureBuilder(db_path=':memory:', incremental=False, since_date=None)

        # Mock tournaments_df
        self.mock_tournaments_df = pd.DataFrame({
            'tournament_id': ['T1', 'T2', 'T3_WTA'],
            'tournament_name': ['Recent ATP Tournament', 'Old ATP Tournament', 'Recent WTA Tournament'],
            'surface': ['Hard', 'Clay', 'Hard'],
            'tournament_level': ['G', 'M', 'G'],
            'draw_size': [128, 64, 128],
            'tour': ['ATP', 'ATP', 'WTA'] 
        })

        # Mock matches_df
        # Ensure match_date is a datetime object
        self.mock_matches_df = pd.DataFrame({
            'tournament_id': ['T1', 'T1', 'T2', 'T3_WTA'],
            'match_date': [
                pd.Timestamp(f'{self.current_year - 2}-01-15'), # Recent ATP
                pd.Timestamp(f'{self.current_year - 2}-01-16'), # Recent ATP
                pd.Timestamp(f'{self.current_year - 5}-06-10'), # Old ATP
                pd.Timestamp(f'{self.current_year - 1}-03-20')  # Recent WTA
            ],
            'winner_id': [101, 102, 201, 301],
            'loser_id': [103, 104, 202, 302],
            'winner_rank': [10, 20, 5, 15],
            'loser_rank': [50, 60, 25, 55],
            'minutes': [120, 90, 150, 100],
            'surface': ['Hard', 'Hard', 'Clay', 'Hard'], # Match surface should align with tournament
            'tournament_level': ['G', 'G', 'M', 'G'] # Match level should align
        })
        
        # Assign mock data to the builder instance
        self.builder.tournaments_df = self.mock_tournaments_df
        self.builder.matches_df = self.mock_matches_df
        # Ensure the builder doesn't try to load from DB by providing minimal player data
        # if other methods were called. For build_tournament_features, it's not strictly needed
        # but good practice if the builder instance is more generally used.
        self.builder.players_df = pd.DataFrame({
            'player_id': [101,102,103,104,201,202,301,302],
            'first_name': ['P']*8, 'last_name':['N']*8, 'hand':['R']*8, 'country_code':['USA']*8, 'tour':['ATP']*6 + ['WTA']*2
        })


    def tearDown(self):
        # Restore original FEATURES_DIR
        bf.FEATURES_DIR = self.original_features_dir
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_tournament_filtering(self):
        # Call the method under test
        self.builder.build_tournament_features()

        # Check the output parquet file
        output_file = self.features_dir_path / 'tournament_features.parquet'
        self.assertTrue(output_file.exists(), "Output parquet file was not created.")
        
        result_df = pd.read_parquet(output_file)

        # Assertions
        # Tournament T1 (recent ATP) should be included
        self.assertTrue('T1' in result_df['tournament_id'].values, "Recent ATP tournament T1 is missing.")
        # Tournament T3_WTA (recent WTA) should be included
        self.assertTrue('T3_WTA' in result_df['tournament_id'].values, "Recent WTA tournament T3_WTA is missing.")
        
        # Tournament T2 (old ATP) should NOT be included
        self.assertFalse('T2' in result_df['tournament_id'].values, "Old ATP tournament T2 should have been filtered out.")
        
        # Verify the number of tournaments included
        # We expect T1 and T3_WTA
        self.assertEqual(len(result_df), 2, "Incorrect number of tournaments in the output.")

        # Verify last_year for included tournaments
        t1_last_year = result_df[result_df['tournament_id'] == 'T1']['last_year'].iloc[0]
        self.assertEqual(t1_last_year, self.current_year - 2, "Incorrect last_year for T1.")

        t3_last_year = result_df[result_df['tournament_id'] == 'T3_WTA']['last_year'].iloc[0]
        self.assertEqual(t3_last_year, self.current_year - 1, "Incorrect last_year for T3_WTA.")


if __name__ == '__main__':
    unittest.main()
