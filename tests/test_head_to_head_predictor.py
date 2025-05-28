import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
import tempfile
import sqlite3
import joblib
import argparse

# Adjust PYTHONPATH to import HeadToHeadPredictor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.head_to_head_predictor import HeadToHeadPredictor
# Import main from script for CLI test, ensure it's importable or adjust structure
try:
    from src.models.head_to_head_predictor import main as predictor_main
except ImportError:
    # This might happen if main() is not easily separable or if __name__ == "__main__": guard is too strict
    predictor_main = None 


class TestHeadToHeadPredictor(unittest.TestCase):

    def _create_mock_df(self, data_list):
        return pd.DataFrame(data_list)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.db_path = self.temp_path / "test_tennis.db"
        self.features_dir = self.temp_path / "features"
        self.model_path = self.temp_path / "test_h2h_model.pkl"
        self.features_dir.mkdir(parents=True, exist_ok=True)

        self.P1, self.P2, self.P3 = 'P1', 'P2', 'P3'

        # Mock Player Features
        player_data = [
            {'player_id': self.P1, 'current_ranking': 10, 'win_rate': 0.6, 'hard_win_rate': 0.5, 'clay_win_rate': 0.8, 'grass_win_rate': 0.5, 'carpet_win_rate': 0.5, 'name': 'Player One', 'hand': 'R', 'country': 'USA', 'tour': 'ATP'},
            {'player_id': self.P2, 'current_ranking': 15, 'win_rate': 0.6, 'hard_win_rate': 0.8, 'clay_win_rate': 0.5, 'grass_win_rate': 0.5, 'carpet_win_rate': 0.5, 'name': 'Player Two', 'hand': 'L', 'country': 'ESP', 'tour': 'ATP'},
            {'player_id': self.P3, 'current_ranking': 20, 'win_rate': 0.6, 'hard_win_rate': 0.6, 'clay_win_rate': 0.6, 'grass_win_rate': 0.6, 'carpet_win_rate': 0.6, 'name': 'Player Three', 'hand': 'R', 'country': 'SRB', 'tour': 'ATP'},
        ]
        self.player_features_df = self._create_mock_df(player_data)
        self.player_features_df.to_parquet(self.features_dir / "player_features.parquet")

        # Mock Matchup Features
        # For P1 vs P2: P1=player1_id, P2=player2_id (sorted)
        # P1 strong on Clay H2H, P2 strong on Hard H2H
        matchup_data = [
            {
                'player1_id': self.P1, 'player2_id': self.P2, 'tour': 'ATP',
                'total_matches': 4, 
                'player1_wins': 2, 'player2_wins': 2, # Overall H2H
                'player1_win_rate': 0.5, 'player2_win_rate': 0.5,
                'clay_matches': 2, 'clay_player1_wins': 2, 'clay_player2_wins': 0,
                'clay_player1_win_rate': 1.0, 'clay_player2_win_rate': 0.0,
                'hard_matches': 2, 'hard_player1_wins': 0, 'hard_player2_wins': 2,
                'hard_player1_win_rate': 0.0, 'hard_player2_win_rate': 1.0,
                # Add other surfaces with neutral or no H2H
                'grass_matches': 0, 'grass_player1_wins': 0, 'grass_player2_wins': 0,
                'grass_player1_win_rate': None, 'grass_player2_win_rate': None, # Use None for pandas NA
            }
        ]
        self.matchup_features_df = self._create_mock_df(matchup_data)
        # Fill NA for win rates where matches are 0, pandas will write them as nulls
        for surface in ['grass', 'carpet']: # Assuming carpet also has 0 matches
             for p_num in ['1', '2']:
                self.matchup_features_df[f'{surface}_player{p_num}_win_rate'] = self.matchup_features_df.apply(
                    lambda row: row[f'{surface}_player{p_num}_wins'] / row[f'{surface}_matches'] if row[f'{surface}_matches'] > 0 else None, axis=1
                )

        self.matchup_features_df.to_parquet(self.features_dir / "matchup_features.parquet")


        # Mock Tournament Features (minimal)
        tournament_data = [{'tournament_id': 'T1', 'surface': 'Hard'}]
        self.tournament_features_df = self._create_mock_df(tournament_data)
        self.tournament_features_df.to_parquet(self.features_dir / "tournament_features.parquet")

        self.predictor = HeadToHeadPredictor(
            db_path=self.db_path,
            features_dir=self.features_dir,
            model_path=self.model_path
        )
        
        # Setup for in-memory DB for training test
        self.conn = sqlite3.connect(':memory:') # Use in-memory for training test
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
        CREATE TABLE matches (
            winner_id TEXT, loser_id TEXT, surface TEXT,
            match_date TEXT, tournament_id TEXT, tournament_level TEXT,
            winner_rank INTEGER, loser_rank INTEGER, minutes INTEGER
        )
        """)
        self.conn.commit()


    def tearDown(self):
        self.conn.close() # Close in-memory DB connection
        self.temp_dir.cleanup()

    def test_predict_surface_sensitivity(self):
        # Load features into predictor
        self.predictor._load_features()

        # Mock the model's predict_proba behavior
        # It should give higher prob if the sum of key diff features is positive
        def mock_predict_proba(X):
            # X is a DataFrame with features
            # surface_specific_win_rate_diff
            # surface_specific_h2h_win_rate_diff (if matchup features are used)
            val = X['surface_specific_win_rate_diff'].iloc[0] + X.get('surface_specific_h2h_win_rate_diff', 0.0).iloc[0]
            # Simple sigmoid-like scaling for probability
            prob_p1_wins = 1 / (1 + pd.np.exp(-val)) 
            return pd.np.array([[1 - prob_p1_wins, prob_p1_wins]])

        self.predictor.model = MagicMock()
        self.predictor.model.predict_proba = MagicMock(side_effect=mock_predict_proba)

        # P1 vs P2 on Clay
        prob_p1_clay = self.predictor.predict_matchups([(self.P1, self.P2, 'Clay')])[0]
        # P1 vs P2 on Hard
        prob_p1_hard = self.predictor.predict_matchups([(self.P1, self.P2, 'Hard')])[0]
        
        self.assertTrue(pd.notna(prob_p1_clay) and pd.notna(prob_p1_hard), "Probabilities should not be NaN")
        self.assertGreater(prob_p1_clay, prob_p1_hard, "P1 win prob should be higher on Clay than Hard vs P2")

        # P2 vs P1 on Clay
        prob_p2_clay = self.predictor.predict_matchups([(self.P2, self.P1, 'Clay')])[0]
        # P2 vs P1 on Hard
        prob_p2_hard = self.predictor.predict_matchups([(self.P2, self.P1, 'Hard')])[0]
        
        self.assertTrue(pd.notna(prob_p2_clay) and pd.notna(prob_p2_hard), "Probabilities should not be NaN")
        self.assertGreater(prob_p2_hard, prob_p2_clay, "P2 win prob should be higher on Hard than Clay vs P1")
        
        # Test a neutral surface (Grass) where H2H data might be missing or neutral
        # P1 vs P2 on Grass (expecting closer to player's general grass_win_rate if H2H is neutral/missing)
        # From player_features, P1 and P2 have same grass_win_rate (0.5)
        # From matchup_features, grass H2H is neutral (0 matches, so rates are None, leading to 0 diff)
        # So, surface_specific_win_rate_diff should be 0, surface_specific_h2h_win_rate_diff should be 0
        # Resulting val in mock_predict_proba should be 0, prob_p1_wins = 0.5
        prob_p1_grass = self.predictor.predict_matchups([(self.P1, self.P2, 'Grass')])[0]
        self.assertEqual(prob_p1_grass, 0.5, "P1 win prob on Grass vs P2 should be 0.5 due to neutral stats")


    def test_training_and_prediction_e2e(self):
        # Use the in-memory DB for this test by overriding predictor's db_path
        self.predictor.db_path = self.db_path # Point to a file DB for training, as in-memory is tricky with class re-init
        
        # Re-init predictor for this test to use file-based DB
        # Or ensure the self.conn is used by the predictor instance.
        # For simplicity, let's write to the file DB self.db_path
        conn_file = sqlite3.connect(self.db_path)
        cursor_file = conn_file.cursor()
        cursor_file.execute("DROP TABLE IF EXISTS matches") # Ensure clean table
        cursor_file.execute("""
        CREATE TABLE matches (
            winner_id TEXT, loser_id TEXT, surface TEXT,
            match_date TEXT, tournament_id TEXT, tournament_level TEXT,
            winner_rank INTEGER, loser_rank INTEGER, minutes INTEGER
        )""")
        
        # Mock match data for training
        training_matches = [
            (self.P1, self.P2, 'Clay', '2023-01-01', 'T_Clay', 'G', 10, 15, 120),
            (self.P2, self.P1, 'Hard', '2023-01-08', 'T_Hard', 'G', 15, 10, 90),
            (self.P1, self.P3, 'Clay', '2023-01-15', 'T_Clay2', 'M', 10, 20, 100),
            (self.P2, self.P3, 'Hard', '2023-01-22', 'T_Hard2', 'M', 15, 20, 80),
            # Add a few more to make model more stable
            (self.P1, self.P3, 'Clay', '2023-02-01', 'T_Clay3', 'A', 10, 20, 110), 
            (self.P3, self.P2, 'Clay', '2023-02-08', 'T_Clay4', 'A', 20, 15, 130), # P3 beats P2 on clay
        ]
        cursor_file.executemany("INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?)", training_matches)
        conn_file.commit()
        conn_file.close()

        # Train the model
        self.predictor.train()
        self.assertTrue(self.model_path.exists(), "Model file was not created after training.")

        # Load the trained model (implicitly done by predict_matchups if model is None, but good to be explicit)
        self.predictor.load_model() # This also loads features

        # Predictions
        prob_p1_vs_p2_clay = self.predictor.predict_matchups([(self.P1, self.P2, 'Clay')])[0]
        prob_p1_vs_p2_hard = self.predictor.predict_matchups([(self.P1, self.P2, 'Hard')])[0]
        
        self.assertTrue(pd.notna(prob_p1_vs_p2_clay) and pd.notna(prob_p1_vs_p2_hard), "Probabilities should not be NaN")
        # P1 is strong on clay (player features) and has H2H clay advantage.
        # P2 is strong on hard (player features) and has H2H hard advantage.
        # The training data reinforces this.
        self.assertGreater(prob_p1_vs_p2_clay, prob_p1_vs_p2_hard, "P1 win prob vs P2 should be higher on Clay than Hard after training.")

        # Test P3 vs P1 on Clay - P1 should be favored
        prob_p3_vs_p1_clay = self.predictor.predict_matchups([(self.P3, self.P1, 'Clay')])[0] # P3 is row.loser_id, P1 is row.winner_id
        self.assertLess(prob_p3_vs_p1_clay, 0.5, "P3's probability of beating P1 on Clay should be less than 0.5.")


    @unittest.skipIf(predictor_main is None, "predictor_main not imported, skipping CLI test")
    @patch('argparse.ArgumentParser.parse_args')
    def test_cli_predict_format(self, mock_parse_args):
        # Create a dummy trained model file
        dummy_model_obj = {'model': 'dummy'} # A simple dict can act as a model for joblib
        joblib.dump(dummy_model_obj, self.model_path)
        
        # Create a dummy input CSV for prediction
        input_csv_path = self.temp_path / "input_matchups.csv"
        input_matchups_data = [
            {'player1_id': self.P1, 'player2_id': self.P2, 'surface': 'Clay'},
            {'player1_id': self.P1, 'player2_id': self.P2, 'surface': 'Hard'},
        ]
        input_df = pd.DataFrame(input_matchups_data)
        input_df.to_csv(input_csv_path, index=False)

        # Mock args for the CLI
        mock_args = argparse.Namespace(
            command='predict',
            matchups_csv=input_csv_path,
            model_path=self.model_path,
            features_dir=self.features_dir,
            db_path=self.db_path # Though not used by predict if model exists
        )
        mock_parse_args.return_value = mock_args
        
        # Mock predictor's predict_matchups to return fixed values for CLI test simplicity
        with patch.object(HeadToHeadPredictor, 'predict_matchups', return_value=[0.7, 0.3]) as mock_predict:
            predictor_main() # Call the main function from the script

            mock_predict.assert_called_once()
            # Check args passed to predict_matchups
            call_args = mock_predict.call_args[0][0] # First positional argument
            self.assertEqual(len(call_args), 2)
            self.assertEqual(call_args[0], (self.P1, self.P2, 'Clay'))
            self.assertEqual(call_args[1], (self.P1, self.P2, 'Hard'))

        # Check output file
        expected_output_csv_path = self.temp_path / "input_matchups_predictions.csv"
        self.assertTrue(expected_output_csv_path.exists(), "Output prediction CSV was not created.")
        
        output_df = pd.read_csv(expected_output_csv_path)
        self.assertTrue('player1_win_prob' in output_df.columns)
        self.assertEqual(len(output_df), 2)
        self.assertEqual(output_df['player1_win_prob'].iloc[0], 0.7)
        self.assertEqual(output_df['player1_win_prob'].iloc[1], 0.3)


if __name__ == '__main__':
    unittest.main()
