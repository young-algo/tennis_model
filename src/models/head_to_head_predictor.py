import sqlite3
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


class HeadToHeadPredictor:
    """Train and apply a simple logistic regression model for head to head matches."""

    def __init__(
        self,
        db_path: Path | str = Path("data/processed/tennis.db"),
        features_dir: Path | str = Path("data/processed/features"),
        model_path: Path | str = Path("data/processed/h2h_model.pkl"),
    ) -> None:
        self.db_path = Path(db_path)
        self.features_dir = Path(features_dir)
        self.model_path = Path(model_path)
        self.model = LogisticRegression(max_iter=1000)

        self.player_features = None
        self.matchup_features = None
        self.tournament_features = None # Added for tournament features

    # ------------------------------------------------------------------
    def _load_features(self) -> None:
        """Load precomputed player, matchup, and tournament features."""
        player_path = self.features_dir / "player_features.parquet"
        matchup_path = self.features_dir / "matchup_features.parquet"
        tournament_path = self.features_dir / "tournament_features.parquet" # Added

        if player_path.exists():
            self.player_features = pd.read_parquet(player_path)
        else:
            raise FileNotFoundError(f"Player feature file not found: {player_path}")

        if matchup_path.exists():
            self.matchup_features = pd.read_parquet(matchup_path)
        else:
            self.matchup_features = pd.DataFrame() # Allow running without matchup features

        if tournament_path.exists(): # Added for tournament features
            self.tournament_features = pd.read_parquet(tournament_path)
        else:
            self.tournament_features = pd.DataFrame() # Allow running without tournament features

    # ------------------------------------------------------------------
    def _pair_features(self, p1: str, p2: str, surface: str) -> dict:
        """Return feature dictionary for a given ordered pair of players and surface."""

        pf1 = self.player_features[self.player_features["player_id"] == p1]
        pf2 = self.player_features[self.player_features["player_id"] == p2]

        if pf1.empty or pf2.empty:
            raise ValueError("Missing player features for one of the players")

        # Ensure pf1 and pf2 are single rows for safe .iloc[0] access
        pf1 = pf1.iloc[0]
        pf2 = pf2.iloc[0]

        feats = {
            "rank_diff": float(pf1.get("current_ranking", 0) or 0) 
            - float(pf2.get("current_ranking", 0) or 0),
            "win_rate_diff": pf1.get("win_rate", 0.0) - pf2.get("win_rate", 0.0),
        }

        # Dynamic surface-specific win rate difference
        VALID_SURFACES = {'Hard', 'Clay', 'Grass', 'Carpet'}
        surface_specific_win_rate_diff = 0.0
        if surface and surface.capitalize() in VALID_SURFACES:
            surface_col = f"{surface.lower()}_win_rate"
            if surface_col in pf1 and surface_col in pf2:
                p1_surface_wr = pf1[surface_col]
                p2_surface_wr = pf2[surface_col]
                if pd.notna(p1_surface_wr) and pd.notna(p2_surface_wr):
                    surface_specific_win_rate_diff = p1_surface_wr - p2_surface_wr
        
        feats["surface_specific_win_rate_diff"] = surface_specific_win_rate_diff

        if not self.matchup_features.empty:
            # matchup features use sorted player ids
            sp1, sp2 = sorted([p1, p2])
            match_row = self.matchup_features[
                (self.matchup_features["player1_id"] == sp1)
                & (self.matchup_features["player2_id"] == sp2)
            ]
            if not match_row.empty:
                row_data = match_row.iloc[0]
                sign = 1 if (sp1 == p1) else -1
                
                # Overall H2H
                h2h_win_rate_diff = 0.0
                if pd.notna(row_data.get("player1_win_rate")) and pd.notna(row_data.get("player2_win_rate")):
                    h2h_win_rate_diff = sign * (
                        row_data["player1_win_rate"] - row_data["player2_win_rate"]
                    )
                feats["h2h_win_rate_diff"] = h2h_win_rate_diff

                # Surface specific H2H from matchup_features
                surface_specific_h2h_win_rate_diff = 0.0
                if surface and surface.capitalize() in VALID_SURFACES:
                    surface_h2h_p1_col = f"{surface.lower()}_player1_win_rate"
                    surface_h2h_p2_col = f"{surface.lower()}_player2_win_rate"
                    if surface_h2h_p1_col in row_data and surface_h2h_p2_col in row_data:
                        p1_surface_h2h_wr = row_data[surface_h2h_p1_col]
                        p2_surface_h2h_wr = row_data[surface_h2h_p2_col]
                        if pd.notna(p1_surface_h2h_wr) and pd.notna(p2_surface_h2h_wr):
                             surface_specific_h2h_win_rate_diff = sign * (p1_surface_h2h_wr - p2_surface_h2h_wr)
                feats["surface_specific_h2h_win_rate_diff"] = surface_specific_h2h_win_rate_diff
            else:
                feats["h2h_win_rate_diff"] = 0.0
                feats["surface_specific_h2h_win_rate_diff"] = 0.0
        else:
            feats["h2h_win_rate_diff"] = 0.0
            feats["surface_specific_h2h_win_rate_diff"] = 0.0

        return feats

    # ------------------------------------------------------------------
    def _build_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Construct training dataset from historical matches."""
        self._load_features()

        conn = sqlite3.connect(self.db_path)
        # Fetch surface from matches
        matches = pd.read_sql_query(
            "SELECT winner_id, loser_id, surface FROM matches WHERE surface IS NOT NULL",
            conn,
        )
        conn.close()

        X_rows = []
        y = []

        for _, row in matches.iterrows():
            w = row["winner_id"]
            l = row["loser_id"]
            surface = row["surface"] # Get surface

            if not surface: # Skip if surface is None or empty
                continue

            try:
                # Pass surface to _pair_features
                feats_win = self._pair_features(w, l, surface)
                feats_lose = self._pair_features(l, w, surface)
            except ValueError:
                # Skip if player features missing
                continue

            X_rows.append(feats_win)
            y.append(1)
            X_rows.append(feats_lose)
            y.append(0)

        X = pd.DataFrame(X_rows).fillna(0.0) # Fill NaNs that might arise from missing specific surface stats
        y_series = pd.Series(y)
        return X, y_series

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Train the logistic regression model and save it to disk."""
        X, y = self._build_training_data()
        if X.empty:
            print("No training data generated. Ensure your database has matches with surface information.")
            return
        self.model.fit(X, y)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load a previously saved model and associated features."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        # Crucially, load features when loading the model
        self._load_features()

    # ------------------------------------------------------------------
    def predict_matchups(self, pairs: List[Tuple[str, str, str]]) -> List[float]:
        """Predict probability that the first player in each pair wins, given the surface."""
        if self.player_features is None: # Check if features are loaded
            print("Features not loaded. Loading them now...")
            self._load_features() # Load features if not already loaded
        
        if self.model is None: # Check if model is loaded
             # Attempt to load model if not explicitly trained/loaded before predict
            try:
                print("Model not loaded. Loading model from default path...")
                self.load_model()
            except FileNotFoundError:
                raise RuntimeError(
                    "Model not found. Train the model first or provide a valid model path."
                )


        probs = []
        for p1, p2, surface in pairs: # Include surface in iteration
            try:
                feats = self._pair_features(p1, p2, surface) # Pass surface
                X = pd.DataFrame([feats]).fillna(0.0) # Fill NaNs for robustness
                prob = self.model.predict_proba(X)[0, 1]
                probs.append(prob)
            except ValueError as e:
                print(f"Skipping prediction for ({p1}, {p2}, {surface}): {e}")
                probs.append(float('nan')) # Append NaN for pairs that couldn't be processed
        return probs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Head to head prediction model")
    subparsers = parser.add_subparsers(dest="command")

    train_p = subparsers.add_parser("train", help="Train the model")
    train_p.add_argument(
        "--db-path", type=Path, default=Path("data/processed/tennis.db")
    )
    train_p.add_argument(
        "--features-dir", type=Path, default=Path("data/processed/features")
    )
    train_p.add_argument(
        "--model-path", type=Path, default=Path("data/processed/h2h_model.pkl")
    )

    pred_p = subparsers.add_parser("predict", help="Predict matchups from a CSV")
    pred_p.add_argument(
        "matchups_csv", type=Path, help="CSV with player1_id,player2_id,surface" # Updated help text
    )
    pred_p.add_argument(
        "--model-path", type=Path, default=Path("data/processed/h2h_model.pkl")
    )
    pred_p.add_argument(
        "--features-dir", type=Path, default=Path("data/processed/features")
    )

    args = parser.parse_args()

    if args.command == "train":
        predictor = HeadToHeadPredictor(
            db_path=args.db_path,
            features_dir=args.features_dir,
            model_path=args.model_path,
        )
        predictor.train()
    elif args.command == "predict":
        predictor = HeadToHeadPredictor(
            # db_path is not strictly needed for predict if features are pre-loaded,
            # but load_model now calls _load_features which might use self.db_path implicitly
            # if we were to extend it further. Let's keep it for consistency for now.
            db_path=Path("data/processed/tennis.db"), # Assuming a default, though not directly used by predict if model exists
            features_dir=args.features_dir,
            model_path=args.model_path,
        )
        # load_model now also loads features
        predictor.load_model() 
        
        df = pd.read_csv(args.matchups_csv)
        # Read surface from CSV
        if not {"player1_id", "player2_id", "surface"}.issubset(df.columns):
            raise ValueError("CSV must contain 'player1_id', 'player2_id', and 'surface' columns.")
            
        pairs = list(zip(df["player1_id"], df["player2_id"], df["surface"]))
        probs = predictor.predict_matchups(pairs)
        df["player1_win_prob"] = probs
        output_csv = args.matchups_csv.with_name(
            args.matchups_csv.stem + "_predictions.csv"
        )
        df.to_csv(output_csv, index=False)
        print(f"Saved predictions to {output_csv}")
    else:
        parser.print_help()
