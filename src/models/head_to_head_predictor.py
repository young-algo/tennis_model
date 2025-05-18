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

    # ------------------------------------------------------------------
    def _load_features(self) -> None:
        """Load precomputed player and matchup features."""
        player_path = self.features_dir / "player_features.parquet"
        matchup_path = self.features_dir / "matchup_features.parquet"

        if player_path.exists():
            self.player_features = pd.read_parquet(player_path)
        else:
            raise FileNotFoundError(f"Player feature file not found: {player_path}")

        if matchup_path.exists():
            self.matchup_features = pd.read_parquet(matchup_path)
        else:
            self.matchup_features = pd.DataFrame()

    # ------------------------------------------------------------------
    def _pair_features(self, p1: str, p2: str) -> dict:
        """Return feature dictionary for a given ordered pair of players."""

        pf1 = self.player_features[self.player_features["player_id"] == p1]
        pf2 = self.player_features[self.player_features["player_id"] == p2]

        if pf1.empty or pf2.empty:
            raise ValueError("Missing player features for one of the players")

        feats = {
            "rank_diff": float(pf1["current_ranking"].iloc[0] or 0)
            - float(pf2["current_ranking"].iloc[0] or 0),
            "win_rate_diff": pf1["win_rate"].iloc[0] - pf2["win_rate"].iloc[0],
            "clay_win_rate_diff": pf1["clay_win_rate"].iloc[0]
            - pf2["clay_win_rate"].iloc[0],
        }

        if not self.matchup_features.empty:
            # matchup features use sorted player ids
            sp1, sp2 = sorted([p1, p2])
            match_row = self.matchup_features[
                (self.matchup_features["player1_id"] == sp1)
                & (self.matchup_features["player2_id"] == sp2)
            ]
            if not match_row.empty:
                row = match_row.iloc[0]
                sign = 1 if (sp1 == p1) else -1
                feats["h2h_win_rate_diff"] = sign * (
                    row["player1_win_rate"] - row["player2_win_rate"]
                )
            else:
                feats["h2h_win_rate_diff"] = 0.0
        else:
            feats["h2h_win_rate_diff"] = 0.0

        return feats

    # ------------------------------------------------------------------
    def _build_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Construct training dataset from historical matches."""
        self._load_features()

        conn = sqlite3.connect(self.db_path)
        matches = pd.read_sql_query(
            "SELECT winner_id, loser_id FROM matches",
            conn,
        )
        conn.close()

        X_rows = []
        y = []

        for _, row in matches.iterrows():
            w = row["winner_id"]
            l = row["loser_id"]

            try:
                feats_win = self._pair_features(w, l)
                feats_lose = self._pair_features(l, w)
            except ValueError:
                # Skip if player features missing
                continue

            X_rows.append(feats_win)
            y.append(1)
            X_rows.append(feats_lose)
            y.append(0)

        X = pd.DataFrame(X_rows)
        y_series = pd.Series(y)
        return X, y_series

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Train the logistic regression model and save it to disk."""
        X, y = self._build_training_data()
        self.model.fit(X, y)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load a previously saved model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        self._load_features()

    # ------------------------------------------------------------------
    def predict_matchups(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict probability that the first player in each pair wins."""
        if self.model is None:
            self.load_model()

        probs = []
        for p1, p2 in pairs:
            feats = self._pair_features(p1, p2)
            X = pd.DataFrame([feats])
            prob = self.model.predict_proba(X)[0, 1]
            probs.append(prob)
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
        "matchups_csv", type=Path, help="CSV with player1_id,player2_id"
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
            features_dir=args.features_dir,
            model_path=args.model_path,
        )
        predictor.load_model()
        df = pd.read_csv(args.matchups_csv)
        pairs = list(zip(df["player1_id"], df["player2_id"]))
        probs = predictor.predict_matchups(pairs)
        df["player1_win_prob"] = probs
        output_csv = args.matchups_csv.with_name(
            args.matchups_csv.stem + "_predictions.csv"
        )
        df.to_csv(output_csv, index=False)
        print(f"Saved predictions to {output_csv}")
    else:
        parser.print_help()
