# Tennis Match Prediction Model

A data-driven tennis match prediction model using historical ATP and WTA match data from Jeff Sackmann's repositories.

## Overview

This project collects and processes professional tennis match data to build predictive models for match outcomes. It includes comprehensive data ingestion, feature engineering, and machine learning pipelines.

## Features

- **Comprehensive Data Collection**: Ingests ATP and WTA match data from 1968 to present
- **Rich Feature Engineering**: Player performance metrics, matchup analysis, surface specialization
- **Extensible Architecture**: Modular design for easy enhancement and maintenance
- **Database Migration System**: Version-controlled schema evolution

## Installation

### Prerequisites

- Python 3.11 or higher
- Git (for cloning data repositories)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/young-algo/tennis_model.git
cd tennis_model
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .  # Install main package
pip install -e ".[dev]"  # Install development dependencies
```

## Usage

### Data Collection

Collect and process tennis match data:

```bash
python src/data/ingest_sackmann_data.py
```

Optional arguments:
- `--reset-db`: Reset the database before ingestion
- `--check-updates`: Check for updates in source repositories
- `--atp-only`: Process only ATP data
- `--wta-only`: Process only WTA data

### Feature Engineering

Generate features for model training:

```bash
python src/features/build_features.py
```

Optional arguments:
- `--player-only`: Only generate player features
- `--matchup-only`: Only generate matchup features
- `--tournament-only`: Only generate tournament features
- `--rank-cutoff <INT>`: Only generate player features for players ranked better than or equal to this value

### Head-to-Head Prediction

Train a logistic regression model and generate matchup predictions:

```bash
python -m src.models.head_to_head_predictor train

# Predict from a CSV of matchups with columns: player1_id,player2_id
python -m src.models.head_to_head_predictor predict matchups.csv
```

The prediction command will append a `player1_win_prob` column to the input CSV.

### Fetch Upcoming Draw

Download the official draw for the next WTA tournament and output a CSV of
matchups ready for prediction. The script uses the WTA public API and
requires an internet connection:

```bash
python src/data/fetch_upcoming_draw.py --output upcoming_draw.csv
```

The resulting file can be passed directly to `HeadToHeadPredictor`:

```bash
python -m src.models.head_to_head_predictor predict upcoming_draw.csv
```

### Database Migrations

Manage database schema:

```bash
# Create a new migration
python src/data/db_migrations.py create "Add new statistic columns"

# Apply pending migrations
python src/data/db_migrations.py apply

# List migration status
python src/data/db_migrations.py list

# Revert a migration
python src/data/db_migrations.py revert migration_20250508213000
```

## Project Structure

```
tennis_model/
├── data/                      # Data storage
│   ├── external/              # External data sources
│   ├── processed/             # Processed data and features
│   └── raw/                   # Raw data (git repositories)
├── docs/                      # Documentation
│   └── data_schemas.md        # Database schema documentation
├── notebooks/                 # Jupyter notebooks for analysis
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   │   ├── ingest_sackmann_data.py  # Data ingestion script
│   │   ├── db_migrations.py   # Database migration system
│   │   ├── fetch_upcoming_draw.py  # Download upcoming WTA draw
│   │   └── migrations/        # Migration files
│   ├── features/              # Feature engineering
│   │   └── build_features.py  # Feature building script
│   ├── models/                # Machine learning models
│   └── visualization/         # Visualization code
└── tests/                     # Tests
```

## Data Sources

This project uses data from the following repositories by Jeff Sackmann:

- [tennis_atp](https://github.com/JeffSackmann/tennis_atp) - ATP tour match results
- [tennis_wta](https://github.com/JeffSackmann/tennis_wta) - WTA tour match results
- [tennis_MatchChartingProject](https://github.com/JeffSackmann/tennis_MatchChartingProject) - Detailed point-by-point data
- [tennis_pointbypoint](https://github.com/JeffSackmann/tennis_pointbypoint) - Additional point-by-point data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

Special thanks to Jeff Sackmann for maintaining comprehensive tennis match datasets.
