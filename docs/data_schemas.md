# Tennis Data Model Schema Documentation

This document outlines the schema of the tennis data collected from Jeff Sackmann's repositories and the features engineered from this data.

## Database Structure

The tennis data is stored in an SQLite database (`data/processed/tennis.db`) with the following tables:

- **matches**: Contains match results and statistics
- **players**: Player biographical information
- **tournaments**: Tournament metadata
- **rankings**: Historical player rankings
- **file_metadata**: Tracks processed files for incremental updates

## Table Schemas

### matches

Contains match results and detailed statistics for both ATP and WTA matches.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| match_id | TEXT | Unique match identifier (PRIMARY KEY) |
| tournament_id | TEXT | Tournament identifier (foreign key to tournaments) |
| tournament_name | TEXT | Tournament name |
| surface | TEXT | Playing surface (Hard, Clay, Grass, Carpet) |
| tour | TEXT | ATP or WTA |
| match_date | TEXT | Date of match (YYYY-MM-DD format) |
| match_round | TEXT | Round of tournament (e.g., F, SF, QF, R16, R32, R64, R128, Q1, Q2) |
| best_of | INTEGER | Number of sets to win match (3 or 5) |
| winner_id | TEXT | ID of match winner (foreign key to players) |
| winner_name | TEXT | Name of match winner |
| winner_hand | TEXT | Dominant hand of winner (L=left, R=right, U=unknown) |
| winner_ht | INTEGER | Height of winner in cm |
| winner_ioc | TEXT | Nationality of winner (IOC code) |
| winner_age | REAL | Age of winner in years |
| winner_rank | INTEGER | ATP/WTA ranking of winner at time of match |
| winner_rank_points | INTEGER | Ranking points of winner at time of match |
| loser_id | TEXT | ID of match loser (foreign key to players) |
| loser_name | TEXT | Name of match loser |
| loser_hand | TEXT | Dominant hand of loser (L=left, R=right, U=unknown) |
| loser_ht | INTEGER | Height of loser in cm |
| loser_ioc | TEXT | Nationality of loser (IOC code) |
| loser_age | REAL | Age of loser in years |
| loser_rank | INTEGER | ATP/WTA ranking of loser at time of match |
| loser_rank_points | INTEGER | Ranking points of loser at time of match |
| score | TEXT | Match score |
| w_sets_won | INTEGER | Number of sets won by match winner |
| l_sets_won | INTEGER | Number of sets won by match loser |
| w_games_won | INTEGER | Number of games won by match winner |
| l_games_won | INTEGER | Number of games won by match loser |
| w_ace | INTEGER | Aces by match winner |
| w_df | INTEGER | Double faults by match winner |
| w_svpt | INTEGER | Serve points played by winner |
| w_1stIn | INTEGER | First serves in by winner |
| w_1stWon | INTEGER | First serve points won by winner |
| w_2ndWon | INTEGER | Second serve points won by winner |
| w_SvGms | INTEGER | Service games played by winner |
| w_bpSaved | INTEGER | Break points saved by winner |
| w_bpFaced | INTEGER | Break points faced by winner |
| l_ace | INTEGER | Aces by match loser |
| l_df | INTEGER | Double faults by match loser |
| l_svpt | INTEGER | Serve points played by loser |
| l_1stIn | INTEGER | First serves in by loser |
| l_1stWon | INTEGER | First serve points won by loser |
| l_2ndWon | INTEGER | Second serve points won by loser |
| l_SvGms | INTEGER | Service games played by loser |
| l_bpSaved | INTEGER | Break points saved by loser |
| l_bpFaced | INTEGER | Break points faced by loser |
| minutes | INTEGER | Match duration in minutes |
| tournament_level | TEXT | Tournament level (G=Grand Slam, M=Masters, A=ATP 500, D=ATP 250, F=Futures, etc.) |

### players

Contains biographical information about tennis players.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| player_id | TEXT | Unique player identifier (PRIMARY KEY) |
| first_name | TEXT | Player's first name |
| last_name | TEXT | Player's last name |
| hand | TEXT | Dominant hand (L=left, R=right, U=unknown) |
| birth_date | TEXT | Date of birth (YYYY-MM-DD format) |
| country_code | TEXT | Nationality (IOC country code) |
| height | INTEGER | Height in cm |
| tour | TEXT | ATP or WTA |

### tournaments

Contains information about tennis tournaments.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| tournament_id | TEXT | Unique tournament identifier (PRIMARY KEY) |
| tournament_name | TEXT | Tournament name |
| surface | TEXT | Playing surface (Hard, Clay, Grass, Carpet) |
| draw_size | INTEGER | Number of players in main draw |
| tournament_level | TEXT | Tournament level code |
| court_type | TEXT | Indoor/Outdoor |
| tour | TEXT | ATP or WTA |

### rankings

Contains historical ATP and WTA rankings.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| ranking_id | INTEGER | Unique ranking entry identifier (PRIMARY KEY) |
| ranking_date | TEXT | Date of ranking (YYYY-MM-DD format) |
| player_id | TEXT | Player identifier (foreign key to players) |
| ranking | INTEGER | Numeric ranking |
| ranking_points | INTEGER | Ranking points |
| tour | TEXT | ATP or WTA |

### file_metadata

Tracks processed files for incremental updates.

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| file_path | TEXT | Path to processed file (PRIMARY KEY) |
| last_modified | TEXT | Last modification timestamp |
| checksum | TEXT | File checksum to detect changes |
| last_processed | TEXT | Timestamp when file was last processed |

## Engineered Features

The system generates several engineered features from the raw data. These are stored as Parquet files in the `data/processed/features/` directory.

### Player Features (`player_features.parquet`)

| Feature Category | Description | Examples |
|------------------|-------------|----------|
| Basic Performance | Overall performance metrics | total_matches, wins, losses, win_rate |
| Surface Performance | Performance metrics by surface | hard_matches, hard_win_rate, clay_matches, clay_win_rate |
| Tournament Level | Performance by tournament level | level_G_matches, level_G_win_rate, level_M_matches, level_M_win_rate |
| Recent Form | Performance over recent matches | last_5_matches_win_rate, last_10_matches_win_rate, last_20_matches_win_rate |
| Service Metrics | Serving performance | avg_ace, avg_df, first_serve_pct, first_serve_win_pct, second_serve_win_pct |
| Return Metrics | Return performance | break_points_saved_pct, serve_points_won_pct |
| Set/Game Statistics | Set and game performance | sets_won, sets_lost, games_won, games_lost, sets_win_pct, games_win_pct |
| Ranking Metrics | Ranking information | current_ranking, current_points, ranking_change_3m, ranking_change_6m, ranking_change_12m |

### Matchup Features (`matchup_features.parquet`)

| Feature Category | Description | Examples |
|------------------|-------------|----------|
| Head-to-Head | Direct matchup history | total_matches, player1_wins, player2_wins, player1_win_rate, player2_win_rate |
| Surface H2H | Surface-specific H2H | hard_matches, hard_player1_win_rate, clay_matches, clay_player1_win_rate |
| Recent Matchups | Recent matchup results | recent_player1_win_rate, recent_player2_win_rate |
| Physical Matchup | Physical comparison | height_diff, dominant_hand_matchup |

### Tournament Features (`tournament_features.parquet`)

| Feature Category | Description | Examples |
|------------------|-------------|----------|
| Basic Statistics | Tournament statistics | total_matches, years_active, unique_players |
| Player Quality | Player rankings | avg_winner_rank, avg_loser_rank |
| Match Characteristics | Match information | avg_match_duration |
| Surface Distribution | Surface statistics | hard_matches, hard_pct, clay_matches, clay_pct |

## Data Sources

The data is sourced from Jeff Sackmann's GitHub repositories:

- **ATP Data**: [tennis_atp](https://github.com/JeffSackmann/tennis_atp) - ATP tour match results
- **WTA Data**: [tennis_wta](https://github.com/JeffSackmann/tennis_wta) - WTA tour match results
- **Match Charting Project**: [tennis_MatchChartingProject](https://github.com/JeffSackmann/tennis_MatchChartingProject) - Detailed point-by-point data
- **Point-by-Point Data**: [tennis_pointbypoint](https://github.com/JeffSackmann/tennis_pointbypoint) - Additional point-by-point data

## Date Formats and Conversions

- Dates in the source data may be in YYYYMMDD format (e.g., "20220101")
- These are converted to YYYY-MM-DD format (e.g., "2022-01-01") in the database
- When loaded into pandas DataFrames, dates are converted to datetime objects

## Match ID Generation

Match IDs are generated using the following pattern:
```
{tournament_id}-{date}-{round}-{winner_id}-{loser_id}
```
This ensures unique identification of each match.

## Indexes

The following indexes are created to optimize query performance:

- matches(winner_id)
- matches(loser_id)
- matches(tournament_id)
- matches(match_date)
- matches(surface)
- matches(tour)
- rankings(player_id)
- rankings(ranking_date)
- players(country_code)

## Database Update Process

The system uses a file checksum approach to track changes and only process updated files:

1. Calculates SHA-256 checksum for each source file
2. Compares with stored checksums to detect changes
3. Only processes files that have changed or are new
4. Updates the file_metadata table with new checksums

## Usage Notes

- Match statistics (aces, double faults, etc.) are not available for all matches, especially older ones
- WTA matches are best-of-3 sets, while ATP matches can be best-of-3 or best-of-5 (in Grand Slams)
- Tournament levels have different meaning in ATP and WTA tours