#!/usr/bin/env python
"""
Tennis Model - Main Entry Point

This script serves as the entry point for the tennis prediction model.
It provides a command-line interface to run different components of the system.
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('tennis_model')


def setup_data_directories():
    """Create necessary data directories if they don't exist"""
    directories = [
        Path('data'),
        Path('data/external'),
        Path('data/processed'),
        Path('data/raw'),
        Path('notebooks'),
        Path('tests')
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
        logger.info(f"Ensured directory exists: {directory}")


def ingest_data(args):
    """Run the data ingestion process"""
    from src.data.ingest_sackmann_data import TennisDataIngestor
    
    logger.info("Starting data ingestion process...")
    ingestor = TennisDataIngestor(reset_db=args.reset_db)
    ingestor.check_updates = args.check_updates
    
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


def build_features(args):
    """Run the feature engineering process"""
    from src.features.build_features import FeatureBuilder
    
    logger.info("Starting feature engineering process...")
    builder = FeatureBuilder()
    
    if args.player_only:
        builder.build_player_features()
    elif args.matchup_only:
        builder.build_matchup_features()
    elif args.tournament_only:
        builder.build_tournament_features()
    else:
        builder.build_all_features()
    
    logger.info("Feature engineering completed successfully")


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='Tennis Match Prediction Model')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up data directories')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest tennis match data')
    ingest_parser.add_argument('--reset-db', action='store_true', help='Reset the database before ingestion')
    ingest_parser.add_argument('--check-updates', action='store_true', help='Check for updates in source repositories')
    ingest_parser.add_argument('--atp-only', action='store_true', help='Only process ATP data')
    ingest_parser.add_argument('--wta-only', action='store_true', help='Only process WTA data')
    
    # Features command
    features_parser = subparsers.add_parser('features', help='Build features for model training')
    features_parser.add_argument('--player-only', action='store_true', help='Only build player features')
    features_parser.add_argument('--matchup-only', action='store_true', help='Only build matchup features')
    features_parser.add_argument('--tournament-only', action='store_true', help='Only build tournament features')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'setup':
        setup_data_directories()
    elif args.command == 'ingest':
        ingest_data(args)
    elif args.command == 'features':
        build_features(args)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
