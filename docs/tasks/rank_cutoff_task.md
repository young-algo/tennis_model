# Rank Cutoff Feature

## Problem
Feature generation currently processes all players present in the match data. This can include thousands of players and dramatically slows down execution.

## Task
Add the ability to provide a ranking cutoff so that player features are only generated for players whose most recent ranking is better than or equal to the cutoff.

## Proposed Solution
- Expose a `--rank-cutoff` option in the feature generation CLI and the main entry point.
- Store this value in `FeatureBuilder` and apply it when determining the set of players to process.
- Skip players with no ranking information or rankings worse than the specified cutoff.
