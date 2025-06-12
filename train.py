import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict # Useful for Elo and history tracking

# For modeling
from sklearn.model_selection import train_test_split # For train/test splitting
from sklearn.preprocessing import StandardScaler # For scaling numerical features
from sklearn.pipeline import Pipeline # To chain preprocessing and model
from sklearn.linear_model import LogisticRegression # A good baseline model
# You can uncomment and add other models later:
# from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score # Evaluation metrics


# --- 1. Load Data ---
try:
    # Use low_memory=False if you get dtype warnings or memory issues with a very large file
    games_players_df = pd.read_csv('data/combined_file.csv', low_memory=False)
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure 'combined_file.csv' is in the 'data' subdirectory.")
    exit() # Exit if data can't be loaded

print("Data loaded successfully.")
print(f"Raw data shape (player-game level): {games_players_df.shape}")

# --- 2. Initial Data Structuring and Cleaning ---

# Rename columns for consistency and easier access
games_players_df.rename(columns={
    'match-datetime': 'match_datetime',
    'team1-score': 'game_team1_score', # Score for this specific game (map)
    'team2-score': 'game_team2_score', # Score for this specific game (map)
    'player-name': 'player_name',
    'player-team': 'player_team',
    'team1': 'match_team1_name', # Name of team1 in the match header
    'team2': 'match_team2_name', # Name of team2 in the match header
    'patch': 'game_patch', # Rename patch to avoid confusion later
    'map': 'game_map',
    'rating-t': 'rating_t', 'rating-ct': 'rating_ct',
    'acs-t': 'acs_t', 'acs-ct': 'acs_ct',
    'k-t': 'k_t', 'k-ct': 'k_ct',
    'd-t': 'd_t', 'd-ct': 'd_ct',
    'a-t': 'a_t', 'a-ct': 'a_ct',
    'tkmd-t': 'tkmd_t', 'tkmd-ct': 'tkmd_ct',
    'kast-t': 'kast_t', 'kast-ct': 'kast_ct',
    'adr-t': 'adr_t', 'adr-ct': 'adr_ct',
    'hs-t': 'hs_t', 'hs-ct': 'hs_ct',
    'fk-t': 'fk_t', 'fk-ct': 'fk_ct',
    'fd-t': 'fd_t', 'fd-ct': 'fd_ct',
    'fkmd-t': 'fkmd_t', 'fkmd-ct': 'fkmd_ct'
}, inplace=True)

# Convert match_datetime to datetime objects
games_players_df['match_datetime'] = pd.to_datetime(games_players_df['match_datetime'])

# Create a unique identifier for each *game* (map played within a match)
# Assuming match_datetime, team1, team2, and map are enough to identify a unique game within the raw data
games_players_df['game_identifier'] = games_players_df['match_datetime'].astype(str) + '_' + games_players_df['match_team1_name'] + '_vs_' + games_players_df['match_team2_name'] + '_' + games_players_df['game_map']

# Create a unique identifier for each *match* (the series of games)
# Assuming match_datetime, team1, and team2 are enough to identify a unique match
games_players_df['match_identifier'] = games_players_df['match_datetime'].astype(str) + '_' + games_players_df['match_team1_name'] + '_vs_' + games_players_df['match_team2_name']

# Handle potential missing values for numerical stats (fill with 0 is a common approach if missing means 0 action)
numeric_cols = games_players_df.select_dtypes(include=np.number).columns.tolist()
# Exclude the unnamed index column if it exists
if 'Unnamed: 0' in numeric_cols:
    numeric_cols.remove('Unnamed: 0')
games_players_df[numeric_cols] = games_players_df[numeric_cols].fillna(0)

print("Initial cleaning and structuring complete.")

# --- 3. Aggregate to Game Level and then Match Level ---

# Determine the winner of each game (map) based on scores
games_players_df['game_winner_match_team'] = games_players_df.apply(
    lambda row: row['match_team1_name'] if row['game_team1_score'] > row['game_team2_score'] else (row['match_team2_name'] if row['game_team2_score'] > row['game_team1_score'] else 'Draw'),
    axis=1
)

game_level_list = []
# Group by game_identifier first to process each game
for game_id, game_group in games_players_df.groupby('game_identifier'):
    if len(game_group) == 0: continue # Skip empty groups

    # Get match info from the first row of the game group (all rows in a game_group share this info)
    first_row = game_group.iloc[0]
    match_identifier = first_row['match_identifier']
    match_datetime = first_row['match_datetime']
    match_team1_name = first_row['match_team1_name']
    match_team2_name = first_row['match_team2_name']
    game_map = first_row['game_map']
    game_team1_score = first_row['game_team1_score']
    game_team2_score = first_row['game_team2_score']
    game_winner_match_team = first_row['game_winner_match_team']

    game_data_row = {
        'game_identifier': game_id,
        'match_identifier': match_identifier,
        'match_datetime': match_datetime,
        'match_team1_name': match_team1_name,
        'match_team2_name': match_team2_name,
        'game_map': game_map,
        'game_team1_score': game_team1_score,
        'game_team2_score': game_team2_score,
        'game_winner_match_team': game_winner_match_team
    }

    # Aggregate player stats for Team 1 (based on match header name) for this specific game
    team1_players_in_game = game_group[game_group['player_team'] == match_team1_name]
    game_data_row['team1_game_total_k'] = team1_players_in_game['k'].sum()
    game_data_row['team1_game_total_d'] = team1_players_in_game['d'].sum()
    game_data_row['team1_game_total_a'] = team1_players_in_game['a'].sum()
    game_data_row['team1_game_total_adr'] = team1_players_in_game['adr'].sum()
    game_data_row['team1_game_total_hs'] = team1_players_in_game['hs'].sum()
    game_data_row['team1_game_total_fk'] = team1_players_in_game['fk'].sum()
    game_data_row['team1_game_total_fd'] = team1_players_in_game['fd'].sum()
    game_data_row['team1_game_avg_rating'] = team1_players_in_game['rating'].mean()
    game_data_row['team1_game_avg_acs'] = team1_players_in_game['acs'].mean()
    game_data_row['team1_game_avg_kast'] = team1_players_in_game['kast'].mean() # Kast is already average per player, averaging per game team is fine.


    # Aggregate player stats for Team 2 (based on match header name) for this specific game
    team2_players_in_game = game_group[game_group['player_team'] == match_team2_name]
    game_data_row['team2_game_total_k'] = team2_players_in_game['k'].sum()
    game_data_row['team2_game_total_d'] = team2_players_in_game['d'].sum()
    game_data_row['team2_game_total_a'] = team2_players_in_game['a'].sum()
    game_data_row['team2_game_total_adr'] = team2_players_in_game['adr'].sum()
    game_data_row['team2_game_total_hs'] = team2_players_in_game['hs'].sum()
    game_data_row['team2_game_total_fk'] = team2_players_in_game['fk'].sum()
    game_data_row['team2_game_total_fd'] = team2_players_in_game['fd'].sum()
    game_data_row['team2_game_avg_rating'] = team2_players_in_game['rating'].mean()
    game_data_row['team2_game_avg_acs'] = team2_players_in_game['acs'].mean()
    game_data_row['team2_game_avg_kast'] = team2_players_in_game['kast'].mean()

    game_level_list.append(game_data_row)

game_level_data = pd.DataFrame(game_level_list)

# Ensure numeric types after aggregation
for col in game_level_data.columns:
    if col.startswith('team1_game_total_') or col.startswith('team2_game_total_') or col.startswith('team1_game_avg_') or col.startswith('team2_game_avg_'):
         game_level_data[col] = pd.to_numeric(game_level_data[col], errors='coerce').fillna(0)


print(f"\nAggregated game-level data shape: {game_level_data.shape}")
print(game_level_data.head())

# Aggregate games to match level to get match outcomes (map wins)

# Correct map win calculation using apply on the group
def count_map_wins(group):
    team1_name = group['match_team1_name'].iloc[0]
    team2_name = group['match_team2_name'].iloc[0]
    team1_wins = (group['game_winner_match_team'] == team1_name).sum()
    team2_wins = (group['game_winner_match_team'] == team2_name).sum()
    return pd.Series({
        'match_team1_name': team1_name,
        'match_team2_name': team2_name,
        'team1_maps_won': team1_wins,
        'team2_maps_won': team2_wins,
        'match_datetime': group['match_datetime'].iloc[0]
    })

match_outcomes = game_level_data.groupby('match_identifier').apply(count_map_wins).reset_index()


# Determine match winner
match_outcomes['match_winner_name'] = match_outcomes.apply(
    lambda row: row['match_team1_name'] if row['team1_maps_won'] > row['team2_maps_won'] else (row['match_team2_name'] if row['team2_maps_won'] > row['team1_maps_won'] else 'Draw'),
    axis=1
)

# Filter out draws if necessary (most tournaments don't have draws in playoffs)
match_outcomes = match_outcomes[match_outcomes['match_winner_name'] != 'Draw'].copy()


print(f"\nAggregated match-level outcomes shape: {match_outcomes.shape}")
print(match_outcomes.head())


# --- 4. Feature Engineering (Chronological Processing) ---

# Sort matches by date to process chronologically
match_outcomes = match_outcomes.sort_values(by='match_datetime').reset_index(drop=True)
game_level_data = game_level_data.sort_values(by='match_datetime').reset_index(drop=True)
games_players_df = games_players_df.sort_values(by='match_datetime').reset_index(drop=True)


# Initialize history storage (defaultdict provides default values for new keys)
team_history = defaultdict(lambda: {
    'matches_played': 0, 'matches_won': 0,
    'maps_played': 0, 'maps_won': 0,
    'games_played': 0, # Number of games played by the team over history (maps where they were team1/team2)
    'total_k': 0, 'total_d': 0, 'total_a': 0, 'total_adr': 0, 'total_fk': 0, 'total_fd': 0, 'total_hs': 0,
    'game_avg_rating_sum': 0, # Sum of game averages for rating
    'game_avg_acs_sum': 0,    # Sum of game averages for ACS
    'game_avg_kast_sum': 0,   # Sum of game averages for KAST
})

# Player histories (accumulated stats from all their games played)
player_history = defaultdict(lambda: {'games_played': 0, 'total_k': 0, 'total_d': 0, 'total_a': 0,
                                      'total_adr': 0, 'total_rating': 0, 'total_acs': 0,
                                      'total_fk': 0, 'total_fd': 0, 'total_hs': 0, 'total_kast': 0})

# Elo ratings (start with a base rating)
initial_elo = 1500
team_elo = defaultdict(lambda: initial_elo)


match_features_list = []

print("\nCalculating chronological features and updating history...")

# Loop through matches chronologically
for index, match in match_outcomes.iterrows():
    team1_name = match['match_team1_name']
    team2_name = match['match_team2_name']
    match_datetime = match['match_datetime']
    winner_name = match['match_winner_name']
    match_identifier = match['match_identifier']

    # --- Features for this match (calculated *before* processing its outcome) ---
    features = {
        'match_datetime': match_datetime,
        'team1_name': team1_name,
        'team2_name': team2_name,
        'match_winner_name': winner_name, # Target variable

        # Team History Features (Overall accumulated stats from games)
        'team1_overall_winrate_matches': team_history[team1_name]['matches_won'] / max(1, team_history[team1_name]['matches_played']),
        'team2_overall_winrate_matches': team_history[team2_name]['matches_won'] / max(1, team_history[team2_name]['matches_played']),
        'team1_overall_winrate_maps': team_history[team1_name]['maps_won'] / max(1, team_history[team1_name]['maps_played']),
        'team2_overall_winrate_maps': team_history[team2_name]['maps_won'] / max(1, team_history[team2_name]['maps_played']),

        # Team Average Stats (per game, accumulated over history) - calculated using sums / games_played
        'team1_overall_k_d': team_history[team1_name]['total_k'] / max(1, team_history[team1_name]['total_d']),
        'team2_overall_k_d': team_history[team2_name]['total_k'] / max(1, team_history[team2_name]['total_d']),
        'team1_overall_avg_adr': team_history[team1_name]['total_adr'] / max(1, team_history[team1_name]['games_played']),
        'team2_overall_avg_adr': team_history[team2_name]['total_adr'] / max(1, team_history[team2_name]['games_played']),
        'team1_overall_avg_rating': team_history[team1_name]['game_avg_rating_sum'] / max(1, team_history[team1_name]['games_played']),
        'team2_overall_avg_rating': team_history[team2_name]['game_avg_rating_sum'] / max(1, team_history[team2_name]['games_played']),
        'team1_overall_avg_acs': team_history[team1_name]['game_avg_acs_sum'] / max(1, team_history[team1_name]['games_played']),
        'team2_overall_avg_acs': team_history[team2_name]['game_avg_acs_sum'] / max(1, team_history[team2_name]['games_played']),
        'team1_overall_avg_kast': team_history[team1_name]['game_avg_kast_sum'] / max(1, team_history[team1_name]['games_played']),
        'team2_overall_avg_kast': team_history[team2_name]['game_avg_kast_sum'] / max(1, team_history[team2_name]['games_played']),


        # Elo Feature
        'team1_elo': team_elo[team1_name],
        'team2_elo': team_elo[team2_name],
        'elo_difference': team_elo[team1_name] - team_elo[team2_name],
    }
    # Head-to-Head Feature (matches)
    # Count past matches between team1 and team2
    past_h2h = match_outcomes.iloc[:index][ # Look at matches before the current one
        ((match_outcomes['match_team1_name'] == team1_name) & (match_outcomes['match_team2_name'] == team2_name)) |
        ((match_outcomes['match_team1_name'] == team2_name) & (match_outcomes['match_team2_name'] == team1_name))
    ]
    team1_h2h_wins = (past_h2h['match_winner_name'] == team1_name).sum()
    total_h2h_matches = len(past_h2h)

    # Calculate team1's winrate in H2H
    team1_h2h_winrate = team1_h2h_wins / max(1, total_h2h_matches)
    # Calculate the difference: Team1 H2H winrate - Team2 H2H winrate (Team2 H2H winrate is 1 - Team1 H2H winrate)
    h2h_winrate_diff = team1_h2h_winrate - (1 - team1_h2h_winrate) if total_h2h_matches > 0 else 0 # Handle case with no H2H

    features['team1_h2h_winrate'] = team1_h2h_winrate # Still keep this feature if desired
    features['h2h_winrate_diff'] = h2h_winrate_diff # Add the difference feature directly


    # Player Aggregated Stats Features (based on players' history *before* this match)
    # Find the specific games belonging to this match in the raw data
    current_match_raw_games = games_players_df[games_players_df['match_identifier'] == match_identifier]

    # Get the list of players on each team for this specific match
    team1_players_in_match = current_match_raw_games[current_match_raw_games['player_team'] == team1_name]['player_name'].unique()
    team2_players_in_match = current_match_raw_games[current_match_raw_games['player_team'] == match_team2_name]['player_name'].unique() # Make sure this correctly gets team2 players


    # Calculate accumulated stats for each team's *roster* based on their player history (total stats up to *before* this match)
    team1_roster_past_k = sum(player_history[p]['total_k'] for p in team1_players_in_match)
    team1_roster_past_d = sum(player_history[p]['total_d'] for p in team1_players_in_match)
    team1_roster_past_adr = sum(player_history[p]['total_adr'] for p in team1_players_in_match)
    team1_roster_past_rating_sum = sum(player_history[p]['total_rating'] for p in team1_players_in_match)
    team1_roster_past_acs_sum = sum(player_history[p]['total_acs'] for p in team1_players_in_match)
    team1_roster_past_games = sum(player_history[p]['games_played'] for p in team1_players_in_match) # Total games played by all players on roster

    team2_roster_past_k = sum(player_history[p]['total_k'] for p in team2_players_in_match)
    team2_roster_past_d = sum(player_history[p]['total_d'] for p in team2_players_in_match)
    team2_roster_past_adr = sum(player_history[p]['total_adr'] for p in team2_players_in_match)
    team2_roster_past_rating_sum = sum(player_history[p]['total_rating'] for p in team2_players_in_match)
    team2_roster_past_acs_sum = sum(player_history[p]['total_acs'] for p in team2_players_in_match)
    team2_roster_past_games = sum(player_history[p]['games_played'] for p in team2_players_in_match)

    # Roster features (average stats per game played by roster members)
    features['team1_roster_k_d'] = team1_roster_past_k / max(1, team1_roster_past_d)
    features['team2_roster_k_d'] = team2_roster_past_k / max(1, team2_roster_past_d)
    features['team1_roster_avg_adr'] = team1_roster_past_adr / max(1, team1_roster_past_games)
    features['team2_roster_avg_adr'] = team2_roster_past_adr / max(1, team2_roster_past_games)
    features['team1_roster_avg_rating'] = team1_roster_past_rating_sum / max(1, team1_roster_past_games)
    features['team2_roster_avg_rating'] = team2_roster_past_rating_sum / max(1, team2_roster_past_games)
    features['team1_roster_avg_acs'] = team1_roster_past_acs_sum / max(1, team1_roster_past_games)
    features['team2_roster_avg_acs'] = team2_roster_past_acs_sum / max(1, team2_roster_past_games)


    # Overall Map Winrate Feature (already calculated in team_history)
    features['team1_overall_map_winrate'] = team_history[team1_name]['maps_won'] / max(1, team_history[team1_name]['maps_played'])
    features['team2_overall_map_winrate'] = team_history[team2_name]['maps_won'] / max(1, team_history[team2_name]['maps_played'])

    # Append the features dictionary to the list
    match_features_list.append(features)

    # --- Update Histories (using this match's outcome and stats) ---

    # Update Team History (using stats from the games in this match)
    # Find the *game_level_data* rows for this match identifier
    current_match_game_level_stats = game_level_data[game_level_data['match_identifier'] == match_identifier]

    team_history[team1_name]['matches_played'] += 1
    team_history[team2_name]['matches_played'] += 1
    if winner_name == team1_name:
        team_history[team1_name]['matches_won'] += 1
    elif winner_name == team2_name:
        team_history[team2_name]['matches_won'] += 1

    team_history[team1_name]['maps_played'] += match['team1_maps_won'] + match['team2_maps_won'] # Total maps in this match
    team_history[team1_name]['maps_won'] += match['team1_maps_won'] # Use map wins from match_outcomes
    team_history[team2_name]['maps_played'] += match['team1_maps_won'] + match['team2_maps_won']
    team_history[team2_name]['maps_won'] += match['team2_maps_won'] # Use map wins from match_outcomes

    # Accumulate total stats from games in this match for overall team averages
    team_history[team1_name]['games_played'] += len(current_match_game_level_stats) # Add number of games played in this match
    team_history[team1_name]['total_k'] += current_match_game_level_stats['team1_game_total_k'].sum()
    team_history[team1_name]['total_d'] += current_match_game_level_stats['team1_game_total_d'].sum()
    team_history[team1_name]['total_a'] += current_match_game_level_stats['team1_game_total_a'].sum()
    team_history[team1_name]['total_adr'] += current_match_game_level_stats['team1_game_total_adr'].sum()
    team_history[team1_name]['total_hs'] += current_match_game_level_stats['team1_game_total_hs'].sum()
    team_history[team1_name]['total_fk'] += current_match_game_level_stats['team1_game_total_fk'].sum()
    team_history[team1_name]['total_fd'] += current_match_game_level_stats['team1_game_total_fd'].sum()
    team_history[team1_name]['game_avg_rating_sum'] += current_match_game_level_stats['team1_game_avg_rating'].sum()
    team_history[team1_name]['game_avg_acs_sum'] += current_match_game_level_stats['team1_game_avg_acs'].sum()
    team_history[team1_name]['game_avg_kast_sum'] += current_match_game_level_stats['team1_game_avg_kast'].sum()


    team_history[team2_name]['games_played'] += len(current_match_game_level_stats)
    team_history[team2_name]['total_k'] += current_match_game_level_stats['team2_game_total_k'].sum()
    team_history[team2_name]['total_d'] += current_match_game_level_stats['team2_game_total_d'].sum()
    team_history[team2_name]['total_a'] += current_match_game_level_stats['team2_game_total_a'].sum()
    team_history[team2_name]['total_adr'] += current_match_game_level_stats['team2_game_total_adr'].sum()
    team_history[team2_name]['total_hs'] += current_match_game_level_stats['team2_game_total_hs'].sum()
    team_history[team2_name]['total_fk'] += current_match_game_level_stats['team2_game_total_fk'].sum()
    team_history[team2_name]['total_fd'] += current_match_game_level_stats['team2_game_total_fd'].sum()
    team_history[team2_name]['game_avg_rating_sum'] += current_match_game_level_stats['team2_game_avg_rating'].sum()
    team_history[team2_name]['game_avg_acs_sum'] += current_match_game_level_stats['team2_game_avg_acs'].sum()
    team_history[team2_name]['game_avg_kast_sum'] += current_match_game_level_stats['team2_game_avg_kast'].sum()


    # Update Elo Ratings (Simplified Elo update)
    R1 = team_elo[team1_name]
    R2 = team_elo[team2_name]
    E1 = 1 / (1 + 10**((R2 - R1) / 400))
    E2 = 1 / (1 + 10**((R1 - R2) / 400))
    S1 = 1 if winner_name == team1_name else 0
    S2 = 1 if winner_name == team2_name else 0
    K = 30 # K-factor (can be tuned)
    team_elo[team1_name] += K * (S1 - E1)
    team_elo[team2_name] += K * (S2 - E2)


    # Update Player History (using stats from the games in this match)
    for player_index, player_game_stats in current_match_raw_games.iterrows():
         p_name = player_game_stats['player_name']

         player_history[p_name]['games_played'] += 1
         player_history[p_name]['total_k'] += player_game_stats['k']
         player_history[p_name]['total_d'] += player_game_stats['d']
         player_history[p_name]['total_a'] += player_game_stats['a']
         player_history[p_name]['total_adr'] += player_game_stats['adr']
         player_history[p_name]['total_rating'] += player_game_stats['rating']
         player_history[p_name]['total_acs'] += player_game_stats['acs']
         player_history[p_name]['total_fk'] += player_game_stats['fk']
         player_history[p_name]['total_fd'] += player_game_stats['fd']
         player_history[p_name]['total_hs'] += player_game_stats['hs']
         player_history[p_name]['total_kast'] += player_game_stats['kast']


# Convert the list of features to a DataFrame
model_data = pd.DataFrame(match_features_list)

# Fill NaN values created by initial calculations (e.g., winrate for teams with 0 matches)
# Fill with 0 for winrates/averages if the team/player had no previous games.
model_data.fillna(0, inplace=True)


# Add relative features (e.g., feature difference between team1 and team2)
model_data['winrate_matches_diff'] = model_data['team1_overall_winrate_matches'] - model_data['team2_overall_winrate_matches']
model_data['winrate_maps_diff'] = model_data['team1_overall_winrate_maps'] - model_data['team2_overall_winrate_maps']
model_data['roster_k_d_diff'] = model_data['team1_roster_k_d'] - model_data['team2_roster_k_d']
model_data['roster_avg_adr_diff'] = model_data['team1_roster_avg_adr'] - model_data['team2_roster_avg_adr']
model_data['roster_avg_rating_diff'] = model_data['team1_roster_avg_rating'] - model_data['team2_roster_avg_rating']
model_data['roster_avg_acs_diff'] = model_data['team1_roster_avg_acs'] - model_data['team2_roster_avg_acs']
model_data['overall_k_d_diff'] = model_data['team1_overall_k_d'] - model_data['team2_overall_k_d']
model_data['overall_avg_adr_diff'] = model_data['team1_overall_avg_adr'] - model_data['team2_overall_avg_adr']
model_data['overall_avg_rating_diff'] = model_data['team1_overall_avg_rating'] - model_data['team2_overall_avg_rating']
model_data['overall_avg_acs_diff'] = model_data['team1_overall_avg_acs'] - model_data['team2_overall_avg_acs']
model_data['overall_avg_kast_diff'] = model_data['team1_overall_avg_kast'] - model_data['team2_overall_avg_kast']
# h2h_winrate_diff was already calculated directly within the loop and added to 'features'
model_data['overall_map_winrate_diff'] = model_data['team1_overall_map_winrate'] - model_data['team2_overall_map_winrate']


# Drop the individual team features used to create difference features, keep difference and Elo
# Ensure 'team1_h2h_winrate' is dropped if you only want the 'h2h_winrate_diff'
# Adjust features_to_keep based on which individual team features you want to keep vs difference
features_to_keep = [col for col in model_data.columns if 'diff' in col or 'elo' in col or 'team1_overall_' in col or 'team2_overall_' in col or 'team1_roster_' in col or 'team2_roster_' in col]
# Add the team names and winner back temporarily for splitting
features_to_keep.extend(['match_datetime', 'team1_name', 'team2_name', 'match_winner_name'])

# Select only the desired columns for the final model data
model_data_final = model_data[features_to_keep].copy()

# Drop redundant individual H2H winrate if difference is preferred
if 'team1_h2h_winrate' in model_data_final.columns and 'h2h_winrate_diff' in model_data_final.columns:
     model_data_final = model_data_final.drop('team1_h2h_winrate', axis=1)


print("\nFinal model data with difference features.")
print(model_data_final.head())
print(f"Final model data shape: {model_data_final.shape}")


# --- 5. Data Splitting (Chronological) ---

# Define target: 1 if team1 wins, 0 if team2 wins
model_data_final['target'] = (model_data_final['match_winner_name'] == model_data_final['team1_name']).astype(int)

# Drop original winner column, team names, and datetime
X = model_data_final.drop(['match_winner_name', 'team1_name', 'team2_name', 'match_datetime', 'target'], axis=1)
y = model_data_final['target']

# Identify features for scaling - all remaining are numerical
feature_cols = X.columns.tolist()


# Split data chronologically
# Use data up to the end of 2023 for training, 2024 data for testing
split_date = '2024-01-01'

# Find the index where the date changes from 2023 to 2024
# Assuming your data is already sorted by datetime from the chronological loop
split_index = model_data_final[model_data_final['match_datetime'] >= split_date].index.min()

if pd.isna(split_index) or split_index == 0: # If no data in 2024 or later, or all data is in 2024+
    print(f"\nWarning: Cannot create a chronological test set with data after {split_date}.")
    print("Splitting data randomly (20% test set). This is less ideal for time-series data.")
    # Use all data for random split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Use stratify to maintain winner distribution
else:
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    print(f"\nData split chronologically (before {split_date} for train, after for test):")

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- 6. Select and Train Model ---

# Define the preprocessing steps (scaling)
# We use StandardScaler because most ML models benefit from having features on a similar scale
preprocessor = StandardScaler()

# Create the full pipeline including preprocessing and the classifier
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(random_state=42, solver='liblinear'))]) # Start with Logistic Regression

# --- You can try other models here ---
# model = Pipeline(steps=[('preprocessor', preprocessor),
#                         ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

# model = Pipeline(steps=[('preprocessor', preprocessor),
#                         ('classifier', xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss'))])

# model = Pipeline(steps=[('preprocessor', preprocessor),
#                         ('classifier', lgb.LGBMClassifier(objective='binary', n_estimators=100, learning_rate=0.1, random_state=42))])


print(f"\nTraining {type(model.named_steps['classifier']).__name__} model...")
model.fit(X_train, y_train)
print("Training complete.")

# --- 7. Evaluate Model ---
print("\nEvaluating model on test data...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (Team 1 winning)

# Evaluate using various metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}") # AUC is very important for probability models

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 8. Make Predictions for Masters Toronto 2025 (Conceptual) ---

# To predict for Masters Toronto 2025, you would need:
# 1. The list of teams participating in that tournament (e.g., ['Team Alpha', 'Team Beta', ...]).
# 2. The roster for each team at the time of the tournament (important for roster features).
# 3. The match schedule (who plays whom).

# You would then use the team_history, player_history, and team_elo objects
# calculated using ALL the data up to the end of your latest data (e.g., end of 2024 data from the file).

# Example function to prepare features for a future match
def prepare_future_match_features(team1_name, team2_name, current_team_history, current_player_history, current_team_elo, team_rosters, full_match_outcomes_df, training_feature_columns):
    """
    Pulls the final historical data and calculates features for a future match.

    Args:
        team1_name (str): Name of the first team in the match.
        team2_name (str): Name of the second team in the match.
        current_team_history (defaultdict): The team_history dict calculated up to the end of your data.
        current_player_history (defaultdict): The player_history dict calculated up to the end of your data.
        current_team_elo (defaultdict): The team_elo dict calculated up to the end of your data.
        team_rosters (dict): A dictionary mapping team names to their predicted/known roster
                             for the future tournament, e.g., {'Team Alpha': ['PlayerA', 'PlayerB', ...], ...}.
                             Only players in the roster are used for roster features.
        full_match_outcomes_df (pd.DataFrame): The DataFrame containing all match outcomes. Used for H2H calculation.
        training_feature_columns (pd.Index): The columns of the training features (X_train.columns)
                                               to ensure the output DataFrame has the correct structure.

    Returns:
        pandas.DataFrame: A DataFrame with one row containing the features for the future match.
                          Returns None if rosters are missing or teams have no history in the provided data.
    """
    # Get roster for future match
    team1_roster = team_rosters.get(team1_name)
    team2_roster = team_rosters.get(team2_name)

    if not team1_roster or not team2_roster:
        print(f"Roster information missing for {team1_name} or {team2_name}.")
        return None

    # Check if teams have any history in the provided data
    if current_team_history[team1_name]['matches_played'] == 0 or current_team_history[team2_name]['matches_played'] == 0:
         print(f"Warning: One or both teams ({team1_name}, {team2_name}) have no prior history in the data.")
         # Assign baseline features for new teams (e.g., average winrate, initial Elo, 0 roster stats)
         # This is handled by the fillna(0) later, but could be more sophisticated.
         # Continue the process, they will get default 0 values from defaultdict
         pass


    # Calculate features using the final history states
    features = {}

    # Team History Features
    features['team1_overall_winrate_matches'] = current_team_history[team1_name]['matches_won'] / max(1, current_team_history[team1_name]['matches_played'])
    features['team2_overall_winrate_matches'] = current_team_history[team2_name]['matches_won'] / max(1, current_team_history[team2_name]['matches_played'])
    features['team1_overall_winrate_maps'] = current_team_history[team1_name]['maps_won'] / max(1, current_team_history[team1_name]['maps_played'])
    features['team2_overall_winrate_maps'] = current_team_history[team2_name]['maps_won'] / max(1, current_team_history[team2_name]['maps_played'])

    # Team Average Stats
    features['team1_overall_k_d'] = current_team_history[team1_name]['total_k'] / max(1, current_team_history[team1_name]['total_d'])
    features['team2_overall_k_d'] = current_team_history[team2_name]['total_k'] / max(1, current_team_history[team2_name]['total_d'])
    features['team1_overall_avg_adr'] = current_team_history[team1_name]['total_adr'] / max(1, current_team_history[team1_name]['games_played'])
    features['team2_overall_avg_adr'] = current_team_history[team2_name]['total_adr'] / max(1, current_team_history[team2_name]['games_played'])
    features['team1_overall_avg_rating'] = current_team_history[team1_name]['game_avg_rating_sum'] / max(1, current_team_history[team1_name]['games_played'])
    features['team2_overall_avg_rating'] = current_team_history[team2_name]['game_avg_rating_sum'] / max(1, current_team_history[team2_name]['games_played'])
    features['team1_overall_avg_acs'] = current_team_history[team1_name]['game_avg_acs_sum'] / max(1, current_team_history[team1_name]['games_played'])
    features['team2_overall_avg_acs'] = current_team_history[team2_name]['game_avg_acs_sum'] / max(1, current_team_history[team2_name]['games_played'])
    features['team1_overall_avg_kast'] = current_team_history[team1_name]['game_avg_kast_sum'] / max(1, current_team_history[team1_name]['games_played'])
    features['team2_overall_avg_kast'] = current_team_history[team2_name]['game_avg_kast_sum'] / max(1, current_team_history[team2_name]['games_played'])


    # Elo Feature
    features['team1_elo'] = current_team_elo[team1_name]
    features['team2_elo'] = current_team_elo[team2_name]
    features['elo_difference'] = current_team_elo[team1_name] - current_team_elo[team2_name]

    # Head-to-Head Feature (Recalculate H2H using all historical data)
    past_h2h_all_data = full_match_outcomes_df[ # Use the full match_outcomes DataFrame computed earlier
        ((full_match_outcomes_df['match_team1_name'] == team1_name) & (full_match_outcomes_df['match_team2_name'] == team2_name)) |
        ((full_match_outcomes_df['match_team1_name'] == team2_name) & (full_match_outcomes_df['match_team2_name'] == team1_name))
    ]
    team1_h2h_wins = (past_h2h_all_data['match_winner_name'] == team1_name).sum()
    total_h2h_matches = len(past_h2h_all_data)

    # Calculate team1's winrate in H2H
    team1_h2h_winrate = team1_h2h_wins / max(1, total_h2h_matches)
    # Calculate the difference: Team1 H2H winrate - Team2 H2H winrate (Team2 H2H winrate is 1 - Team1 H2H winrate if no draws)
    h2h_winrate_diff = team1_h2h_winrate - (total_h2h_matches - team1_h2h_wins) / max(1, total_h2h_matches)


    features['team1_h2h_winrate'] = team1_h2h_winrate # Still keep this feature if desired
    features['h2h_winrate_diff'] = h2h_winrate_diff # Add the difference feature


    # Player Aggregated Stats Features for the roster
    # Only use players from the provided roster who actually exist in the player_history (have played games in the data)
    team1_roster_players_with_history = [p for p in team1_roster if p in current_player_history]
    team2_roster_players_with_history = [p for p in team2_roster if p in current_player_history]


    # Calculate accumulated stats for each team's *roster* based on their player history (total stats up to the end of data)
    team1_roster_past_k = sum(current_player_history[p]['total_k'] for p in team1_roster_players_with_history)
    team1_roster_past_d = sum(current_player_history[p]['total_d'] for p in team1_roster_players_with_history)
    team1_roster_past_adr = sum(current_player_history[p]['total_adr'] for p in team1_roster_players_with_history)
    team1_roster_past_rating_sum = sum(current_player_history[p]['total_rating'] for p in team1_roster_players_with_history)
    team1_roster_past_acs_sum = sum(current_player_history[p]['total_acs'] for p in team1_roster_players_with_history)
    team1_roster_past_games = sum(current_player_history[p]['games_played'] for p in team1_roster_players_with_history) # Total games played by all players on roster

    team2_roster_past_k = sum(current_player_history[p]['total_k'] for p in team2_roster_players_with_history)
    team2_roster_past_d = sum(current_player_history[p]['total_d'] for p in team2_roster_players_with_history)
    team2_roster_past_adr = sum(current_player_history[p]['total_adr'] for p in team2_roster_players_with_history)
    team2_roster_past_rating_sum = sum(current_player_history[p]['total_rating'] for p in team2_roster_players_with_history)
    team2_roster_past_acs_sum = sum(current_player_history[p]['total_acs'] for p in team2_roster_players_with_history)
    team2_roster_past_games = sum(current_player_history[p]['games_played'] for p in team2_roster_players_with_history)


    # Roster features (average stats per game played by roster members)
    features['team1_roster_k_d'] = team1_roster_past_k / max(1, team1_roster_past_d)
    features['team2_roster_k_d'] = team2_roster_past_k / max(1, team2_roster_past_d)
    features['team1_roster_avg_adr'] = team1_roster_past_adr / max(1, team1_roster_past_games)
    features['team2_roster_avg_adr'] = team2_roster_past_adr / max(1, team2_roster_past_games)
    features['team1_roster_avg_rating'] = team1_roster_past_rating_sum / max(1, team1_roster_past_games)
    features['team2_roster_avg_rating'] = team2_roster_past_rating_sum / max(1, team2_roster_past_games)
    features['team1_roster_avg_acs'] = team1_roster_past_acs_sum / max(1, team1_roster_past_games)
    features['team2_roster_avg_acs'] = team2_roster_past_acs_sum / max(1, team2_roster_past_games)


    # Overall Map Winrate Feature (already calculated in team_history)
    features['team1_overall_map_winrate'] = current_team_history[team1_name]['maps_won'] / max(1, current_team_history[team1_name]['maps_played'])
    features['team2_overall_map_winrate'] = current_team_history[team2_name]['maps_won'] / max(1, current_team_history[team2_name]['maps_played'])

    # Difference features (calculated from individual team/roster features)
    features['winrate_matches_diff'] = features['team1_overall_winrate_matches'] - features['team2_overall_winrate_matches']
    features['winrate_maps_diff'] = features['team1_overall_winrate_maps'] - features['team2_overall_map_winrate'] # Corrected this again to use team2 overall map winrate
    features['roster_k_d_diff'] = features['team1_roster_k_d'] - features['team2_roster_k_d']
    features['roster_avg_adr_diff'] = features['team1_roster_avg_adr'] - features['team2_roster_avg_adr']
    features['roster_avg_rating_diff'] = features['team1_roster_avg_rating'] - features['team2_roster_avg_rating']
    features['roster_avg_acs_diff'] = features['team1_roster_avg_acs'] - features['team2_roster_avg_acs']
    features['overall_k_d_diff'] = features['team1_overall_k_d'] - features['team2_overall_k_d']
    features['overall_avg_adr_diff'] = features['team1_overall_avg_adr'] - features['team2_overall_avg_adr']
    features['overall_avg_rating_diff'] = features['team1_overall_avg_rating'] - features['team2_overall_avg_rating']
    features['overall_avg_acs_diff'] = features['team1_overall_avg_acs'] - features['team2_overall_avg_acs']
    features['overall_avg_kast_diff'] = features['team1_overall_avg_kast'] - features['team2_overall_avg_kast']
    # h2h_winrate_diff was already calculated directly above and added to 'features'
    features['overall_map_winrate_diff'] = features['team1_overall_map_winrate'] - features['team2_overall_map_winrate']


    # Create DataFrame, ensuring columns match X_train.columns
    # Create a dictionary with the keys matching the feature names in training_feature_columns
    feature_vector_dict = {col: features.get(col, 0) for col in training_feature_columns} # Get features by name, default to 0 if missing

    future_match_df = pd.DataFrame([feature_vector_dict])

    # Handle potential NaNs (should be filled with 0 from defaultdict, but double check)
    future_match_df.fillna(0, inplace=True) # Fill with 0 as done for training data

    return future_match_df




# --- Use the trained model to predict the G2 vs Paper Rex match ---

# Define the teams and their rosters for this specific match (from the image)
team1_name_pred = 'G2 Esports'
team2_name_pred = 'Paper Rex'

# Rosters from the image (use the player names exactly as they appear in your data)
# You might need to verify player names in your combined_file if these don't match exactly.
prediction_rosters = {
    team1_name_pred: ['trent', 'valyn', 'JonahP', 'leaf', 'jawgemo'],
    team2_name_pred: ['mindfreak', 'f0rsakeN', 'd4v41', 'Jinggg', 'something'],
}

# Prepare the features for this specific future match
# Pass the necessary historical data objects and the full match_outcomes DataFrame
# Pass X_train.columns to ensure the output DataFrame structure is correct
future_match_features_df = prepare_future_match_features(
    team1_name_pred,
    team2_name_pred,
    team_history,          # History calculated over all your data
    player_history,        # Player history calculated over all your data
    team_elo,              # Final Elo ratings
    prediction_rosters,    # The rosters for this specific match
    match_outcomes,        # The full match_outcomes DataFrame for H2H calculation
    X_train.columns        # The columns from your training feature DataFrame
)

if future_match_features_df is not None:
    # Use the trained model to predict the probability
    # model.predict_proba returns probabilities for each class [prob_class_0, prob_class_1]
    # We want the probability of Team 1 winning (G2 Esports), which is class 1 (index 1)
    future_pred_proba = model.predict_proba(future_match_features_df)[:, 1]

    print(f"\n--- Prediction for {team1_name_pred} vs {team2_name_pred} ---")
    print(f"Predicted probability of {team1_name_pred} winning: {future_pred_proba[0]:.4f}")
    print(f"Predicted probability of {team2_name_pred} winning: {1 - future_pred_proba[0]:.4f}")

    # To get a simple winner prediction (0 or 1):
    # future_prediction = model.predict(future_match_features_df)
    # predicted_winner_name = team1_name_pred if future_prediction[0] == 1 else team2_name_pred
    # print(f"Predicted Winner: {predicted_winner_name}")

else:
    print(f"\nCould not prepare features for {team1_name_pred} vs {team2_name_pred}.")


print("\nPrediction script finished.")