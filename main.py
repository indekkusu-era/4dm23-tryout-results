import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.stats import t

from utils import maximize_tournament_score, logit, inverse_logit, weights

n_players = 6
n_players_team = 3

df = pd.read_csv('dataset/tryout-scores.csv')

def rename_columns(df: pd.DataFrame):
    rename_cols = {
        'Timestamp': 'timestamp',
        'Username (Ex. HowToPlayLN)': 'username',
        'Map': 'beatmap',
        'Score': 'score',
        'Replay': 'replay',
        'Screenshot of Local Ranking': 'screenshot'
    }
    return df.rename(rename_cols, axis=1)

def rename_players(df: pd.DataFrame):
    df['username'] = df['username'].apply(lambda x: x.replace("-", "\\-"))
    return df

def clean_score_columns(df: pd.DataFrame):
    df['score'] = df['score'].apply(lambda x: int(x.replace(",", "")))
    return df

def pivot_max_score(df: pd.DataFrame):
    score_df = df.groupby(['username', 'beatmap'])['score'].max().reset_index()
    return score_df.pivot(index='username', columns='beatmap', values='score')

def standardize(df: pd.DataFrame):
    standardized = StandardScaler().fit_transform(df)
    return pd.DataFrame(standardized, index=df.index, columns=df.columns)

def impute_raw_scores(df: pd.DataFrame):
    df = df.fillna(5e5)
    return df

def get_rating(df: pd.DataFrame):
    df = logit(df)
    return standardize(df)
    
def preprocess(df: pd.DataFrame, rating=False):
    df = rename_columns(df)
    df = rename_players(df)
    df = clean_score_columns(df)
    table_df = pivot_max_score(df)
    table_df = impute_raw_scores(table_df)
    if rating:
        return get_rating(table_df)
    return table_df

def get_rosters(player_list: np.ndarray, opt_res: np.ndarray):
    return player_list[opt_res == 1]

def calculate_scores(df: pd.DataFrame, player_list: np.ndarray, n_players_team: int):
    candidate_scores = df.loc[player_list]
    vals = candidate_scores.values
    scores = np.sort(vals.T, axis=1)[:, ::-1][:, :n_players_team].mean(axis=1)
    beatmaps = df.columns
    return pd.DataFrame(scores, index=beatmaps, columns=['score'])

class Dashboard:
    def __init__(self, df: pd.DataFrame, n_players: int, n_players_team: int):
        self._df = df
        self._n_players = n_players
        self._n_players_team = n_players_team
    
    def _ci(self, mean, std, n, alpha=0.05):
        df = n-1
        stderr = std / np.sqrt(n)
        t_stat = abs(t.ppf(alpha / 2, df))
        return mean - t_stat * stderr, mean + t_stat * stderr
    
    def get_raw_scores(self):
        return preprocess(self._df, rating=False)
    
    def get_normalized_scores(self):
        return preprocess(self._df, rating=True)

    def score_distribution(self, _logit: bool):
        """
        Score distribution of each map, including boxplot, histogram

        average, standard deviation, confidence interval via t distribution
        """
        st.subheader("Statistics")
        st.text("This is the statistics of each beatmap")
        if _logit:
            process_df = logit(self.get_raw_scores())
        else:
            process_df = self.get_raw_scores().copy()
        beatmap_option = st.selectbox("Select a beatmap", process_df.columns)
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        score_distribution = process_df[beatmap_option].values
        ax.hist(score_distribution)
        ax2.boxplot(score_distribution)
        st.pyplot(fig)
        st.pyplot(fig2)
        if _logit:
            average_logit = score_distribution.mean()
            average_score = inverse_logit(average_logit)
            std_score = score_distribution.std(ddof=1)
            n_samples = len(score_distribution)
            lower, upper = self._ci(average_logit, std_score, n_samples)
            lower = inverse_logit(lower); upper = inverse_logit(upper)
        else:
            average_score = score_distribution.mean()
            std_score = score_distribution.std(ddof=1)
            n_samples = len(score_distribution)
            lower, upper = self._ci(average_score, std_score, n_samples)

        st.text(f"Average Score: {int(average_score)}")
        st.text(f"95% Lower Bound: {int(lower)}")
        st.text(f"95% Upper Bound: {int(upper)}")
    
    def select_players(self, _logit: bool):
        """
        Interactive, let users select the players and calculate the scores
        """
        st.subheader("Players Selection")
        st.text("You will be able to try to select players and try to maximize the average score on this section")
        if _logit:
            process_df = logit(self.get_raw_scores())
        else:
            process_df = self.get_raw_scores().copy()
        all_players_list = process_df.index
        players_selected = [st.checkbox(player) for player in all_players_list]
        player_list = []
        for player, selected in zip(all_players_list, players_selected):
            if selected:
                player_list.append(player)
        if player_list:
            scores = calculate_scores(process_df, player_list, self._n_players_team)
            if _logit:
                scores = inverse_logit(scores)
            st.dataframe(scores.astype(int))
    
    def results(self):
        """
        Display the results and scores obtained by results
        """
        st.subheader("And the Thailand Team rosters are...")
        process_df = self.get_normalized_scores()
        res = maximize_tournament_score(process_df.values, list(weights.values()), n_players)
        rosters = get_rosters(process_df.index, res.x)
        st.markdown("\n".join([f"- {player}" for player in rosters]))
        
    
    def render(self):
        st.title("4dm2023 Thailand Team Tryout Results")
        self._toggle_logit = st.checkbox('Toggle Logit mode')
        self.score_distribution(self._toggle_logit)
        self.select_players(self._toggle_logit)
        self.results()

def main():
    dashboard = Dashboard(df, n_players, n_players_team)
    dashboard.render()

if __name__ == "__main__":
    main()
