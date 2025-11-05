import streamlit as st
import pandas as pd
import numpy as np
from statsbombpy import sb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Bristol Rovers Player Profiling", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

COLORS = {'primary': '#0066CC', 'secondary': '#003366', 'accent': '#00AAFF', 'background': '#F8FAFC', 'text': '#1E293B', 'border': '#E2E8F0', 'green': '#10B981', 'grey': '#6B7280', 'red': '#EF4444', 'amber': '#F59E0B'}

st.markdown(f"""<style>.main {{background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);}}.stApp {{background: transparent;}}h1, h2, h3 {{color: white !important;}}.stSelectbox label {{color: white !important;}}</style>""", unsafe_allow_html=True)

CREDS = {"user": "dhillon.gil@bristolrovers.co.uk", "passwd": "004laVPb"}

STATSBOMB_FIELDS = {
    'player_name': 'Name',
    'team_name': 'Team',
    'season_name': 'Season',
    'competition_name': 'Competition',
    'player_season_minutes': 'Minutes',
    'primary_position': 'Primary Position',
    'birth_date': 'Birth Date',
    'player_season_aerial_ratio': 'Aerial Win%',
    'player_season_ball_recoveries_90': 'Ball Recoveries',
    'player_season_blocks_per_shot': 'Blocks/Shot',
    'player_season_carries_90': 'Carry Length',
    'player_season_crossing_ratio': 'Successful Crosses',
    'player_season_deep_progressions_90': 'Deep Progressions',
    'player_season_defensive_action_regains_90': 'Defensive Regains',
    'player_season_defensive_actions_90': 'Defensive Actions',
    'player_season_dribble_faced_ratio': 'Dribbles Stopped%',
    'player_season_dribble_ratio': 'Dribble%',
    'player_season_dribbles_90': 'Dribbles',
    'player_season_dribbles_90': 'Successful Dribbles',
    'player_season_np_shots_90': 'Shots',
    'player_season_np_xg_90': 'xG',
    'player_season_np_xg_per_shot': 'xG/Shot',
    'player_season_npg_90': 'NP Goals',
    'player_season_npxgxa_90': 'xG Assisted',
    'player_season_obv_90': 'OBV',
    'player_season_obv_defensive_action_90': 'DA OBV',
    'player_season_obv_dribble_carry_90': 'D&C OBV',
    'player_season_obv_pass_90': 'Pass OBV',
    'player_season_obv_shot_90': 'Shot OBV',
    'player_season_op_f3_passes_90': 'OP F3 Passes',
    'player_season_op_key_passes_90': 'OP Key Passes',
    'player_season_op_passes_into_and_touches_inside_box_90': 'PINTIN',
    'player_season_op_passes_into_box_90': 'OP Passes Into Box',
    'player_season_padj_clearances_90': 'PAdj Clearances',
    'player_season_padj_clearances_90': 'Clearances',
    'player_season_padj_interceptions_90': 'PAdj Interceptions',
    'player_season_padj_pressures_90': 'PAdj Pressures',
    'player_season_padj_tackles_90': 'PAdj Tackles',
    'player_season_passing_ratio': 'Passing%',
    'player_season_shot_on_target_ratio': 'Shooting%',
    'player_season_shot_on_target_ratio': 'Shot%',
    'player_season_shot_touch_ratio': 'Shot Touch %',
    'player_season_touches_inside_box_90': 'Touches In Box',
    'player_season_touches_inside_box_ratio': 'Touches In Box%',
    'player_season_xgbuildup_90': 'xGBuildup',
    'player_season_op_xa_90': 'OP XG ASSISTED',
    'player_season_pressured_passing_ratio': 'Pr. Pass%',
    'player_season_da_aggressive_distance': 'GK Aggressive Dist.',
    'player_season_clcaa': 'Claims%',
    'player_season_gsaa_ratio': 'xSv%',
    'player_season_gsaa_90': 'GSAA',
    'player_season_save_ratio': 'Save%',
    'player_season_xs_ratio': 'XSv%',
    'player_season_positive_outcome_score': 'Positive Outcome',
    'player_season_obv_gk_90': 'GK OBV',
    'player_season_forward_pass_ratio': 'Pass Forward%',
    'player_season_scoring_contribution_90': 'Scoring Contribution',
    'player_season_fouls_won_90': 'Fouls Won',
    'player_season_pressures_90': 'Pressures',
    'player_season_counterpressures_90': 'Counterpressures',
    'player_season_aggressive_actions_90': 'Aggressive Actions'
}

POSITION_MAPPING = {'Right Wing Back': 'FB', 'Left Wing Back': 'FB', 'Right Back': 'FB', 'Left Back': 'FB', 'Right Centre Back': 'CB', 'Left Centre Back': 'CB', 'Centre Back': 'CB', 'Left Defensive Midfielder': 'DM', 'Right Defensive Midfielder': 'DM', 'Defensive Midfielder': 'DM', 'Left Centre Midfielder': 'CM', 'Right Centre Midfielder': 'CM', 'Centre Midfielder': 'CM', 'Attacking Midfield': 'AM', 'Left Attacking Midfield': 'AM', 'Right Attacking Midfield': 'AM', 'Left Centre Forward': 'CF', 'Right Centre Forward': 'CF', 'Centre Forward': 'CF', 'Right Wing': 'WF', 'Left Wing': 'WF', 'Right Midfielder': 'WF', 'Left Midfielder': 'WF', 'Goalkeeper': 'GK'}

PROFILE_WEIGHTS = {'GK': {'Ball Playing GK': {'Passing%': 0.14, 'Pr. Pass%': 0.14, 'xGBuildup': 0.14}, 'Sweeper Keeper': {'Pr. Pass%': 0.12, 'GK Aggressive Dist.': 0.24, 'OP Key Passes': 0.12}, 'Shot Stopper': {'GSAA': 0.20, 'xSv%': 0.20}}, 'CB': {'Wide CB': {'Dribbles': 0.12, 'Dribbles Stopped%': 0.18, 'Aerial Win%': 0.12, 'Carry Length': 0.12, 'Successful Crosses': 0.12}, 'Ball Playing CB': {'Pass OBV': 0.12, 'Passing%': 0.12, 'Pr. Pass%': 0.14}, 'Box Defender': {'PAdj Clearances': 0.16, 'Aerial Win%': 0.14, 'Ball Recoveries': 0.08, 'DA OBV': 0.16}}, 'FB': {'Attacking WB': {'xG Assisted': 0.18, 'Successful Crosses': 0.16, 'Carry Length': 0.12, 'D&C OBV': 0.18}, 'Defensive FB': {'PAdj Interceptions': 0.14, 'PAdj Clearances': 0.10, 'Dribbles Stopped%': 0.18, 'Aerial Win%': 0.12, 'DA OBV': 0.12}}, 'DM': {'Ball Playing 6': {'Passing%': 0.20, 'Pr. Pass%': 0.18, 'xGBuildup': 0.16}, 'Ball Winning DM': {'PAdj Clearances': 0.16, 'Aerial Win%': 0.16, 'Ball Recoveries': 0.10, 'DA OBV': 0.20}}, 'CM': {'Box to Box CM': {'PINTIN': 0.10, 'Aerial Win%': 0.10, 'Carry Length': 0.10, 'PAdj Interceptions': 0.12, 'D&C OBV': 0.16, 'DA OBV': 0.14}, 'Playmaking CM': {'OP Key Passes': 0.12, 'Deep Progressions%': 0.12, 'xGBuildup': 0.14}, 'Box Crashing CM': {'Shots': 0.20, 'Touches In Box': 0.20, 'Aerial Win%': 0.20}, 'Pressing CM': {'Aggressive Actions': 0.14, 'Counterpressures': 0.14, 'Pressures': 0.14}}, 'AM': {'Goal Threat AM': {'xG': 0.16, 'Shots': 0.14, 'Scoring Contribution': 0.16, 'Shooting%': 0.08, 'Carry Length': 0.08}, 'Creative AM': {'xG Assisted': 0.22, 'OP Key Passes': 0.18, 'OP Passes Into Box': 0.14}}, 'WF': {'1v1 Winger': {'Successful Dribbles': 0.20, 'Dribble%': 0.16, 'Successful Crosses': 0.16, 'Carry Length': 0.14, 'D&C OBV': 0.18}, 'Creative Winger': {'xG Assisted': 0.25, 'OP Key Passes': 0.20, 'Passes Into box': 0.20}, 'Wide Forward': {'xG': 0.16, 'Shots': 0.16, 'Scoring Contribution': 0.14, 'Touches In Box': 0.10}}, 'CF': {'Target Man': {'Aerial Win%': 0.14, 'Clearances': 0.12, 'Shots': 0.10, 'Touches In Box%': 0.12}, 'Pressing Forward': {'Counterpressures': 0.18, 'Aggressive Actions': 0.14, 'Pressures': 0.16}}}

ATTRIBUTES_BY_POSITION = {'GK': {'Passing': ['Passing%', 'Pr. Pass%'], 'Cross Claiming': ['Claims%'], 'Sweeping': ['GK Aggressive Dist.'], 'Shot Stopping': ['GSAA', 'xSv%']}, 'CB': {'Carrying': ['Carry Length', 'D&C OBV'], 'Passing': ['Passing%'], '1v1 Defending': ['Aggressive Actions', 'Dribbles Stopped%'], 'Box Defending': ['PAdj Clearances', 'Aerial Win%'], 'OP Defending': ['PAdj Interceptions', 'Ball Recoveries', 'DA OBV']}, 'FB': {'Creation': ['xG Assisted', 'Successful Crosses', 'OP Key Passes'], 'Dribbling/Carrying': ['Successful Dribbles', 'D&C OBV', 'Carry Length'], 'Passing': ['Passing%'], '1v1 Defending': ['Dribbles Stopped%'], 'Box Defending': ['PAdj Clearances', 'Aerial Win%'], 'OP Defending': ['PAdj Interceptions', 'Ball Recoveries', 'DA OBV']}, 'DM': {'Ball Retention': ['Passing%', 'Pr. Pass%'], 'Ball Progression': ['Deep Progressions'], 'Carrying': ['Carry Length', 'D&C OBV'], 'OP Defending': ['Ball Recoveries'], 'Box Defending': ['PAdj Clearances', 'Aerial Win%'], 'Pressing': ['Pressures', 'Counterpressures'], 'Creativity': ['xG Assisted', 'OP Key Passes']}, 'CM': {'Ball Retention': ['Passing%', 'Pr. Pass%'], 'Ball Progression': ['Deep Progressions', 'xGBuildup'], 'Carrying': ['Carry Length', 'D&C OBV'], 'Defensive Play': ['PAdj Clearances', 'Aerial Win%'], 'Goal Threat': ['xG', 'Shots', 'Touches In Box'], 'Pressing': ['Pressures', 'Counterpressures'], 'Creativity': ['xG Assisted', 'OP Key Passes', 'Successful Crosses']}, 'AM': {'Passing': ['Passing%', 'Pr. Pass%'], 'Carrying': ['Carry Length', 'D&C OBV', 'Successful Dribbles'], 'Aerial Ability': ['Aerial Win%'], 'Goal Threat': ['xG', 'Shots', 'Scoring Contribution'], 'Pressing': ['Pressures', 'Counterpressures'], 'Creativity': ['xG Assisted', 'OP Key Passes', 'Successful Crosses']}, 'WF': {'Passing': ['Passing%'], 'Dribbling': ['Dribble%', 'Successful Dribbles'], 'Carrying': ['Carry Length', 'D&C OBV'], 'Aerial Ability': ['Aerial Win%'], 'Goal Threat': ['xG', 'Shots', 'Scoring Contribution'], 'Pressing': ['Pressures', 'Counterpressures'], 'Creativity': ['xG Assisted', 'OP Key Passes', 'Successful Crosses']}, 'CF': {'Dribbling/Carrying': ['Carry Length', 'Successful Dribbles'], 'Aerial Ability': ['Aerial Win%'], 'Goal Threat': ['Shots', 'Touches In Box'], 'Pressing': ['Pressures', 'Counterpressures'], 'Chance Creation': ['xG Assisted', 'OP Key Passes', 'Successful Crosses'], 'Finishing': ['xG']}}

PROFILE_METRICS = {
    'GK': {
        'Ball Playing GK': ['Passing%', 'Pr. Pass%', 'xGBuildup', 'GK Aggressive Dist.', 'GSAA', 'xSv%', 'Claims%'],
        'Sweeper Keeper': ['GK Aggressive Dist.', 'Pr. Pass%', 'OP Key Passes', 'Passing%', 'GSAA', 'xSv%', 'Claims%'],
        'Shot Stopper': ['GSAA', 'xSv%', 'Claims%', 'Passing%', 'Pr. Pass%', 'GK Aggressive Dist.']
    },
    'FB': {
        'Attacking Wing Back': ['Dribbles Stopped%', 'Carry Length', 'xG Assisted', 'Successful Dribbles', 'Successful Crosses', 'Ball Recoveries', 'OP Passes Into Box', 'Pr. Pass%', 'PAdj Interceptions', 'Aerial Win%'],
        'Defensive Wing Back': ['Dribbles Stopped%', 'Carry Length', 'PAdj Tackles', 'Successful Dribbles', 'DA OBV', 'Ball Recoveries', 'PAdj Clearances', 'Defensive Regains', 'PAdj Interceptions', 'Aerial Win%']
    },
    'CB': {
        'Box CB': ['Aerial Win%', 'Dribbles Stopped%', 'PAdj Tackles', 'PAdj Interceptions', 'PAdj Clearances', 'Pass Forward%', 'Defensive Regains', 'DA OBV', 'Ball Recoveries', 'Pr. Pass%'],
        'Ball Playing CB': ['Aerial Win%', 'Dribbles Stopped%', 'PAdj Tackles', 'PAdj Interceptions', 'PAdj Clearances', 'Pass Forward%', 'Defensive Regains', 'Pass OBV', 'Ball Recoveries', 'Pr. Pass%']
    },
    'DM': {
        'Ball Playing 6': ['PAdj Tackles', 'PAdj Interceptions', 'OP F3 Passes', 'Ball Recoveries', 'Aerial Win%', 'xGBuildup', 'DA OBV', 'Pr. Pass%', 'Pass OBV', 'Deep Progressions'],
        'Destroyer 6': ['PAdj Tackles', 'PAdj Interceptions', 'Pass Forward%', 'Ball Recoveries', 'Aerial Win%', 'Dribbles Stopped%', 'DA OBV', 'Pr. Pass%', 'Pass OBV', 'Defensive Regains']
    },
    'CM': {
        'No8 - Box to Box': ['xG', 'OP Key Passes', 'xG Assisted', 'PAdj Interceptions', 'PAdj Tackles', 'Aerial Win%', 'Pressures', 'Ball Recoveries', 'PINTIN', 'Scoring Contribution'],
        'No8 - Technical': ['xG', 'Shots', 'xG Assisted', 'OP Key Passes', 'Pass OBV', 'Dribble%', 'xGBuildup', 'Carry Length', 'PINTIN', 'Scoring Contribution']
    },
    'AM': {
        'No10': ['Shots', 'xG', 'Scoring Contribution', 'Pr. Pass%', 'OP Key Passes', 'Successful Dribbles', 'PINTIN', 'xG Assisted', 'Carry Length', 'Shooting%']
    },
    'WF': {
        'Winger': ['xG', 'Shots', 'OP Key Passes', 'Dribbles', 'Successful Dribbles', 'OBV', 'PINTIN', 'Successful Crosses', 'xG Assisted', 'D&C OBV']
    },
    'CF': {
        'Target Forward': ['NP Goals', 'Shots', 'Shooting%', 'xG', 'xG/Shot', 'Shot Touch %', 'Aerial Win%', 'Touches In Box', 'Carry Length', 'Fouls Won'],
        'CF Runner': ['NP Goals', 'Shots', 'Shooting%', 'xG', 'xG/Shot', 'Shot Touch %', 'Aggressive Actions', 'Fouls Won', 'Pressures', 'Counterpressures']
    }
}

@st.cache_data(ttl=7200)
def fetch_competitions_list():
    """Fetch just the list of competitions - fast initial load"""
    try:
        comps = sb.competitions(creds=CREDS)
        return comps
    except Exception as e:
        st.error(f"Error loading competitions: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=7200)
def fetch_player_names_all():
    """Fetch all player names across all competitions for search - optimized"""
    try:
        comps = sb.competitions(creds=CREDS)
        all_players = []
        for _, comp in comps.iterrows():
            try:
                stats = sb.player_season_stats(competition_id=comp['competition_id'], season_id=comp['season_id'], creds=CREDS)
                stats['competition_id'] = comp['competition_id']
                stats['season_id'] = comp['season_id']
                # Only keep essential columns for player search
                stats_minimal = stats[['player_name', 'player_season_minutes', 'competition_id', 'season_id', 'competition_name', 'season_name']].copy()
                all_players.append(stats_minimal)
            except:
                continue
        if all_players:
            df_minimal = pd.concat(all_players, ignore_index=True)
            df_minimal = df_minimal[df_minimal['player_season_minutes'] >= 300]
            df_minimal['Name'] = df_minimal['player_name']
            df_minimal['Competition'] = df_minimal['competition_name']
            df_minimal['Season'] = df_minimal['season_name']
            return df_minimal
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=7200)
def fetch_competition_data(competition_id, season_id):
    """Fetch full data for a specific competition only"""
    try:
        stats = sb.player_season_stats(competition_id=competition_id, season_id=season_id, creds=CREDS)
        stats['competition_id'] = competition_id
        stats['season_id'] = season_id
        # Map StatsBomb fields to our naming convention
        for sb_field, code_name in STATSBOMB_FIELDS.items():
            if sb_field in stats.columns:
                stats[code_name] = stats[sb_field]
        return stats
    except Exception as e:
        st.error(f"Error loading competition data: {str(e)}")
        return pd.DataFrame()

def normalize_stats(df, stats_cols, position_col='Primary Position'):
    df = df.copy()
    df['Position Group'] = df[position_col].map(POSITION_MAPPING) if position_col in df.columns else None
    for col in stats_cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]) or df[col].isna().all():
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Position Group' in df.columns:
            for pos_group in df['Position Group'].dropna().unique():
                mask = df['Position Group'] == pos_group
                values = df.loc[mask, col].dropna()
                if len(values) > 0:
                    try:
                        q01, q99 = values.quantile(0.01), values.quantile(0.99)
                        df.loc[mask, col] = df.loc[mask, col].clip(lower=q01, upper=q99)
                        mean, std = df.loc[mask, col].mean(), df.loc[mask, col].std()
                        if std > 0:
                            df.loc[mask, col] = (df.loc[mask, col] - mean) / std
                            min_val, max_val = df.loc[mask, col].min(), df.loc[mask, col].max()
                            if max_val > min_val:
                                df.loc[mask, col] = 100 * ((df.loc[mask, col] - min_val) / (max_val - min_val))
                    except:
                        continue
    return df

def calculate_percentiles(df, stats_cols, position_col='Primary Position'):
    df_pct = df.copy()
    df_pct['Position Group'] = df_pct[position_col].map(POSITION_MAPPING) if position_col in df.columns else None
    for col in stats_cols:
        if col not in df_pct.columns or not pd.api.types.is_numeric_dtype(df_pct[col]) or df_pct[col].isna().all():
            continue
        df_pct[col] = pd.to_numeric(df_pct[col], errors='coerce')
        if 'Position Group' in df_pct.columns:
            for pos_group in df_pct['Position Group'].dropna().unique():
                mask = df_pct['Position Group'] == pos_group
                values = df_pct.loc[mask, col].dropna()
                if len(values) > 0:
                    for idx in df_pct[mask].index:
                        if pd.notna(df_pct.loc[idx, col]):
                            df_pct.loc[idx, col] = (values <= df_pct.loc[idx, col]).sum() / len(values) * 100
    return df_pct

def calculate_profile_scores(player_row, position):
    if position not in PROFILE_WEIGHTS:
        return {}
    scores = {}
    for profile_name, weights in PROFILE_WEIGHTS[position].items():
        available = [s for s in weights.keys() if s in player_row.index and not pd.isna(player_row[s])]
        if available:
            score = sum(player_row[stat] * weights[stat] for stat in available) / sum(weights[stat] for stat in available)
            scores[profile_name] = round(score)
    if scores:
        scores[f"Complete {position}"] = round(sum(scores.values()) / len(scores))
    return scores

def calculate_attribute_scores(player_row, position):
    if position not in ATTRIBUTES_BY_POSITION:
        return {}
    scores = {}
    for attr_name, metrics in ATTRIBUTES_BY_POSITION[position].items():
        available = [m for m in metrics if m in player_row.index and not pd.isna(player_row[m])]
        scores[attr_name] = round(sum(player_row[m] for m in available) / len(available)) if available else 0
    return scores

def create_styled_percentile_chart(player_data_pct, metrics_dict, player_name):
    metrics_data = []
    for category, metrics in metrics_dict.items():
        for metric in metrics:
            if metric in player_data_pct.index and not pd.isna(player_data_pct[metric]):
                val = round(float(player_data_pct[metric]))
                metrics_data.append({'Metric': metric, 'Percentile': val})
    if not metrics_data:
        return None
    df_display = pd.DataFrame(metrics_data).sort_values('Percentile', ascending=False)
    chart_html = f"<div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 15px; padding: 2rem; box-shadow: 0 8px 16px rgba(0,0,0,0.3); margin: 1rem 0;'><h3 style='color: white; margin-bottom: 1.5rem; text-align: center; font-size: 24px;'>{player_name} - Percentile Rankings</h3>"
    for _, row in df_display.iterrows():
        metric, pct = row['Metric'], row['Percentile']
        color = COLORS['green'] if pct >= 70 else COLORS['amber'] if pct >= 50 else COLORS['red']
        chart_html += f"<div style='margin-bottom: 1rem;'><div style='display: flex; justify-content: space-between; margin-bottom: 0.3rem;'><span style='color: white; font-weight: 500; font-style: italic;'>{metric}</span><span style='color: white; font-weight: bold; font-size: 18px;'>{pct}%</span></div><div style='background: rgba(255,255,255,0.2); height: 28px; border-radius: 14px; overflow: hidden; position: relative;'><div style='background: {color}; width: {pct}%; height: 100%; border-radius: 14px; position: relative;'><div style='position: absolute; right: 10px; top: 50%; transform: translateY(-50%); width: 16px; height: 16px; background: white; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.3);'></div></div></div></div>"
    chart_html += "</div>"
    return chart_html

def create_comparison_card(p1_data, p2_data, p1_name, p2_name, metrics_dict):
    all_metrics, p1_vals, p2_vals = [], [], []
    for category, metrics in metrics_dict.items():
        for metric in metrics:
            if metric in p1_data.index and not pd.isna(p1_data[metric]) and metric in p2_data.index and not pd.isna(p2_data[metric]):
                all_metrics.append(metric)
                p1_vals.append(round(float(p1_data[metric]), 2))
                p2_vals.append(round(float(p2_data[metric]), 2))
    if not all_metrics:
        return None
    html = f"<div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 2rem; border-radius: 15px; margin: 1rem 0;'><h2 style='text-align: center; color: white; margin-bottom: 2rem;'>{p1_name} vs {p2_name}</h2><div style='display: grid; grid-template-columns: 1fr 2fr 1fr; gap: 1rem; align-items: center;'>"
    for metric, p1_val, p2_val in zip(all_metrics, p1_vals, p2_vals):
        p1_color = COLORS['green'] if p1_val > p2_val else COLORS['grey']
        p2_color = COLORS['green'] if p2_val > p1_val else COLORS['grey']
        html += f"<div style='text-align: right; padding: 0.75rem;'><span style='color: {p1_color}; font-size: 18px; font-weight: bold;'>{p1_val}</span></div><div style='text-align: center; padding: 0.75rem; color: white; font-weight: 500;'>{metric}</div><div style='text-align: left; padding: 0.75rem;'><span style='color: {p2_color}; font-size: 18px; font-weight: bold;'>{p2_val}</span></div>"
    html += "</div></div>"
    return html

def main():
    st.markdown(f"<div style='background: {COLORS['secondary']}; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'><h1 style='margin: 0; color: white; text-align: center;'>🔵 Bristol Rovers Player Profiling</h1></div>", unsafe_allow_html=True)
    st.sidebar.title("🔍 Search & Filter")
    
    # Stage 1: Load player names only (fast)
    with st.spinner("Loading player database..."):
        df_players = fetch_player_names_all()
    
    if df_players.empty:
        st.error("No data loaded")
        st.stop()
    
    all_player_names = sorted(df_players['Name'].dropna().unique().tolist())
    selected_player = st.sidebar.selectbox("1️⃣ Search Player", ["Select Player..."] + all_player_names)
    
    if selected_player == "Select Player...":
        st.info("👈 Start by searching for a player")
        st.stop()
    
    # Get competitions for selected player
    player_comps = df_players[df_players['Name'] == selected_player][['Competition', 'Season', 'competition_id', 'season_id']].drop_duplicates()
    comp_displays = player_comps.apply(lambda x: f"{x['Competition']} - {x['Season']}", axis=1).tolist()
    selected_comp_display = st.sidebar.selectbox("2️⃣ Select Competition", ["Select..."] + comp_displays)
    
    if selected_comp_display == "Select...":
        st.info(f"👈 Select a competition for {selected_player}")
        st.stop()
    
    comp_idx = comp_displays.index(selected_comp_display)
    comp_row = player_comps.iloc[comp_idx]
    comp_id, season_id = comp_row['competition_id'], comp_row['season_id']
    comp_name = comp_row['Competition']
    
    # Stage 2: Load full data for selected competition only (fast)
    with st.spinner(f"Loading {comp_name} data..."):
        df_comp = fetch_competition_data(comp_id, season_id)
    
    if df_comp.empty:
        st.error("Failed to load competition data")
        st.stop()
    
    df_comp = df_comp[df_comp['Minutes'] >= 300] if 'Minutes' in df_comp.columns else df_comp
    player_data_raw = df_comp[df_comp['Name'] == selected_player].iloc[0]
    player_position = POSITION_MAPPING.get(player_data_raw.get('Primary Position'), 'CM')
    selected_position = st.sidebar.selectbox("3️⃣ Select Position", ['GK', 'CB', 'FB', 'DM', 'CM', 'AM', 'WF', 'CF'], index=['GK', 'CB', 'FB', 'DM', 'CM', 'AM', 'WF', 'CF'].index(player_position) if player_position in ['GK', 'CB', 'FB', 'DM', 'CM', 'AM', 'WF', 'CF'] else 4)
    info_cols = ['Name', 'Team', 'Competition', 'Season', 'Minutes', 'Primary Position']
    stats_cols = [col for col in df_comp.columns if col not in info_cols and pd.api.types.is_numeric_dtype(df_comp[col])]
    df_normalized = normalize_stats(df_comp, stats_cols)
    df_percentile = calculate_percentiles(df_comp, stats_cols)
    player_data = df_normalized[df_normalized['Name'] == selected_player].iloc[0]
    player_data_pct = df_percentile[df_percentile['Name'] == selected_player].iloc[0]
    tab1, tab2, tab3 = st.tabs(["📊 Player Profile", "📈 Comparison", "🏆 Leaderboard"])
    
    with tab1:
        st.markdown(f"<h2 style='color: white;'>{selected_player}</h2><p style='color: {COLORS['accent']};'>{player_data.get('Team')} • {comp_name} • {int(player_data.get('Minutes', 0))} mins</p>", unsafe_allow_html=True)
        profile_scores = calculate_profile_scores(player_data, selected_position)
        attribute_scores = calculate_attribute_scores(player_data, selected_position)
        if profile_scores:
            st.subheader("🎯 Profile Suitability (vs same position)")
            for prof, score in sorted(profile_scores.items(), key=lambda x: x[1], reverse=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{prof}**")
                with col2:
                    st.write(f"**{score}**")
                st.progress(score/100)
        st.markdown("---")
        if attribute_scores:
            st.subheader("⚡ Key Attributes (vs same position)")
            for attr, score in sorted(attribute_scores.items(), key=lambda x: x[1], reverse=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{attr}**")
                with col2:
                    color = COLORS['green'] if score >= 80 else COLORS['primary'] if score >= 60 else COLORS['amber'] if score >= 40 else COLORS['red']
                    st.markdown(f"<span style='color: {color}; font-weight: bold; font-size: 18px;'>{score}</span>", unsafe_allow_html=True)
                st.progress(score/100)
        st.markdown("---")
        
        # Percentile Chart with Profile Selection - ONLY profile-specific metrics
        st.subheader("📊 Percentile Rankings")
        
        # Check if position has profile-specific metrics
        if selected_position in PROFILE_METRICS:
            profile_options = list(PROFILE_METRICS[selected_position].keys())
            # Default to first profile option
            default_index = 0
            chart_type = st.selectbox(
                "Select Profile View",
                profile_options,
                index=default_index,
                key="profile_chart_selector"
            )
            
            # Use profile-specific metrics
            profile_metrics = PROFILE_METRICS[selected_position][chart_type]
            # Create a simple dict structure for the chart
            metrics_dict = {chart_type: profile_metrics}
            chart_html = create_styled_percentile_chart(player_data_pct, metrics_dict, selected_player)
            if chart_html:
                st.markdown(chart_html, unsafe_allow_html=True)
        else:
            # Fallback to default attributes if no profile metrics defined
            if selected_position in ATTRIBUTES_BY_POSITION:
                chart_html = create_styled_percentile_chart(player_data_pct, ATTRIBUTES_BY_POSITION[selected_position], selected_player)
                if chart_html:
                    st.markdown(chart_html, unsafe_allow_html=True)
    
    with tab2:
        st.header("📈 Cross-League Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Player 1")
            comp1_list = df_players[['Competition', 'Season', 'competition_id', 'season_id']].drop_duplicates()
            comp1_displays = comp1_list.apply(lambda x: f"{x['Competition']} - {x['Season']}", axis=1).tolist()
            sel_comp1 = st.selectbox("Competition 1", ["Select..."] + comp1_displays, key="c1")
            if sel_comp1 != "Select...":
                c1_idx = comp1_displays.index(sel_comp1)
                c1_row = comp1_list.iloc[c1_idx]
                # Load data for comp1 on demand
                df_c1 = fetch_competition_data(c1_row['competition_id'], c1_row['season_id'])
                df_c1 = df_c1[df_c1['Minutes'] >= 300] if 'Minutes' in df_c1.columns else df_c1
                players1 = sorted(df_c1['Name'].unique().tolist())
                player1 = st.selectbox("Player 1", ["Select..."] + players1, key="p1")
            else:
                player1 = "Select..."
                df_c1 = pd.DataFrame()
        with col2:
            st.subheader("Player 2")
            comp2_list = df_players[['Competition', 'Season', 'competition_id', 'season_id']].drop_duplicates()
            comp2_displays = comp2_list.apply(lambda x: f"{x['Competition']} - {x['Season']}", axis=1).tolist()
            sel_comp2 = st.selectbox("Competition 2", ["Select..."] + comp2_displays, key="c2")
            if sel_comp2 != "Select...":
                c2_idx = comp2_displays.index(sel_comp2)
                c2_row = comp2_list.iloc[c2_idx]
                # Load data for comp2 on demand
                df_c2 = fetch_competition_data(c2_row['competition_id'], c2_row['season_id'])
                df_c2 = df_c2[df_c2['Minutes'] >= 300] if 'Minutes' in df_c2.columns else df_c2
                players2 = sorted(df_c2['Name'].unique().tolist())
                player2 = st.selectbox("Player 2", ["Select..."] + players2, key="p2")
            else:
                player2 = "Select..."
                df_c2 = pd.DataFrame()
        if player1 != "Select..." and player2 != "Select..." and not df_c1.empty and not df_c2.empty:
            st.markdown("---")
            # Get stats columns for normalization
            info_cols = ['Name', 'Team', 'Competition', 'Season', 'Minutes', 'Primary Position']
            stats_cols_c1 = [col for col in df_c1.columns if col not in info_cols and pd.api.types.is_numeric_dtype(df_c1[col])]
            stats_cols_c2 = [col for col in df_c2.columns if col not in info_cols and pd.api.types.is_numeric_dtype(df_c2[col])]
            df_c1_norm = normalize_stats(df_c1, stats_cols_c1)
            df_c2_norm = normalize_stats(df_c2, stats_cols_c2)
            p1_data = df_c1_norm[df_c1_norm['Name'] == player1].iloc[0]
            p2_data = df_c2_norm[df_c2_norm['Name'] == player2].iloc[0]
            p1_pos = POSITION_MAPPING.get(p1_data.get('Primary Position'), 'CM')
            p2_pos = POSITION_MAPPING.get(p2_data.get('Primary Position'), 'CM')
            st.subheader("⚖️ Profile Scores Comparison")
            profile1 = calculate_profile_scores(p1_data, p1_pos)
            profile2 = calculate_profile_scores(p2_data, p2_pos)
            if profile1 and profile2:
                all_profiles = set(profile1.keys()) | set(profile2.keys())
                for prof in sorted(all_profiles):
                    s1 = profile1.get(prof, 0)
                    s2 = profile2.get(prof, 0)
                    col_p1, col_label, col_p2 = st.columns([1, 2, 1])
                    with col_p1:
                        c1 = COLORS['green'] if s1 > s2 else COLORS['grey']
                        st.markdown(f"<div style='text-align: right; color: {c1}; font-weight: bold; font-size: 18px;'>{s1}</div>", unsafe_allow_html=True)
                    with col_label:
                        st.markdown(f"<div style='text-align: center; color: white;'>{prof}</div>", unsafe_allow_html=True)
                    with col_p2:
                        c2 = COLORS['green'] if s2 > s1 else COLORS['grey']
                        st.markdown(f"<div style='text-align: left; color: {c2}; font-weight: bold; font-size: 18px;'>{s2}</div>", unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("📊 Detailed Metrics Comparison")
            use_position = p1_pos if p1_pos == p2_pos else selected_position
            if use_position in ATTRIBUTES_BY_POSITION:
                comparison_html = create_comparison_card(p1_data, p2_data, player1, player2, ATTRIBUTES_BY_POSITION[use_position])
                if comparison_html:
                    st.markdown(comparison_html, unsafe_allow_html=True)
                else:
                    st.warning("Unable to create comparison")
    
    with tab3:
        st.header("🏆 Leaderboard")
        if selected_position in PROFILE_WEIGHTS:
            all_profiles = list(PROFILE_WEIGHTS[selected_position].keys()) + [f"Complete {selected_position}"]
            leaderboard_profile = st.selectbox("Select Profile Filter", ["All Profiles"] + all_profiles)
        else:
            leaderboard_profile = "All Profiles"
        st.subheader(f"Top {selected_position} Players - {comp_name}")
        if leaderboard_profile != "All Profiles":
            st.caption(f"Ranked by {leaderboard_profile} score")
        leaderboard_data = []
        for _, row in df_normalized.iterrows():
            row_pos = POSITION_MAPPING.get(row.get('Primary Position'), 'CM')
            if row_pos == selected_position:
                scores = calculate_profile_scores(row, selected_position)
                if scores:
                    if leaderboard_profile == "All Profiles":
                        score = max(scores.values())
                        profile = max(scores, key=scores.get)
                    else:
                        score = scores.get(leaderboard_profile, 0)
                        profile = leaderboard_profile
                    leaderboard_data.append({'Player': row.get('Name'), 'Team': row.get('Team'), 'Minutes': int(row.get('Minutes', 0)), 'Profile': profile, 'Score': score})
        if leaderboard_data:
            lb_df = pd.DataFrame(leaderboard_data).sort_values('Score', ascending=False).head(20)
            lb_df.insert(0, 'Rank', range(1, len(lb_df) + 1))
            st.dataframe(lb_df, use_container_width=True, height=600, hide_index=True)
            csv = lb_df.to_csv(index=False)
            st.download_button(label="📥 Download Leaderboard CSV", data=csv, file_name=f"{selected_position}_leaderboard_{comp_name}.csv", mime="text/csv")
        else:
            st.warning("No leaderboard data available")

if __name__ == "__main__":
    main()