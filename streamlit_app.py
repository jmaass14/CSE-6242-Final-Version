##############################################################
# Regime-Aware Market Conditions Analyzer - Streamlit Dashboard
##############################################################
# CSE 6242 Project — Team #255
#
# Requirements:
#    streamlit 
#    plotly 
#    pandas 
#    numpy 
#    scikit-learn 
#    hmmlearn
#
# Usage:
#   In CMD Run:
#    streamlit run streamlit_app.py
###############################################################

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    import json
    import os

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPS = str(e)
    import json
    import os # Keep for ts

# Initilization
DATA_PATH = "FRED_DATA.csv"
MODEL_OUTPUT_PATH = "regime_output.json"

# Regime metadata
REGIME_META = [
    {"id": 0, "key": "growth", "label": "Low-Vol Growth", "color": "#2ecc71"},
    {"id": 1, "key": "crisis", "label": "High-Vol Crisis", "color": "#e74c3c"},
    {"id": 2, "key": "transition", "label": "Transition/Tightening", "color": "#f39c12"},
]

# Page Structure Config
st.set_page_config(
    page_title="Regime-Aware Market Analyzer",
    page_icon="icons8-bar-chart-100.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Troubleshoot
# Loading Regime Data
@st.cache_data # Use this to cache for performance
def load_regime_data():
    try:
        with open(MODEL_OUTPUT_PATH, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"""
        **Could not find {MODEL_OUTPUT_PATH}**
        Please run the HMM model first: python hmm_model.py
        Try again once complete.
        """)
        st.stop()
    except Exception as e:
        st.error(f"Error loading regime data: {str(e)}")
        st.stop()

# Data manipulation from the JSON created by the hmm_model.py
@st.cache_data
def process_time_series_data(data):
    df = pd.DataFrame(data['time_series'])
    df['date'] = pd.to_datetime(df['date'])

    numeric_cols = ['SP500_return', 'bond_return', 'cumReturn', 'bondCumReturn',
                   'FEDFUNDS', 'UNRATE', 'yield_spread', 'CPI', 'DGS10']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'CPI' in df.columns:
        df = df.sort_values('date').reset_index(drop = True)
        cpi_lookup = df.set_index('date')['CPI']
        def calc_cpi_yoy(row):
            if pd.isna(row['CPI']):
                return None
            prior_date = row['date'] - pd.DateOffset(years = 1)
            prior_matches = cpi_lookup[(cpi_lookup.index.year == prior_date.year) &
                                       (cpi_lookup.index.month == prior_date.month)]
            if prior_matches.empty or pd.isna(prior_matches.iloc[0]):
                return None
            return (row['CPI'] - prior_matches.iloc[0]) / prior_matches.iloc[0] * 100
        df['CPI_YoY'] = df.apply(calc_cpi_yoy, axis = 1)

    # Calculating cumulative returns
    if 'cumReturn' not in df.columns:
        df = calculate_cumulative_returns(df)

    return df

# Cumulative returns for both bonds and equities
# Equity & Bond cumulative returns are indexed to 100
def calculate_cumulative_returns(df):
    df_clean = df.copy()
    equity_returns = df_clean['SP500_return'].dropna()
    if len(equity_returns) > 0:
        cum_equity = (1 + equity_returns).cumprod() * 100
        df_clean.loc[equity_returns.index, 'cumReturn'] = cum_equity

    bond_returns = df_clean['bond_return'].dropna()
    if len(bond_returns) > 0:
        cum_bond = (1 + bond_returns).cumprod() * 100
        df_clean.loc[bond_returns.index, 'bondCumReturn'] = cum_bond

    return df_clean

def get_regime_colors():
    # Colors for viz
    return {regime['id']: regime['color'] for regime in REGIME_META}

# Background regime shading that I had used in the original D3 variation
def format_regime_bands(df, regime_colors):
    bands = []
    current_regime = None
    start_date = None

    for idx, row in df.iterrows():
        if pd.isna(row['regime']):
            continue

        regime_id = int(row['regime'])

        if current_regime != regime_id:
            if current_regime is not None:
                bands.append({
                    'regime_id': current_regime,
                    'start': start_date,
                    'end': df.iloc[idx-1]['date'],
                    'color': regime_colors[current_regime],
                    'label': next(r['label'] for r in REGIME_META if r['id'] == current_regime)
                })

            current_regime = regime_id
            start_date = row['date']

    if current_regime is not None:
        bands.append({
            'regime_id': current_regime,
            'start': start_date,
            'end': df.iloc[-1]['date'],
            'color': regime_colors[current_regime],
            'label': next(r['label'] for r in REGIME_META if r['id'] == current_regime)
        })

    return bands

def main():
    # Check dependencies first
    if not DEPENDENCIES_AVAILABLE:
        st.error(f"""
        Missing Dependencies -
        Required packages are not installed: {MISSING_DEPS}
        Please install the required packages: streamlit plotly pandas numpy scikit-learn hmmlearn
        ```
        """)
        st.stop()

    # Title and description
    st.title("Regime-Aware Market Conditions Analyzer")
    st.markdown("""
    **Hidden Markov Model · 3 Market Regimes · FRED Data 1997–2026 · CSE 6242 Team #255**

    This dashboard analyzes market regimes using a Hidden Markov Model to identify periods of:
    - **Low-Vol Growth**: Stable market conditions with positive returns
    - **High-Vol Crisis**: Volatile periods with negative performance
    - **Transition/Tightening**: Intermediate periods of market uncertainty
    """)

    with st.spinner("Loading regime analysis data..."):
        regime_data = load_regime_data()
        df = process_time_series_data(regime_data)
        regime_colors = get_regime_colors()
        regime_bands = format_regime_bands(df, regime_colors)

    # Sidebar controls
    st.sidebar.image("icons8-bar-chart-64.png", width = 40) # Can't align, keep as is
    st.sidebar.header("Dashboard Controls")

    # Data range selector
    st.sidebar.subheader("Date Range")
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(df['date'].min().date(), df['date'].max().date()),
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
    )

    # Date Range Filter
    if len(date_range) == 2:
        mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
        df_filtered = df.loc[mask].copy()
        regime_bands_filtered = [
            band for band in regime_bands
            if band['end'].date() >= date_range[0] and band['start'].date() <= date_range[1]
        ]
    else:
        df_filtered = df.copy()
        regime_bands_filtered = regime_bands

    # Legend
    st.subheader("Regime Legend")
    cols = st.columns(len(REGIME_META))
    for i, regime in enumerate(REGIME_META):
        with cols[i]:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 20px; height: 20px; background-color: {regime['color']}; border-radius: 3px;"></div>
                <span style="font-weight: 600;">{regime['label']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Regime Timeline", "Macro Conditions", "Summary Statistics", "Model Diagnostics"])

    # Tab 1: Regime Timeline
    with tab1:
        st.subheader("Regime Timeline — Cumulative Returns")

        # Series selection
        col1, col2 = st.columns(2)
        with col1:
            show_equity = st.checkbox("S&P 500", value=True, key="equity_toggle")
        with col2:
            show_bonds = st.checkbox("10Y Treasury Bond", value=True, key="bond_toggle")

        if show_equity or show_bonds:
            fig = create_timeline_chart(df_filtered, regime_bands_filtered, show_equity, show_bonds, regime_colors)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one series to display.")

    # Tab 2 Here - Macro Conditions
    with tab2:
        st.subheader("Macro Conditions")

        # Macro indicators selection
        macro_options = {
            'UNRATE': 'Unemployment Rate (%)',
            'FEDFUNDS': 'Fed Funds Rate (%)',
            'yield_spread': 'Yield Spread (%)',
            'CPI_YoY': 'CPI YoY (%)'
        }

        selected_macros = st.multiselect(
            "Select indicators to display",
            options=list(macro_options.keys()),
            default=list(macro_options.keys()),
            format_func=lambda x: macro_options[x]
        )

        if selected_macros:
            fig = create_macro_chart(df_filtered, regime_bands_filtered, selected_macros, macro_options, regime_colors)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one macro indicator to display.")

    # Tab 3: Summary Statistics
    with tab3:
        st.subheader("Regime Summary Statistics")
        display_summary_statistics(regime_data['state_statistics'], regime_colors)

    # Tab 4: Model Diagnostics
    with tab4:
        st.subheader("Model Diagnostics")
        display_model_diagnostics(regime_data)

# Visualization functionality

def create_timeline_chart(df, regime_bands, show_equity, show_bonds, regime_colors):
    fig = go.Figure()

    # Add regime background
    for band in regime_bands:
        fig.add_vrect(
            x0=band['start'],
            x1=band['end'],
            fillcolor=band['color'],
            opacity=0.2,
            layer="below",
            line_width=0,
        )

    # Add series lines
    if show_equity and 'cumReturn' in df.columns:
        equity_data = df.dropna(subset=['cumReturn'])
        fig.add_trace(go.Scatter(
            x=equity_data['date'],
            y=equity_data['cumReturn'],
            mode='lines',
            name='S&P 500',
            line=dict(color='#2c3e50', width=2),
            hovertemplate='<b>S&P 500</b><br>Date: %{x}<br>Value: %{y:.1f}<extra></extra>'
        ))

    if show_bonds and 'bondCumReturn' in df.columns:
        bond_data = df.dropna(subset=['bondCumReturn'])
        fig.add_trace(go.Scatter(
            x=bond_data['date'],
            y=bond_data['bondCumReturn'],
            mode='lines',
            name='10Y Treasury Bond',
            line=dict(color='#e67e22', width=2, dash='dash'),
            hovertemplate='<b>10Y Treasury Bond</b><br>Date: %{x}<br>Value: %{y:.1f}<extra></extra>'
        ))

    # base line
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.7)

    fig.update_layout(
        title="Cumulative Returns with Market Regimes",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (base 100)",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_macro_chart(df, regime_bands, selected_macros, macro_options, regime_colors):
    fig = go.Figure()

    # regime background
    for band in regime_bands:
        fig.add_vrect(
            x0=band['start'],
            x1=band['end'],
            fillcolor=band['color'],
            opacity=0.2,
            layer="below",
            line_width=0,
        )

    # Color palette for macro indicators
    macro_colors = {
        'UNRATE': '#8e44ad',
        'FEDFUNDS': '#2980b9',
        'yield_spread': '#16a085',
        'CPI_YoY': '#c0392b'
    }

    # macro indicator lines
    for macro in selected_macros:
        if macro in df.columns:
            macro_data = df.dropna(subset=[macro])
            fig.add_trace(go.Scatter(
                x=macro_data['date'],
                y=macro_data[macro],
                mode='lines',
                name=macro_options[macro],
                line=dict(color=macro_colors.get(macro, '#666666'), width=2),
                hovertemplate=f'<b>{macro_options[macro]}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}%<extra></extra>'
            ))

    # zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.7)

    fig.update_layout(
        title="Macro Economic Indicators with Market Regimes",
        xaxis_title="Date",
        yaxis_title="Indicator Value (%)",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

# Regime summary statistics

def display_summary_statistics(state_stats, regime_colors):

    # Metrics
    metrics = [
        {'key': 'mean_SP500_return', 'label': 'Avg Monthly Return', 'format': '{:.2f}%', 'scale': 100},
        {'key': 'std_SP500_return', 'label': 'Ann. Volatility', 'format': '{:.1f}%', 'scale': 100 * np.sqrt(12)},
        {'key': 'max_drawdown_equity', 'label': 'Max Drawdown (EQ)', 'format': '{:.1f}%', 'scale': 100},
        {'key': 'equity_bond_corr', 'label': 'Equity-Bond Corr', 'format': '{:.3f}', 'scale': 1},
        {'key': 'mean_UNRATE', 'label': 'Avg Unemployment', 'format': '{:.1f}%', 'scale': 1},
        {'key': 'mean_yield_spread', 'label': 'Avg Yield Spread', 'format': '{:.2f}%', 'scale': 1}
    ]

    # Summary cards
    cols = st.columns(3)

    for i, metric in enumerate(metrics[:3]):
        with cols[i]:
            st.markdown(f"### {metric['label']}")
            fig = create_metric_bar_chart(state_stats, metric, regime_colors)
            st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(3)
    for i, metric in enumerate(metrics[3:]):
        with cols[i]:
            st.markdown(f"### {metric['label']}")
            fig = create_metric_bar_chart(state_stats, metric, regime_colors)
            st.plotly_chart(fig, use_container_width=True)

def create_metric_bar_chart(state_stats, metric, regime_colors):
    """Create a bar chart for a specific metric across regimes."""
    labels = [stat['label'] for stat in state_stats]
    values = []
    colors = []

    for stat in state_stats:
        value = stat.get(metric['key'])
        if value is not None:
            values.append(value * metric['scale'])
        else:
            values.append(0)
        colors.append(regime_colors[stat['regime_id']])

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[metric['format'].format(v) if v != 0 else '—' for v in values],
            textposition='auto',
        )
    ])

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        xaxis_title="",
        yaxis_title=""
    )

    # Add zero line if needed
    if min(values) < 0:
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.7)

    return fig

# Model diagnostics

def display_model_diagnostics(regime_data):
    metadata = regime_data['metadata']

    # Model performance
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Log-Likelihood", f"{metadata['log_likelihood']:.2f}")
    with col2:
        st.metric("AIC", f"{metadata['aic']:.2f}")
    with col3:
        st.metric("BIC", f"{metadata['bic']:.2f}")

    # Model details
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Features Used:**")
        for feature in metadata['features_used']:
            st.markdown(f"• {feature}")

        st.markdown(f"**Number of Regimes:** {metadata['n_regimes']}")
        st.markdown(f"**Observations:** {metadata['n_observations']}")

    with col2:
        st.markdown("**Date Ranges:**")
        st.markdown(f"• **Full Data:** {metadata['date_range']['start']} to {metadata['date_range']['end']}")
        st.markdown(f"• **Model Data:** {metadata['model_date_range']['start']} to {metadata['model_date_range']['end']}")

    # Transition matrix
    st.subheader("Transition Matrix")
    st.markdown("Probability of transitioning from one regime to another (row → column)")

    transition_matrix = np.array(regime_data['transition_matrix'])
    regime_labels = [regime['label'] for regime in REGIME_META]

    # Transition matrix heatmap
    fig = px.imshow(
        transition_matrix,
        x=regime_labels,
        y=regime_labels,
        color_continuous_scale='RdYlBu_r',
        aspect='auto',
        text_auto='.3f'
    )

    fig.update_layout(
        title="Regime Transition Probabilities",
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Regime duration statistics
    st.subheader("Regime Statistics")
    stats_df = pd.DataFrame(regime_data['state_statistics'])

    display_cols = ['label', 'n_months', 'pct_months', 'recession_overlap_pct']
    display_names = ['Regime', 'Duration (Months)', 'Percentage of Time', 'Recession Overlap (%)']

    formatted_df = stats_df[display_cols].copy()
    formatted_df.columns = display_names
    formatted_df['Percentage of Time'] = (formatted_df['Percentage of Time'] * 100).round(1)
    formatted_df['Recession Overlap (%)'] = (formatted_df['Recession Overlap (%)'] * 100).round(1)

    st.dataframe(formatted_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()