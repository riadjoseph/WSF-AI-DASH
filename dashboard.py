import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings

# Optional imports for advanced analytics
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.decomposition import seasonal_decompose
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="AI Overview Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(sample_size=None):
    """Load optimized sample data for Streamlit Cloud"""
    try:
        # Load from local sample file (optimized for Streamlit Cloud)
        df = pd.read_parquet('data/streamlit_sample.parquet')
        
        # Basic data cleaning
        df = df.dropna(subset=['brand', 'country', 'AI Overview presence'])
        
        # Convert data types for better performance
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
        df['Search Volume'] = pd.to_numeric(df['Search Volume'], errors='coerce')
        df['Traffic'] = pd.to_numeric(df['Traffic'], errors='coerce')
        
        # Convert Month to datetime
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
        
        # Sample data if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def get_ai_overview_stats(df):
    """Calculate AI Overview summary statistics"""
    total_records = len(df)
    ai_present = df['AI Overview presence'].sum()
    ai_percentage = (ai_present / total_records) * 100 if total_records > 0 else 0
    
    stats = {
        'total_records': total_records,
        'ai_present_count': ai_present,
        'ai_present_percentage': ai_percentage,
        'brands_with_ai': df[df['AI Overview presence']]['brand'].nunique(),
        'countries_with_ai': df[df['AI Overview presence']]['country'].nunique(),
        'avg_position_with_ai': df[df['AI Overview presence']]['Position'].mean(),
        'avg_position_without_ai': df[~df['AI Overview presence']]['Position'].mean(),
        'avg_search_volume_with_ai': df[df['AI Overview presence']]['Search Volume'].mean(),
        'avg_search_volume_without_ai': df[~df['AI Overview presence']]['Search Volume'].mean()
    }
    return stats

def create_ai_overview_by_brand(df):
    """Create AI Overview presence analysis by brand - both calculations"""
    # Calculation 1: AI Overview rate per brand (what % of each brand's records have AI Overview)
    brand_rates = df.groupby('brand').agg({
        'AI Overview presence': ['count', 'sum'],
        'Search Volume': 'sum'
    }).round(2)
    
    brand_rates.columns = ['Total_Records', 'AI_Present_Count', 'Total_Search_Volume']
    brand_rates['AI_Rate_Percent'] = (brand_rates['AI_Present_Count'] / brand_rates['Total_Records'] * 100).round(2)
    brand_rates = brand_rates.reset_index()
    
    # Calculation 2: AI Overview distribution (what % of all AI Overview records belong to each brand)
    ai_only_df = df[df['AI Overview presence'] == True]
    brand_distribution = ai_only_df['brand'].value_counts(normalize=True) * 100
    brand_distribution_df = brand_distribution.reset_index()
    brand_distribution_df.columns = ['brand', 'AI_Distribution_Percent']
    brand_distribution_df['AI_Distribution_Percent'] = brand_distribution_df['AI_Distribution_Percent'].round(1)
    
    # Merge both calculations
    brand_combined = brand_rates.merge(brand_distribution_df, on='brand', how='left')
    
    # Create two visualizations
    fig1 = px.bar(
        brand_combined,
        x='brand',
        y='AI_Rate_Percent',
        title="AI Overview Rate per Brand<br><sub>What % of each brand's records have AI Overview</sub>",
        color='AI_Rate_Percent',
        color_continuous_scale='Blues',
        text='AI_Rate_Percent'
    )
    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig1.update_layout(height=700, showlegend=False)
    
    fig2 = px.bar(
        brand_combined,
        x='brand',
        y='AI_Distribution_Percent',
        title="AI Overview Distribution by Brand<br><sub>Of all AI Overview records, what % belong to each brand</sub>",
        color='AI_Distribution_Percent',
        color_continuous_scale='Greens',
        text='AI_Distribution_Percent'
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(height=700, showlegend=False)
    
    return fig1, fig2, brand_combined

def create_ai_overview_by_country(df):
    """Create AI Overview presence analysis by country - both calculations"""
    # Calculation 1: AI Overview rate per country (what % of each country's records have AI Overview)
    country_rates = df.groupby('country').agg({
        'AI Overview presence': ['count', 'sum'],
        'Search Volume': 'sum'
    }).round(2)
    
    country_rates.columns = ['Total_Records', 'AI_Present_Count', 'Total_Search_Volume']
    country_rates['AI_Rate_Percent'] = (country_rates['AI_Present_Count'] / country_rates['Total_Records'] * 100).round(2)
    country_rates = country_rates.reset_index().sort_values('AI_Rate_Percent', ascending=False)
    
    # Calculation 2: AI Overview distribution (what % of all AI Overview records belong to each country)
    ai_only_df = df[df['AI Overview presence'] == True]
    country_distribution = ai_only_df['country'].value_counts(normalize=True) * 100
    country_distribution_df = country_distribution.reset_index()
    country_distribution_df.columns = ['country', 'AI_Distribution_Percent']
    country_distribution_df['AI_Distribution_Percent'] = country_distribution_df['AI_Distribution_Percent'].round(1)
    
    # Merge both calculations
    country_combined = country_rates.merge(country_distribution_df, on='country', how='left')
    
    # Create two visualizations
    fig1 = px.bar(
        country_rates,
        x='country',
        y='AI_Rate_Percent',
        title="AI Overview Rate per Country<br><sub>What % of each country's records have AI Overview</sub>",
        color='AI_Rate_Percent',
        color_continuous_scale='Reds',
        text='AI_Rate_Percent'
    )
    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig1.update_layout(height=700, showlegend=False, xaxis_tickangle=-45)
    
    fig2 = px.bar(
        country_distribution_df,
        x='country',
        y='AI_Distribution_Percent',
        title="AI Overview Distribution by Country<br><sub>Of all AI Overview records, what % belong to each country</sub>",
        color='AI_Distribution_Percent',
        color_continuous_scale='Oranges',
        text='AI_Distribution_Percent'
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(height=700, showlegend=False, xaxis_tickangle=-45)
    
    return fig1, fig2, country_combined

def create_ai_overview_by_position(df):
    """Create AI Overview analysis by SERP position"""
    # Create position bins
    df['Position_Bin'] = pd.cut(df['Position'], bins=[0, 3, 10, 20, 50, 100], 
                               labels=['1-3', '4-10', '11-20', '21-50', '51-100'])
    
    position_ai = df.groupby('Position_Bin').agg({
        'AI Overview presence': ['count', 'sum'],
        'Search Volume': 'sum'
    }).round(2)
    
    position_ai.columns = ['Total_Records', 'AI_Present_Count', 'Total_Search_Volume']
    position_ai['AI_Percentage'] = (position_ai['AI_Present_Count'] / position_ai['Total_Records'] * 100).round(2)
    position_ai = position_ai.reset_index()
    
    fig = px.bar(
        position_ai,
        x='Position_Bin',
        y='AI_Percentage',
        title="AI Overview Presence by SERP Position Range (%)",
        color='AI_Percentage',
        color_continuous_scale='Greens',
        text='AI_Percentage'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=700, showlegend=False)
    return fig, position_ai

def create_ai_overview_search_volume_analysis(df):
    """Create AI Overview analysis by search volume"""
    # Create search volume bins
    df['Volume_Bin'] = pd.cut(df['Search Volume'], 
                             bins=[0, 100, 500, 1000, 5000, float('inf')], 
                             labels=['0-100', '101-500', '501-1K', '1K-5K', '5K+'])
    
    volume_ai = df.groupby('Volume_Bin').agg({
        'AI Overview presence': ['count', 'sum'],
        'Search Volume': 'sum'
    }).round(2)
    
    volume_ai.columns = ['Total_Records', 'AI_Present_Count', 'Total_Search_Volume']
    volume_ai['AI_Percentage'] = (volume_ai['AI_Present_Count'] / volume_ai['Total_Records'] * 100).round(2)
    volume_ai = volume_ai.reset_index()
    
    fig = px.bar(
        volume_ai,
        x='Volume_Bin',
        y='AI_Percentage',
        title="AI Overview Presence by Search Volume Range (%)",
        color='AI_Percentage',
        color_continuous_scale='Oranges',
        text='AI_Percentage'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=700, showlegend=False)
    return fig, volume_ai

def create_ai_overview_trends(df):
    """Create AI Overview trends over time by country"""
    if 'Month' not in df.columns or df['Month'].isna().all():
        return None
    
    # Group by month and country
    monthly_trends = df.groupby(['Month', 'country']).agg({
        'AI Overview presence': ['count', 'sum']
    }).round(2)
    
    monthly_trends.columns = ['Total_Records', 'AI_Present_Count']
    monthly_trends['AI_Percentage'] = (monthly_trends['AI_Present_Count'] / monthly_trends['Total_Records'] * 100).round(2)
    monthly_trends = monthly_trends.reset_index()
    
    fig = px.line(
        monthly_trends,
        x='Month',
        y='AI_Percentage',
        color='country',
        title="AI Overview Presence Trends Over Time by Country (%)",
        markers=True
    )
    fig.update_layout(height=700, xaxis_tickangle=-45)
    return fig

def create_position_type_analysis(df):
    """Create analysis by Position Type"""
    if 'Position Type' not in df.columns:
        return None, None
    
    pos_type_ai = df.groupby('Position Type').agg({
        'AI Overview presence': ['count', 'sum']
    }).round(2)
    
    pos_type_ai.columns = ['Total_Records', 'AI_Present_Count']
    pos_type_ai['AI_Percentage'] = (pos_type_ai['AI_Present_Count'] / pos_type_ai['Total_Records'] * 100).round(2)
    pos_type_ai = pos_type_ai.reset_index().sort_values('AI_Percentage', ascending=False)
    
    fig = px.bar(
        pos_type_ai,
        x='Position Type',
        y='AI_Percentage',
        title="AI Overview Presence by Position Type (%)",
        color='AI_Percentage',
        color_continuous_scale='Purples',
        text='AI_Percentage'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=700, showlegend=False, xaxis_tickangle=-45)
    return fig, pos_type_ai

def create_keyword_intents_analysis(df):
    """Create analysis by Keyword Intents"""
    if 'Keyword Intents' not in df.columns:
        return None, None
    
    intent_ai = df.groupby('Keyword Intents').agg({
        'AI Overview presence': ['count', 'sum']
    }).round(2)
    
    intent_ai.columns = ['Total_Records', 'AI_Present_Count']
    intent_ai['AI_Percentage'] = (intent_ai['AI_Present_Count'] / intent_ai['Total_Records'] * 100).round(2)
    intent_ai = intent_ai.reset_index().sort_values('AI_Percentage', ascending=False)
    
    fig = px.bar(
        intent_ai,
        x='Keyword Intents',
        y='AI_Percentage',
        title="AI Overview Presence by Keyword Intent (%)",
        color='AI_Percentage',
        color_continuous_scale='Viridis',
        text='AI_Percentage'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=700, showlegend=False, xaxis_tickangle=-45)
    return fig, intent_ai

def create_serp_features_analysis(df):
    """Create analysis by SERP Features"""
    if 'SERP Features by Keyword' not in df.columns:
        return None, None
    
    # Get top 15 SERP feature combinations
    top_features = df['SERP Features by Keyword'].value_counts().head(15).index
    df_filtered = df[df['SERP Features by Keyword'].isin(top_features)]
    
    serp_ai = df_filtered.groupby('SERP Features by Keyword').agg({
        'AI Overview presence': ['count', 'sum']
    }).round(2)
    
    serp_ai.columns = ['Total_Records', 'AI_Present_Count']
    serp_ai['AI_Percentage'] = (serp_ai['AI_Present_Count'] / serp_ai['Total_Records'] * 100).round(2)
    serp_ai = serp_ai.reset_index().sort_values('AI_Percentage', ascending=False)
    
    # Truncate long feature names for display
    serp_ai['SERP_Features_Short'] = serp_ai['SERP Features by Keyword'].str[:50] + '...'
    
    fig = px.bar(
        serp_ai,
        x='SERP_Features_Short',
        y='AI_Percentage',
        title="AI Overview Presence by SERP Features (Top 15) (%)",
        color='AI_Percentage',
        color_continuous_scale='Plasma',
        text='AI_Percentage'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=700, showlegend=False, xaxis_tickangle=-45)
    return fig, serp_ai

def create_sos_query_type_analysis(df):
    """Create analysis by SOS_query_type - both calculations"""
    if 'SOS_query_type' not in df.columns:
        return None, None, None
    
    # Calculation 1: AI Overview rate per query type
    query_type_rates = df.groupby('SOS_query_type').agg({
        'AI Overview presence': ['count', 'sum']
    }).round(2)
    
    query_type_rates.columns = ['Total_Records', 'AI_Present_Count']
    query_type_rates['AI_Rate_Percent'] = (query_type_rates['AI_Present_Count'] / query_type_rates['Total_Records'] * 100).round(2)
    query_type_rates = query_type_rates.reset_index().sort_values('AI_Rate_Percent', ascending=False)
    
    # Calculation 2: AI Overview distribution
    ai_only_df = df[df['AI Overview presence'] == True]
    query_type_distribution = ai_only_df['SOS_query_type'].value_counts(normalize=True) * 100
    query_type_distribution_df = query_type_distribution.reset_index()
    query_type_distribution_df.columns = ['SOS_query_type', 'AI_Distribution_Percent']
    query_type_distribution_df['AI_Distribution_Percent'] = query_type_distribution_df['AI_Distribution_Percent'].round(1)
    
    # Merge both calculations
    query_type_combined = query_type_rates.merge(query_type_distribution_df, on='SOS_query_type', how='left')
    
    # Create visualizations
    fig1 = px.bar(
        query_type_rates,
        x='SOS_query_type',
        y='AI_Rate_Percent',
        title="AI Overview Rate by SOS Query Type<br><sub>What % of each query type's records have AI Overview</sub>",
        color='AI_Rate_Percent',
        color_continuous_scale='Blues',
        text='AI_Rate_Percent'
    )
    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig1.update_layout(height=700, showlegend=False)
    
    fig2 = px.bar(
        query_type_distribution_df,
        x='SOS_query_type',
        y='AI_Distribution_Percent',
        title="AI Overview Distribution by SOS Query Type<br><sub>Of all AI Overview records, what % belong to each query type</sub>",
        color='AI_Distribution_Percent',
        color_continuous_scale='Greens', 
        text='AI_Distribution_Percent'
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(height=700, showlegend=False)
    
    return fig1, fig2, query_type_combined

def create_sos_category_analysis(df):
    """Create analysis by SOS_category - both calculations"""
    if 'SOS_category' not in df.columns:
        return None, None, None
    
    # Calculation 1: AI Overview rate per category
    category_rates = df.groupby('SOS_category').agg({
        'AI Overview presence': ['count', 'sum']
    }).round(2)
    
    category_rates.columns = ['Total_Records', 'AI_Present_Count']
    category_rates['AI_Rate_Percent'] = (category_rates['AI_Present_Count'] / category_rates['Total_Records'] * 100).round(2)
    category_rates = category_rates.reset_index().sort_values('AI_Rate_Percent', ascending=False)
    
    # Calculation 2: AI Overview distribution
    ai_only_df = df[df['AI Overview presence'] == True]
    category_distribution = ai_only_df['SOS_category'].value_counts(normalize=True) * 100
    category_distribution_df = category_distribution.reset_index()
    category_distribution_df.columns = ['SOS_category', 'AI_Distribution_Percent']
    category_distribution_df['AI_Distribution_Percent'] = category_distribution_df['AI_Distribution_Percent'].round(1)
    
    # Merge both calculations
    category_combined = category_rates.merge(category_distribution_df, on='SOS_category', how='left')
    
    # Create visualizations
    fig1 = px.bar(
        category_rates,
        x='SOS_category',
        y='AI_Rate_Percent',
        title="AI Overview Rate by SOS Category<br><sub>What % of each category's records have AI Overview</sub>",
        color='AI_Rate_Percent',
        color_continuous_scale='Reds',
        text='AI_Rate_Percent'
    )
    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig1.update_layout(height=700, showlegend=False, xaxis_tickangle=-45)
    
    fig2 = px.bar(
        category_distribution_df,
        x='SOS_category',
        y='AI_Distribution_Percent',
        title="AI Overview Distribution by SOS Category<br><sub>Of all AI Overview records, what % belong to each category</sub>",
        color='AI_Distribution_Percent',
        color_continuous_scale='Oranges',
        text='AI_Distribution_Percent'
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(height=700, showlegend=False, xaxis_tickangle=-45)
    
    return fig1, fig2, category_combined

def create_sos_territory_analysis(df):
    """Create analysis by SOS_territory - handles pipe-separated values"""
    if 'SOS_territory' not in df.columns:
        return None, None, None
    
    # Expand pipe-separated territories
    expanded_rows = []
    for _, row in df.iterrows():
        territories = str(row['SOS_territory']).split('|') if pd.notna(row['SOS_territory']) else ['Unknown']
        for territory in territories:
            territory = territory.strip()
            if territory:  # Skip empty strings
                new_row = row.copy()
                new_row['SOS_territory_expanded'] = territory
                expanded_rows.append(new_row)
    
    if not expanded_rows:
        return None, None, None
        
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Calculation 1: AI Overview rate per territory
    territory_rates = expanded_df.groupby('SOS_territory_expanded').agg({
        'AI Overview presence': ['count', 'sum']
    }).round(2)
    
    territory_rates.columns = ['Total_Records', 'AI_Present_Count']
    territory_rates['AI_Rate_Percent'] = (territory_rates['AI_Present_Count'] / territory_rates['Total_Records'] * 100).round(2)
    territory_rates = territory_rates.reset_index().sort_values('AI_Rate_Percent', ascending=False)
    
    # Get top 15 territories by record count
    top_territories = territory_rates.nlargest(15, 'Total_Records')
    
    # Calculation 2: AI Overview distribution for top territories
    ai_only_expanded = expanded_df[expanded_df['AI Overview presence'] == True]
    territory_distribution = ai_only_expanded['SOS_territory_expanded'].value_counts(normalize=True) * 100
    territory_distribution_df = territory_distribution.reset_index()
    territory_distribution_df.columns = ['SOS_territory_expanded', 'AI_Distribution_Percent']
    territory_distribution_df['AI_Distribution_Percent'] = territory_distribution_df['AI_Distribution_Percent'].round(1)
    
    # Filter to top territories for distribution too
    territory_distribution_top = territory_distribution_df[
        territory_distribution_df['SOS_territory_expanded'].isin(top_territories['SOS_territory_expanded'])
    ]
    
    # Merge both calculations
    territory_combined = top_territories.merge(territory_distribution_df, on='SOS_territory_expanded', how='left')
    
    # Create visualizations
    fig1 = px.bar(
        top_territories,
        x='SOS_territory_expanded',
        y='AI_Rate_Percent',
        title="AI Overview Rate by SOS Territory (Top 15)<br><sub>What % of each territory's records have AI Overview</sub>",
        color='AI_Rate_Percent',
        color_continuous_scale='Purples',
        text='AI_Rate_Percent'
    )
    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig1.update_layout(height=700, showlegend=False, xaxis_tickangle=-45)
    
    fig2 = px.bar(
        territory_distribution_top,
        x='SOS_territory_expanded',
        y='AI_Distribution_Percent',
        title="AI Overview Distribution by SOS Territory (Top 15)<br><sub>Of all AI Overview records, what % belong to each territory</sub>",
        color='AI_Distribution_Percent',
        color_continuous_scale='Viridis',
        text='AI_Distribution_Percent'
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(height=700, showlegend=False, xaxis_tickangle=-45)
    
    return fig1, fig2, territory_combined

def create_keyword_in_sos_analysis(df):
    """Create analysis by Keyword_in_SOS_Queries - both calculations"""
    if 'Keyword_in_SOS_Queries' not in df.columns:
        return None, None, None
    
    # Calculation 1: AI Overview rate per Keyword_in_SOS_Queries
    keyword_sos_rates = df.groupby('Keyword_in_SOS_Queries').agg({
        'AI Overview presence': ['count', 'sum']
    }).round(2)
    
    keyword_sos_rates.columns = ['Total_Records', 'AI_Present_Count']
    keyword_sos_rates['AI_Rate_Percent'] = (keyword_sos_rates['AI_Present_Count'] / keyword_sos_rates['Total_Records'] * 100).round(2)
    keyword_sos_rates = keyword_sos_rates.reset_index().sort_values('AI_Rate_Percent', ascending=False)
    
    # Calculation 2: AI Overview distribution
    ai_only_df = df[df['AI Overview presence'] == True]
    keyword_sos_distribution = ai_only_df['Keyword_in_SOS_Queries'].value_counts(normalize=True) * 100
    keyword_sos_distribution_df = keyword_sos_distribution.reset_index()
    keyword_sos_distribution_df.columns = ['Keyword_in_SOS_Queries', 'AI_Distribution_Percent']
    keyword_sos_distribution_df['AI_Distribution_Percent'] = keyword_sos_distribution_df['AI_Distribution_Percent'].round(1)
    
    # Merge both calculations
    keyword_sos_combined = keyword_sos_rates.merge(keyword_sos_distribution_df, on='Keyword_in_SOS_Queries', how='left')
    
    # Create visualizations
    fig1 = px.bar(
        keyword_sos_rates,
        x='Keyword_in_SOS_Queries',
        y='AI_Rate_Percent',
        title="AI Overview Rate by Keyword in SOS Queries<br><sub>What % of each group's records have AI Overview</sub>",
        color='AI_Rate_Percent',
        color_continuous_scale='Teal',
        text='AI_Rate_Percent'
    )
    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig1.update_layout(height=700, showlegend=False)
    
    fig2 = px.bar(
        keyword_sos_distribution_df,
        x='Keyword_in_SOS_Queries',
        y='AI_Distribution_Percent',
        title="AI Overview Distribution by Keyword in SOS Queries<br><sub>Of all AI Overview records, what % belong to each group</sub>",
        color='AI_Distribution_Percent',
        color_continuous_scale='Magma',
        text='AI_Distribution_Percent'
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(height=700, showlegend=False)
    
    return fig1, fig2, keyword_sos_combined

# ===== NEW ANALYSIS FUNCTIONS =====

def create_ai_overview_keyword_complexity_analysis(df):
    """AI Overview presence by keyword length/complexity"""
    if 'Keyword' not in df.columns:
        return None, None
    
    # Calculate keyword metrics
    df_copy = df.copy()
    df_copy['keyword_length'] = df_copy['Keyword'].str.len()
    df_copy['word_count'] = df_copy['Keyword'].str.split().str.len()
    df_copy['complexity_score'] = df_copy['keyword_length'] * df_copy['word_count']
    
    # Create complexity bins
    df_copy['complexity_bin'] = pd.cut(df_copy['complexity_score'], 
                                     bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    complexity_ai = df_copy.groupby('complexity_bin').agg({
        'AI Overview presence': ['count', 'sum']
    })
    complexity_ai.columns = ['Total_Records', 'AI_Present_Count']
    complexity_ai['AI_Percentage'] = (complexity_ai['AI_Present_Count'] / complexity_ai['Total_Records'] * 100).round(2)
    complexity_ai = complexity_ai.reset_index()
    
    fig = px.bar(complexity_ai, x='complexity_bin', y='AI_Percentage',
                 title="AI Overview Presence by Keyword Complexity",
                 color='AI_Percentage', color_continuous_scale='Blues',
                 text='AI_Percentage')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=700, showlegend=False)
    
    return fig, complexity_ai

def create_ai_overview_search_volume_correlation(df):
    """AI Overview correlation with search volume ranges"""
    volume_ranges = [(0, 100), (101, 500), (501, 1000), (1001, 5000), (5001, float('inf'))]
    volume_labels = ['0-100', '101-500', '501-1K', '1K-5K', '5K+']
    
    correlation_data = []
    for i, (min_vol, max_vol) in enumerate(volume_ranges):
        if max_vol == float('inf'):
            subset = df[df['Search Volume'] >= min_vol]
        else:
            subset = df[(df['Search Volume'] >= min_vol) & (df['Search Volume'] <= max_vol)]
        
        if len(subset) > 0:
            ai_rate = (subset['AI Overview presence'].sum() / len(subset)) * 100
            correlation_data.append({
                'Volume_Range': volume_labels[i],
                'AI_Rate': ai_rate,
                'Record_Count': len(subset)
            })
    
    corr_df = pd.DataFrame(correlation_data)
    
    fig = px.scatter(corr_df, x='Volume_Range', y='AI_Rate', size='Record_Count',
                     title="AI Overview Rate vs Search Volume Ranges",
                     hover_data=['Record_Count'])
    fig.update_layout(height=700)
    
    return fig, corr_df

def create_ai_overview_day_of_week_analysis(df):
    """AI Overview by day of week/seasonality patterns"""
    if 'Month' not in df.columns or df['Month'].isna().all():
        return None, None
    
    df_copy = df.copy()
    df_copy['day_of_week'] = pd.to_datetime(df_copy['Month']).dt.day_name()
    df_copy['month_name'] = pd.to_datetime(df_copy['Month']).dt.month_name()
    
    # Day of week analysis
    dow_ai = df_copy.groupby('day_of_week').agg({
        'AI Overview presence': ['count', 'sum']
    })
    dow_ai.columns = ['Total_Records', 'AI_Present_Count']
    dow_ai['AI_Percentage'] = (dow_ai['AI_Present_Count'] / dow_ai['Total_Records'] * 100).round(2)
    dow_ai = dow_ai.reset_index()
    
    # Order by days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_ai['day_of_week'] = pd.Categorical(dow_ai['day_of_week'], categories=day_order, ordered=True)
    dow_ai = dow_ai.sort_values('day_of_week')
    
    fig = px.bar(dow_ai, x='day_of_week', y='AI_Percentage',
                 title="AI Overview Presence by Day of Week",
                 color='AI_Percentage', color_continuous_scale='Viridis',
                 text='AI_Percentage')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=700, showlegend=False)
    
    return fig, dow_ai

def create_ai_overview_position_scatter(df):
    """AI Overview vs organic position scatter plot"""
    sample_df = df.sample(min(10000, len(df)), random_state=42)  # Sample for performance
    
    fig = px.scatter(sample_df, x='Position', y='AI Overview presence',
                     color='brand', opacity=0.6,
                     title="AI Overview vs Organic Position Scatter Plot",
                     hover_data=['Search Volume', 'Traffic'])
    fig.update_layout(height=700)
    
    return fig

def create_position_distribution_histograms(df):
    """Position distribution histograms by brand"""
    fig = px.histogram(df, x='Position', color='brand', 
                       title="Position Distribution by Brand",
                       nbins=50, opacity=0.7)
    fig.update_layout(height=700, barmode='overlay')
    
    return fig

def create_traffic_position_correlation(df):
    """Traffic vs Position correlation charts"""
    if 'Traffic' not in df.columns:
        return None, None
    
    # Sample for performance
    sample_df = df.sample(min(5000, len(df)), random_state=42)
    
    fig = px.scatter(sample_df, x='Position', y='Traffic', color='brand',
                     title="Traffic vs Position Correlation",
                     hover_data=['Search Volume', 'AI Overview presence'],
                     opacity=0.6)
    fig.update_layout(height=700)
    
    # Calculate correlation coefficient
    correlation_stats = None
    if HAS_SCIPY:
        valid_data = sample_df[['Position', 'Traffic']].dropna()
        if len(valid_data) > 1:
            corr_coef, p_value = pearsonr(valid_data['Position'], valid_data['Traffic'])
            correlation_stats = {'correlation': corr_coef, 'p_value': p_value}
    
    return fig, correlation_stats

def create_serp_features_network(df):
    """SERP features co-occurrence network"""
    if 'SERP Features by Keyword' not in df.columns:
        return None
    
    # Get top 20 most common SERP features for network analysis
    top_features = df['SERP Features by Keyword'].value_counts().head(20).index
    df_filtered = df[df['SERP Features by Keyword'].isin(top_features)]
    
    # Create co-occurrence matrix (simplified for visualization)
    feature_counts = df_filtered['SERP Features by Keyword'].value_counts()
    
    # Create a simple network visualization using plotly
    fig = px.bar(feature_counts.head(15), 
                 title="Top SERP Features Co-occurrence",
                 labels={'index': 'SERP Features', 'value': 'Frequency'})
    fig.update_layout(height=700, xaxis_tickangle=-45)
    
    return fig

def create_position_type_performance(df):
    """Position Type performance comparison"""
    if 'Position Type' not in df.columns:
        return None, None
    
    pos_type_stats = df.groupby('Position Type').agg({
        'Position': 'mean',
        'Search Volume': 'mean',
        'Traffic': 'mean',
        'AI Overview presence': lambda x: (x.sum() / len(x)) * 100
    }).round(2)
    
    pos_type_stats.columns = ['Avg_Position', 'Avg_Search_Volume', 'Avg_Traffic', 'AI_Overview_Rate']
    pos_type_stats = pos_type_stats.reset_index()
    
    # Create multi-metric comparison
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=pos_type_stats['Position Type'], y=pos_type_stats['Avg_Position'],
               name='Avg Position', marker_color='lightblue'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=pos_type_stats['Position Type'], y=pos_type_stats['AI_Overview_Rate'],
                   mode='lines+markers', name='AI Overview Rate (%)', marker_color='red'),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Position Type")
    fig.update_yaxes(title_text="Average Position", secondary_y=False)
    fig.update_yaxes(title_text="AI Overview Rate (%)", secondary_y=True)
    fig.update_layout(title_text="Position Type Performance Comparison", height=700)
    
    return fig, pos_type_stats

def create_brand_market_share_over_time(df):
    """Brand market share over time"""
    if 'Month' not in df.columns or df['Month'].isna().all():
        return None, None
    
    monthly_brand_share = df.groupby(['Month', 'brand']).size().reset_index(name='Record_Count')
    monthly_totals = monthly_brand_share.groupby('Month')['Record_Count'].sum().reset_index()
    monthly_totals.columns = ['Month', 'Total_Records']
    
    monthly_brand_share = monthly_brand_share.merge(monthly_totals, on='Month')
    monthly_brand_share['Market_Share'] = (monthly_brand_share['Record_Count'] / monthly_brand_share['Total_Records']) * 100
    
    fig = px.line(monthly_brand_share, x='Month', y='Market_Share', color='brand',
                  title="Brand Market Share Over Time (%)",
                  markers=True)
    fig.update_layout(height=700)
    
    return fig, monthly_brand_share

def create_competitive_position_analysis(df):
    """Competitive position analysis"""
    brand_position_stats = df.groupby('brand').agg({
        'Position': ['mean', 'median', 'std'],
        'Search Volume': 'sum',
        'Traffic': 'sum',
        'AI Overview presence': lambda x: (x.sum() / len(x)) * 100
    }).round(2)
    
    brand_position_stats.columns = ['Avg_Position', 'Median_Position', 'Position_Std', 
                                   'Total_Search_Volume', 'Total_Traffic', 'AI_Overview_Rate']
    brand_position_stats = brand_position_stats.reset_index()
    
    fig = px.scatter(brand_position_stats, x='Avg_Position', y='AI_Overview_Rate',
                     size='Total_Search_Volume', color='brand',
                     title="Competitive Position Analysis (Avg Position vs AI Overview Rate)",
                     hover_data=['Median_Position', 'Total_Traffic'])
    fig.update_layout(height=700)
    
    return fig, brand_position_stats

def create_search_volume_trends_by_brand(df):
    """Search volume trends by brand"""
    if 'Month' not in df.columns or df['Month'].isna().all():
        return None, None
    
    monthly_volume = df.groupby(['Month', 'brand'])['Search Volume'].sum().reset_index()
    
    fig = px.line(monthly_volume, x='Month', y='Search Volume', color='brand',
                  title="Search Volume Trends by Brand",
                  markers=True)
    fig.update_layout(height=700)
    
    return fig, monthly_volume

def create_geographic_heatmap(df):
    """Geographic heat maps"""
    country_stats = df.groupby('country').agg({
        'AI Overview presence': lambda x: (x.sum() / len(x)) * 100,
        'Position': 'mean',
        'Search Volume': 'sum'
    }).round(2)
    
    country_stats.columns = ['AI_Overview_Rate', 'Avg_Position', 'Total_Search_Volume']
    country_stats = country_stats.reset_index()
    
    fig = px.choropleth(country_stats, 
                        locations='country',
                        color='AI_Overview_Rate',
                        hover_data=['Avg_Position', 'Total_Search_Volume'],
                        color_continuous_scale='Viridis',
                        title="AI Overview Rate by Country (Geographic Heatmap)")
    fig.update_layout(height=700)
    
    return fig, country_stats

def create_keyword_intent_performance_matrix(df):
    """Keyword intent performance matrix"""
    if 'Keyword Intents' not in df.columns:
        return None, None
    
    intent_matrix = df.groupby('Keyword Intents').agg({
        'Position': 'mean',
        'Search Volume': 'mean',
        'Traffic': 'mean',
        'AI Overview presence': lambda x: (x.sum() / len(x)) * 100
    }).round(2)
    
    intent_matrix.columns = ['Avg_Position', 'Avg_Search_Volume', 'Avg_Traffic', 'AI_Overview_Rate']
    intent_matrix = intent_matrix.reset_index()
    
    fig = px.scatter(intent_matrix, x='Avg_Position', y='AI_Overview_Rate',
                     size='Avg_Search_Volume', color='Keyword Intents',
                     title="Keyword Intent Performance Matrix",
                     hover_data=['Avg_Traffic'])
    fig.update_layout(height=700)
    
    return fig, intent_matrix

def create_traffic_efficiency_by_position(df):
    """Traffic efficiency by position ranges"""
    if 'Traffic' not in df.columns:
        return None, None
    
    df_copy = df.copy()
    df_copy['Position_Range'] = pd.cut(df_copy['Position'], 
                                     bins=[0, 3, 10, 20, 50, 100],
                                     labels=['1-3', '4-10', '11-20', '21-50', '51-100'])
    
    efficiency_stats = df_copy.groupby('Position_Range').agg({
        'Traffic': 'sum',
        'Search Volume': 'sum'
    })
    efficiency_stats['Traffic_Efficiency'] = (efficiency_stats['Traffic'] / efficiency_stats['Search Volume']) * 100
    efficiency_stats = efficiency_stats.reset_index()
    
    fig = px.bar(efficiency_stats, x='Position_Range', y='Traffic_Efficiency',
                 title="Traffic Efficiency by Position Range (%)",
                 color='Traffic_Efficiency', color_continuous_scale='Blues',
                 text='Traffic_Efficiency')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=700, showlegend=False)
    
    return fig, efficiency_stats

def create_brand_cannibalization_analysis(df):
    """Brand cannibalization analysis"""
    brand_keyword_overlap = df.groupby('brand')['Keyword'].nunique().reset_index()
    brand_keyword_overlap.columns = ['brand', 'Unique_Keywords']
    
    total_keywords = df['Keyword'].nunique()
    brand_keyword_overlap['Keyword_Share'] = (brand_keyword_overlap['Unique_Keywords'] / total_keywords) * 100
    
    fig = px.pie(brand_keyword_overlap, values='Keyword_Share', names='brand',
                 title="Brand Keyword Share (Potential Cannibalization)")
    fig.update_layout(height=700)
    
    return fig, brand_keyword_overlap

def create_seasonal_trend_decomposition(df):
    """Seasonal trend decomposition"""
    if 'Month' not in df.columns or df['Month'].isna().all():
        return None, None
    
    monthly_ai_rate = df.groupby('Month').agg({
        'AI Overview presence': lambda x: (x.sum() / len(x)) * 100
    }).reset_index()
    monthly_ai_rate.columns = ['Month', 'AI_Rate']
    
    if len(monthly_ai_rate) < 12:  # Need at least 12 months for seasonal decomposition
        fig = px.line(monthly_ai_rate, x='Month', y='AI_Rate',
                      title="AI Overview Rate Trend (Insufficient data for seasonal decomposition)")
        fig.update_layout(height=700)
        return fig, monthly_ai_rate
    
    # Simple trend analysis instead of full seasonal decomposition
    monthly_ai_rate['Month_Num'] = pd.to_datetime(monthly_ai_rate['Month']).dt.month
    monthly_seasonal = monthly_ai_rate.groupby('Month_Num')['AI_Rate'].mean().reset_index()
    
    fig = px.line(monthly_seasonal, x='Month_Num', y='AI_Rate',
                  title="Seasonal AI Overview Rate Pattern (Average by Month)",
                  markers=True)
    fig.update_layout(height=700)
    fig.update_xaxes(title="Month of Year")
    
    return fig, monthly_seasonal

def main():
    
    st.title("🤖 AI Overview Analytics Dashboard")
    st.markdown("Comprehensive analysis of AI Overview presence across brands, countries, and search characteristics")
    
    # Sidebar for controls
    st.sidebar.header("Dashboard Controls")
    
    # Load optimized sample for Streamlit Cloud
    st.sidebar.info("📊 Optimized sample (500K records)")
    st.sidebar.info("🎯 Preserves all key insights")
    st.sidebar.info("⚡ Fast loading on Streamlit Cloud")
    
    # Load data
    with st.spinner("Loading optimized sample (500K records)..."):
        df = load_data(sample_size=None)
    
    if df.empty:
        st.error("Failed to load data. Please check the file path.")
        return
    
    # Calculate AI Overview stats
    stats = get_ai_overview_stats(df)
    
    # Display key metrics
    st.header("🎯 AI Overview Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{stats['total_records']:,}")
    with col2:
        st.metric("AI Overview Present", f"{stats['ai_present_count']:,}")
    with col3:
        st.metric("AI Overview Rate", f"{stats['ai_present_percentage']:.2f}%")
    with col4:
        st.metric("Brands with AI", stats['brands_with_ai'])
    with col5:
        st.metric("Countries with AI", stats['countries_with_ai'])
    
    col6, col7 = st.columns(2)
    with col6:
        st.metric("Avg Position (with AI)", f"{stats['avg_position_with_ai']:.1f}" if not pd.isna(stats['avg_position_with_ai']) else "N/A")
    with col7:
        st.metric("Avg Position (without AI)", f"{stats['avg_position_without_ai']:.1f}" if not pd.isna(stats['avg_position_without_ai']) else "N/A")
    
    # Filters in sidebar
    st.sidebar.header("Filters")
    
    # Brand filter
    available_brands = sorted(df['brand'].unique())
    selected_brands = st.sidebar.multiselect("Select Brands", available_brands, default=available_brands)
    
    # Country filter
    available_countries = sorted(df['country'].unique())
    selected_countries = st.sidebar.multiselect("Select Countries", available_countries, default=available_countries)
    
    # Apply filters
    filtered_df = df[
        (df['brand'].isin(selected_brands)) & 
        (df['country'].isin(selected_countries))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return
    
    # Main AI Overview Analysis
    st.header("🔍 AI Overview Presence Analysis")
    
    # Brand Analysis - Both Calculations
    st.subheader("📊 AI Overview by Brand")
    st.info("💡 **Two Different Calculations Explained:**\n"
            "- **Rate**: What % of each brand's records have AI Overview\n"
            "- **Distribution**: Of all AI Overview records, what % belong to each brand")
    
    brand_fig1, brand_fig2, brand_data = create_ai_overview_by_brand(filtered_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(brand_fig1, use_container_width=True)
    with col2:
        st.plotly_chart(brand_fig2, use_container_width=True)
    
    with st.expander("📋 Brand Analysis Data Table"):
        st.dataframe(brand_data, use_container_width=True)
    
    # Country Analysis - Both Calculations  
    st.subheader("🌍 AI Overview by Country")
    st.info("💡 **Two Different Calculations Explained:**\n"
            "- **Rate**: What % of each country's records have AI Overview\n"
            "- **Distribution**: Of all AI Overview records, what % belong to each country")
    
    country_fig1, country_fig2, country_data = create_ai_overview_by_country(filtered_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(country_fig1, use_container_width=True)
    with col2:
        st.plotly_chart(country_fig2, use_container_width=True)
        
    with st.expander("📋 Country Analysis Data Table"):
        st.dataframe(country_data, use_container_width=True)
    
    # Position and Search Volume Analysis
    st.header("📊 Position & Search Volume Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("By SERP Position")
        position_fig, position_data = create_ai_overview_by_position(filtered_df)
        st.plotly_chart(position_fig, use_container_width=True)
        with st.expander("Position Data Table"):
            st.dataframe(position_data, use_container_width=True)
    
    with col2:
        st.subheader("By Search Volume")
        volume_fig, volume_data = create_ai_overview_search_volume_analysis(filtered_df)
        st.plotly_chart(volume_fig, use_container_width=True)
        with st.expander("Search Volume Data Table"):
            st.dataframe(volume_data, use_container_width=True)
    
    # Time Trends Analysis
    st.header("📈 AI Overview Trends Over Time")
    trends_fig = create_ai_overview_trends(filtered_df)
    if trends_fig:
        st.plotly_chart(trends_fig, use_container_width=True)
    else:
        st.info("Time trend data not available or insufficient data.")
    
    # Additional Analysis
    st.header("🎯 Advanced SERP Analysis")
    
    # Position Type Analysis
    pos_type_fig, pos_type_data = create_position_type_analysis(filtered_df)
    if pos_type_fig:
        st.subheader("By Position Type")
        st.plotly_chart(pos_type_fig, use_container_width=True)
        with st.expander("Position Type Data Table"):
            st.dataframe(pos_type_data, use_container_width=True)
    
    # Keyword Intents Analysis
    intent_fig, intent_data = create_keyword_intents_analysis(filtered_df)
    if intent_fig:
        st.subheader("By Keyword Intent")
        st.plotly_chart(intent_fig, use_container_width=True)
        with st.expander("Keyword Intent Data Table"):
            st.dataframe(intent_data, use_container_width=True)
    
    # SERP Features Analysis
    serp_fig, serp_data = create_serp_features_analysis(filtered_df)
    if serp_fig:
        st.subheader("By SERP Features")
        st.plotly_chart(serp_fig, use_container_width=True)
        with st.expander("SERP Features Data Table"):
            st.dataframe(serp_data, use_container_width=True)
    
    # New SOS Analysis Section
    st.header("🎯 SOS (Share of Search) Analysis")
    
    # SOS Query Type Analysis
    sos_query_fig1, sos_query_fig2, sos_query_data = create_sos_query_type_analysis(filtered_df)
    if sos_query_fig1:
        st.subheader("📋 AI Overview by SOS Query Type")
        st.info("💡 **Two Different Calculations Explained:**\n"
                "- **Rate**: What % of each query type's records have AI Overview\n"
                "- **Distribution**: Of all AI Overview records, what % belong to each query type")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(sos_query_fig1, use_container_width=True)
        with col2:
            st.plotly_chart(sos_query_fig2, use_container_width=True)
        
        with st.expander("📋 SOS Query Type Data Table"):
            st.dataframe(sos_query_data, use_container_width=True)
    
    # SOS Category Analysis
    sos_category_fig1, sos_category_fig2, sos_category_data = create_sos_category_analysis(filtered_df)
    if sos_category_fig1:
        st.subheader("🏷️ AI Overview by SOS Category")
        st.info("💡 **Two Different Calculations Explained:**\n"
                "- **Rate**: What % of each category's records have AI Overview\n"
                "- **Distribution**: Of all AI Overview records, what % belong to each category")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(sos_category_fig1, use_container_width=True)
        with col2:
            st.plotly_chart(sos_category_fig2, use_container_width=True)
        
        with st.expander("📋 SOS Category Data Table"):
            st.dataframe(sos_category_data, use_container_width=True)
    
    # SOS Territory Analysis (handles pipe-separated values)
    sos_territory_fig1, sos_territory_fig2, sos_territory_data = create_sos_territory_analysis(filtered_df)
    if sos_territory_fig1:
        st.subheader("🗺️ AI Overview by SOS Territory")
        st.info("💡 **Two Different Calculations Explained:**\n"
                "- **Rate**: What % of each territory's records have AI Overview\n"
                "- **Distribution**: Of all AI Overview records, what % belong to each territory\n"
                "- **Note**: Pipe-separated territories are expanded (one record can count for multiple territories)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(sos_territory_fig1, use_container_width=True)
        with col2:
            st.plotly_chart(sos_territory_fig2, use_container_width=True)
        
        with st.expander("📋 SOS Territory Data Table"):
            st.dataframe(sos_territory_data, use_container_width=True)
    
    # Keyword in SOS Queries Analysis
    keyword_sos_fig1, keyword_sos_fig2, keyword_sos_data = create_keyword_in_sos_analysis(filtered_df)
    if keyword_sos_fig1:
        st.subheader("🔍 AI Overview by Keyword in SOS Queries")
        st.info("💡 **Two Different Calculations Explained:**\n"
                "- **Rate**: What % of each group's records have AI Overview (True vs False)\n"
                "- **Distribution**: Of all AI Overview records, what % belong to each group")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(keyword_sos_fig1, use_container_width=True)
        with col2:
            st.plotly_chart(keyword_sos_fig2, use_container_width=True)
        
        with st.expander("📋 Keyword in SOS Queries Data Table"):
            st.dataframe(keyword_sos_data, use_container_width=True)
    
    # ===== NEW DASHBOARD SECTIONS =====
    
    # AI Overview Focus Section
    st.header("🤖 AI Overview Focus")
    st.markdown("**Detailed AI Overview analysis with advanced metrics**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("By Keyword Complexity")
        complexity_fig, complexity_data = create_ai_overview_keyword_complexity_analysis(filtered_df)
        if complexity_fig:
            st.plotly_chart(complexity_fig, use_container_width=True)
            with st.expander("Keyword Complexity Data"):
                st.dataframe(complexity_data, use_container_width=True)
        
        st.subheader("Day of Week Patterns")
        dow_fig, dow_data = create_ai_overview_day_of_week_analysis(filtered_df)
        if dow_fig:
            st.plotly_chart(dow_fig, use_container_width=True)
            with st.expander("Day of Week Data"):
                st.dataframe(dow_data, use_container_width=True)
    
    with col2:
        st.subheader("Search Volume Correlation")
        corr_fig, corr_data = create_ai_overview_search_volume_correlation(filtered_df)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
            with st.expander("Correlation Data"):
                st.dataframe(corr_data, use_container_width=True)
        
        st.subheader("Position Scatter Plot")
        scatter_fig = create_ai_overview_position_scatter(filtered_df)
        if scatter_fig:
            st.plotly_chart(scatter_fig, use_container_width=True)
    
    # SERP Analysis Section
    st.header("📊 SERP Analysis")
    st.markdown("**Comprehensive SERP feature and position analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Position Distribution by Brand")
        hist_fig = create_position_distribution_histograms(filtered_df)
        if hist_fig:
            st.plotly_chart(hist_fig, use_container_width=True)
        
        st.subheader("SERP Features Network")
        network_fig = create_serp_features_network(filtered_df)
        if network_fig:
            st.plotly_chart(network_fig, use_container_width=True)
    
    with col2:
        st.subheader("Traffic vs Position Correlation")
        traffic_fig, traffic_corr = create_traffic_position_correlation(filtered_df)
        if traffic_fig:
            st.plotly_chart(traffic_fig, use_container_width=True)
            if traffic_corr:
                st.info(f"**Correlation Coefficient:** {traffic_corr['correlation']:.3f} (p-value: {traffic_corr['p_value']:.3f})")
        
        st.subheader("Position Type Performance")
        pos_type_fig, pos_type_data = create_position_type_performance(filtered_df)
        if pos_type_fig:
            st.plotly_chart(pos_type_fig, use_container_width=True)
            with st.expander("Position Type Performance Data"):
                st.dataframe(pos_type_data, use_container_width=True)
    
    # Business Intelligence Section
    st.header("📈 Business Intelligence")
    st.markdown("**Market share, competitive analysis, and business insights**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Brand Market Share Over Time")
        market_share_fig, market_share_data = create_brand_market_share_over_time(filtered_df)
        if market_share_fig:
            st.plotly_chart(market_share_fig, use_container_width=True)
            with st.expander("Market Share Data"):
                st.dataframe(market_share_data, use_container_width=True)
        
        st.subheader("Search Volume Trends by Brand")
        volume_trends_fig, volume_trends_data = create_search_volume_trends_by_brand(filtered_df)
        if volume_trends_fig:
            st.plotly_chart(volume_trends_fig, use_container_width=True)
            with st.expander("Search Volume Trends Data"):
                st.dataframe(volume_trends_data, use_container_width=True)
    
    with col2:
        st.subheader("Competitive Position Analysis")
        comp_analysis_fig, comp_analysis_data = create_competitive_position_analysis(filtered_df)
        if comp_analysis_fig:
            st.plotly_chart(comp_analysis_fig, use_container_width=True)
            with st.expander("Competitive Analysis Data"):
                st.dataframe(comp_analysis_data, use_container_width=True)
        
        st.subheader("Geographic Heatmap")
        geo_fig, geo_data = create_geographic_heatmap(filtered_df)
        if geo_fig:
            st.plotly_chart(geo_fig, use_container_width=True)
            with st.expander("Geographic Data"):
                st.dataframe(geo_data, use_container_width=True)
    
    # Advanced Analytics Section
    st.header("🔬 Advanced Analytics")
    st.markdown("**Advanced performance metrics and trend analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Keyword Intent Performance Matrix")
        intent_matrix_fig, intent_matrix_data = create_keyword_intent_performance_matrix(filtered_df)
        if intent_matrix_fig:
            st.plotly_chart(intent_matrix_fig, use_container_width=True)
            with st.expander("Intent Performance Data"):
                st.dataframe(intent_matrix_data, use_container_width=True)
        
        st.subheader("Brand Cannibalization Analysis")
        cannib_fig, cannib_data = create_brand_cannibalization_analysis(filtered_df)
        if cannib_fig:
            st.plotly_chart(cannib_fig, use_container_width=True)
            with st.expander("Cannibalization Data"):
                st.dataframe(cannib_data, use_container_width=True)
    
    with col2:
        st.subheader("Traffic Efficiency by Position")
        efficiency_fig, efficiency_data = create_traffic_efficiency_by_position(filtered_df)
        if efficiency_fig:
            st.plotly_chart(efficiency_fig, use_container_width=True)
            with st.expander("Traffic Efficiency Data"):
                st.dataframe(efficiency_data, use_container_width=True)
        
        st.subheader("Seasonal Trend Decomposition")
        seasonal_fig, seasonal_data = create_seasonal_trend_decomposition(filtered_df)
        if seasonal_fig:
            st.plotly_chart(seasonal_fig, use_container_width=True)
            with st.expander("Seasonal Data"):
                st.dataframe(seasonal_data, use_container_width=True)
    
    # Download filtered data
    st.header("💾 Data Export")
    if st.button("Download AI Overview Data as CSV"):
        ai_data = filtered_df[filtered_df['AI Overview presence']]
        csv = ai_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"ai_overview_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()