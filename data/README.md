# Data Directory

## Optimized Sample Data

This dashboard uses a **smart sample** optimized for Streamlit Community Cloud deployment.

### Current Configuration
- **Data Source**: Local sample file (`streamlit_sample.parquet`)
- **Sample Size**: 500K records (3.26% of original 15.3M)
- **File Size**: 19.8MB (perfect for Streamlit Cloud)
- **Load Time**: 5-10 seconds (very fast)

### Data Structure
The parquet file contains these columns:
- `AI Overview presence` (boolean)
- `brand` (string) - Garnier, L'Oréal Paris, La Roche-Posay, Vichy, Maybelline
- `country` (string) - Germany, Spain, Italy, US, Brazil, Mexico, UK, France, India, Poland, Japan
- `Position` (numeric) - SERP ranking position (1-100)
- `Search Volume` (numeric) - Monthly search volume
- `Month` (datetime) - Time period (2024-2025)
- `Position Type` (string) - Organic, People also ask, Image pack, etc.
- `Keyword Intents` (string) - Informational, Commercial, Transactional, etc.
- `SERP Features by Keyword` (string) - SERP features present

### Local Development
For local development, you can:
1. Download the file from Google Drive
2. Place it in this directory
3. Temporarily modify `dashboard.py` to load from local file
4. **Remember to revert before pushing to GitHub**

### Security Note
⚠️ **Local data files are excluded from git commits** via `.gitignore`

# Full data set: 
  ⎿  === DATASET OVERVIEW ===      
     Total rows: 15,348,384
     Total brands: 5
     … +31 lines (ctrl+r to expand)

⏺ 📊 Dataset Summary:

  Overall Statistics:

  - Total rows: 15,348,384
  - Total brands: 5
  - Total countries: 12

  Rows per Brand:

  1. Garnier: 4,583,105 rows (29.9%)
  2. L'Oréal Paris: 4,217,028 rows (27.5%)
  3. La Roche-Posay: 2,690,011 rows (17.5%)
  4. Vichy: 2,214,190 rows (14.4%)
  5. Maybelline: 1,644,050 rows (10.7%)

  Countries per Brand:

  1. Garnier & L'Oréal Paris: 10 countries each
  2. Maybelline & Vichy: 7 countries each
  3. La Roche-Posay: 6 countries

  Brand-Country Coverage:

  🌍 Most Global Brands:
  - Garnier (10 countries): Brazil, France, Germany, India, Italy, Mexico, Poland, Spain, United Kingdom, United States
  - L'Oréal Paris (10 countries): Brazil, Germany, India, Italy, Japan, Mexico, Poland, Spain, United Kingdom, United States

  🎯 Regional Focus Brands:
  - Maybelline (7 countries): France, Germany, India, Mexico, Poland, United States, United States (Spanish)
  - Vichy (7 countries): Brazil, France, Germany, Italy, Mexico, Spain, United Kingdom
  - La Roche-Posay (6 countries): Brazil, Germany, Italy, Mexico, Poland, Spain

  📝 Key Insights:
  - Japan appears only for L'Oréal Paris
  - United States (Spanish) is a separate market segment for Maybelline
  - France appears in most brands except La Roche-Posay
  - Germany is the only country present across all 5 brands