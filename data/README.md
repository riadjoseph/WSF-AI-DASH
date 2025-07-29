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