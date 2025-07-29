# AI Overview Analytics Dashboard

A comprehensive Streamlit dashboard for analyzing AI Overview presence across brands, countries, and search characteristics in marketing data.

## Features

### 🤖 AI Overview Analysis
- **Dual Metrics**: Both distribution and rate calculations with clear explanations
- **Brand Analysis**: AI Overview presence across L'Oréal brands (Garnier, L'Oréal Paris, La Roche-Posay, Vichy, Maybelline)
- **Geographic Analysis**: Country-level AI Overview patterns
- **SERP Analysis**: Position, search volume, and SERP features correlation
- **Time Trends**: Monthly AI Overview evolution by country

### 📊 Visualization Types
- Interactive bar charts with percentage breakdowns
- Time series trends with multi-country comparison
- Position and search volume binning analysis
- SERP features and keyword intent analysis

### 🔐 Security
- Password-protected access
- Session-based authentication
- Secure data handling

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-overview-dashboard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**
   - Place your parquet file in the `data/` directory
   - Update the file path in `dashboard.py` if needed
   - Ensure your data has these required columns:
     - `AI Overview presence` (boolean)
     - `brand` (string)
     - `country` (string)
     - `Position` (numeric)
     - `Search Volume` (numeric)
     - `Month` (datetime)
     - `Position Type` (string)
     - `Keyword Intents` (string)
     - `SERP Features by Keyword` (string)

5. **Run the dashboard**
   ```bash
   streamlit run dashboard.py
   ```

6. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - Enter the password when prompted
   - Explore the AI Overview analytics

## Data Requirements

### Required Columns
| Column | Type | Description |
|--------|------|-------------|
| `AI Overview presence` | boolean | Whether AI Overview appears in SERP |
| `brand` | string | Brand name (Garnier, L'Oréal Paris, etc.) |
| `country` | string | Country name |
| `Position` | numeric | SERP ranking position (1-100) |
| `Search Volume` | numeric | Monthly search volume |
| `Month` | datetime | Time period |
| `Position Type` | string | Type of SERP result |
| `Keyword Intents` | string | Search intent classification |
| `SERP Features by Keyword` | string | SERP features present |

### Data Format
- **File Type**: Parquet format recommended for performance
- **Size**: Optimized for large datasets (15M+ records)
- **Encoding**: UTF-8

## Dashboard Sections

### 🎯 Key Metrics
- Total records processed
- AI Overview presence count and percentage
- Brand and country coverage statistics
- Average position comparison (with/without AI Overview)

### 📊 AI Overview Analysis
**Two calculation types with clear explanations:**

1. **Rate**: What percentage of each brand's records have AI Overview
2. **Distribution**: Of all AI Overview records, what percentage belong to each brand

### 🌍 Geographic Analysis
- Country-level AI Overview rates and distribution
- Time trends by country
- Regional performance comparison

### 🔍 Advanced SERP Analysis
- Position range analysis (1-3, 4-10, 11-20, etc.)
- Search volume correlation
- Position type breakdown
- Keyword intent analysis
- SERP features correlation

## Configuration

### Dashboard Settings
- **Password**: Set in `dashboard.py` (line 342)
- **Data Path**: Update file path in `load_data()` function
- **Port**: Default 8501, configurable in startup script

### Performance Options
- **Full Dataset**: Loads complete dataset (recommended)
- **Memory Usage**: ~8-12GB for 15M records
- **Load Time**: 30-60 seconds initial load

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Ensure sufficient RAM (16GB recommended)
   - Check data file size and format

2. **Password Issues**
   - Password is case-sensitive
   - Clear browser cache if login fails

3. **Data Loading Errors**
   - Verify parquet file path
   - Check required columns are present
   - Ensure proper data types

4. **Performance Issues**
   - Allow 30-60 seconds for initial load
   - Use caching for repeated analysis
   - Close other memory-intensive applications

### Support
For technical issues or questions, please check the troubleshooting section or create an issue in the repository.

## Technical Stack
- **Frontend**: Streamlit 1.47+
- **Data Processing**: Pandas 2.0+
- **Visualizations**: Plotly 6.2+
- **Data Format**: Parquet (PyArrow)
- **Authentication**: Streamlit session state

## License
[Add your license information here]

## Contributing
[Add contribution guidelines here]