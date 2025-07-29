#!/bin/bash

echo "🤖 Starting AI Overview Analytics Dashboard..."
echo "📊 Dashboard will open in your default browser"
echo "🔍 Focus: AI Overview presence analysis across brands and countries"
echo "⚠️  Note: First load may take a few moments due to large dataset"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python -m venv .venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source .venv/bin/activate

# Check if required packages are installed
python -c "import streamlit, pandas, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Required packages not installed. Installing now..."
    pip install -r requirements.txt
    echo "✅ Packages installed"
fi

# Check if data file exists
if [ ! -f "data/"*.parquet ]; then
    echo "⚠️  No parquet file found in data/ directory"
    echo "📝 Please follow these steps:"
    echo "   1. Copy your parquet file to the data/ directory"
    echo "   2. Update the file path in dashboard.py (line 23)"
    echo "   3. Run this script again"
    exit 1
fi

# Run the AI Overview dashboard
echo "🔐 Dashboard is password protected"
echo "🌐 Opening AI Overview dashboard at http://localhost:8501"
echo ""
streamlit run dashboard.py --server.port=8501 --server.address=localhost