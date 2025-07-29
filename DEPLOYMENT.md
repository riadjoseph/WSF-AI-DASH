# Deployment Guide

## Local Deployment

### Quick Start
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ai-overview-dashboard
   ```

2. **Add your data file**
   ```bash
   # Copy your parquet file to the data directory
   cp /path/to/your/data.parquet data/
   ```

3. **Run the dashboard**
   ```bash
   ./run_dashboard.sh
   ```

4. **Access the dashboard**
   - Open browser to `http://localhost:8501`
   - Enter password: `wsfseoteam`

### Manual Setup
If the startup script doesn't work:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard.py
```

## Production Deployment

### Streamlit Cloud
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial dashboard setup"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Deploy the `dashboard.py` file

3. **Add your data**
   - Upload your parquet file to the deployed environment
   - Or use cloud storage (S3, GCS, etc.)

### Docker Deployment
Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t ai-overview-dashboard .
docker run -p 8501:8501 ai-overview-dashboard
```

### Environment Variables
For production, consider using environment variables:

```python
import os
PASSWORD = os.getenv('DASHBOARD_PASSWORD', 'wsfseoteam')
DATA_PATH = os.getenv('DATA_PATH', 'data/combined_SOS_info_with_brand_country_fixed.parquet')
```

## Security Considerations

### Password Protection
- Default password: `wsfseoteam`
- Change password in `dashboard.py` line 342
- Consider using environment variables in production

### Data Security
- Data files are excluded from git commits
- Use secure file transfer for sensitive data
- Consider encrypting data at rest

### Network Security
- Use HTTPS in production
- Implement IP whitelisting if needed
- Consider VPN access for sensitive deployments

## Performance Optimization

### Large Datasets
- Current setup handles 15M+ records
- Requires 8-16GB RAM for full dataset
- Initial load time: 30-60 seconds

### Caching
- Streamlit caching enabled for data loading
- Cache persists across user sessions
- Clear cache if data is updated

### Monitoring
- Monitor memory usage
- Check load times
- Monitor concurrent users

## Troubleshooting

### Common Issues
1. **Data file not found**
   - Ensure parquet file is in `data/` directory
   - Check file path in `dashboard.py`

2. **Memory errors**
   - Increase available RAM
   - Consider data sampling for development

3. **Performance issues**
   - Allow time for initial load
   - Check system resources
   - Consider caching strategies

### Logs
Check Streamlit logs for detailed error information:
```bash
streamlit run dashboard.py --logger.level debug
```