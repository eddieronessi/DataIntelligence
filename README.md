# Data Intelligence - AI-Powered Data Science Platform

A comprehensive Flask-based web application that transforms CSV/Excel files and SQL data into interactive dashboards with AI-powered analysis, machine learning models, and automated insights.

## 🚀 Features

### Core Capabilities
- **Multi-format Data Input**: Support for CSV, Excel files, and SQL database connections
- **Interactive Dashboards**: Dynamic Plotly visualizations with real-time interactivity
- **AI-Powered Analysis**: Automated exploratory data analysis with intelligent insights
- **Machine Learning**: Automated model building with performance evaluation
- **Advanced Analytics**: Clustering, anomaly detection, and statistical testing
- **Export Functionality**: Generate comprehensive analysis reports

### AI-Driven Features
- **Automated Data Quality Assessment**: Completeness, consistency, and validity scoring
- **Smart Insights Generation**: Natural language insights about data patterns
- **Anomaly Detection**: Multivariate outlier detection using Isolation Forest
- **Clustering Analysis**: Automated optimal cluster detection with interpretation
- **Feature Importance Analysis**: Automated feature ranking for predictive modeling
- **Workflow Recommendations**: AI-generated data science workflow suggestions

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, Flask 2.3.3
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Random Forest, Linear/Logistic Regression, Clustering
- **Database**: SQLite, MySQL, PostgreSQL support
- **Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS
- **AI/Statistics**: SciPy, Advanced statistical testing

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/data-intelligence.git
cd data-intelligence
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create Directory Structure
```bash
mkdir uploads
mkdir templates
mkdir static
```

### 5. Set Up Flask Application
Create the main application file (`app.py`) and template file (`templates/index.html`) using the provided code.

## 🚀 Running the Application

### Development Mode
```bash
python app.py
```

### Production Mode
```bash
# Using Gunicorn (install first: pip install gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Access the application at: `http://localhost:5000`

## 📁 Project Structure
```
data-intelligence/
│
├── app.py                 # Main Flask application
├── ai_analyzer.py         # Advanced AI analysis engine
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── templates/
│   └── index.html        # Main web interface
│
├── uploads/              # Temporary file storage
├── static/               # Static assets (CSS, JS)
└── venv/                 # Virtual environment
```

## 💡 Usage Guide

### 1. Data Upload
- **File Upload**: Drag and drop CSV or Excel files
- **SQL Connection**: Connect to databases using connection strings
- Supported formats: `.csv`, `.xlsx`, `.xls`

### 2. Data Analysis
- **Automatic Overview**: View dataset statistics and structure
- **Exploratory Analysis**: Generate correlation matrices, distribution analysis
- **AI Insights**: Get automated insights about data quality and patterns

### 3. Visualizations
- **Interactive Charts**: Correlation heatmaps, histograms, scatter plots
- **Categorical Analysis**: Bar charts for categorical variables
- **Responsive Design**: All charts adapt to screen size

### 4. Machine Learning
- **Automated Model Building**: Select target column and build models
- **Performance Metrics**: Accuracy, R², feature importance
- **Model Comparison**: Compare multiple algorithms automatically

### 5. Advanced Analytics
- **Clustering Analysis**: Automatic cluster detection and interpretation
- **Anomaly Detection**: Identify outliers and unusual patterns
- **Statistical Testing**: Normality tests, independence tests

## 🔧 Configuration

### Database Connections
```python
# SQLite (default)
connection_string = "your_database.db"

# MySQL
connection_string = "mysql+pymysql://user:password@host:port/database"

# PostgreSQL
connection_string = "postgresql://user:password@host:port/database"
```

### File Upload Limits
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard interface |
| `/upload` | POST | Upload and process files |
| `/sql-connect` | POST | Connect to SQL database |
| `/analyze` | POST | Perform exploratory data analysis |
| `/visualize` | POST | Generate interactive visualizations |
| `/predict` | POST | Build machine learning models |
| `/export-report` | POST | Export analysis report |

## 📊 Sample Data Analysis Workflow

1. **Upload Data**: Load your CSV/Excel file or connect to database
2. **Data Overview**: Review automatic data quality assessment
3. **Exploratory Analysis**: Generate statistical insights and correlations
4. **Visualizations**: Create interactive charts and plots
5. **ML Modeling**: Build predictive models with target selection
6. **Export Results**: Download comprehensive analysis report

## 🔍 Advanced Features

### AI-Powered Insights
- Automatic data quality scoring
- Natural language pattern descriptions
- Anomaly detection with explanations
- Feature importance ranking
- Clustering with business interpretations

### Statistical Analysis
- Normality testing (Shapiro-Wilk)
- Independence testing (Chi-square)
- Correlation strength classification
- Distribution type identification

### Machine Learning
- Automatic problem type detection (classification/regression)
- Model comparison and selection
- Feature importance analysis
- Cross-validation support

## 🚨 Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
pip install --upgrade -r requirements.txt
```

**2. File Upload Issues**
- Check file size (max 16MB)
- Ensure proper file format (.csv, .xlsx, .xls)
- Verify file encoding (UTF-8 recommended)

**3. Database Connection Issues**
- Verify connection string format
- Check database credentials
- Ensure database server is running

**4. Visualization Not Loading**
- Check browser JavaScript console
- Ensure Plotly.js is loaded properly
- Verify data format in API responses

## 🔐 Security Considerations

- File uploads are temporarily stored and automatically cleaned
- SQL injection protection through parameterized queries
- Session management for user data isolation
- Input validation and sanitization

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Cloud Deployment Options
- **Heroku**: Easy deployment with buildpacks
- **AWS EC2**: Full control with custom configuration
- **Google Cloud Run**: Serverless container deployment
- **Azure App Service**: Managed platform deployment

## 📈 Performance Optimization

- Use chunked processing for large datasets
- Implement caching for repeated analyses
- Optimize database queries with indexing
- Use CDN for static assets in production

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Flask community for the excellent web framework
- Plotly for interactive visualization capabilities
- Scikit-learn for machine learning algorithms
- Pandas team for data manipulation tools

## 📞 Support

- Create an issue on GitHub for bug reports
- Check existing issues for common problems
- Contribute to documentation improvements

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Added AI-powered insights and advanced analytics
- **v1.2.0**: Enhanced machine learning capabilities
- **v1.3.0**: Improved visualization and dashboard features

---

**Made with ❤️ for the Data Science Community**