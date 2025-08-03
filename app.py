from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
import numpy as np
import sqlite3
import os
import json
import io
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import traceback

# Import our advanced AI analyzer
from ai_analyzer import AdvancedAIAnalyzer, DataScienceWorkflowGenerator

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class DataIntelligenceEngine:
    def __init__(self):
        self.data = None
        self.analysis_results = {}

    def load_csv(self, file_path):
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error with {encoding}: {e}")
                    continue

            if self.data is None:
                raise ValueError("Could not read CSV file with any encoding")

            # Clean data
            self.data = self._clean_data(self.data)
            return self.get_data_summary()
        except Exception as e:
            print(f"CSV loading error: {e}")
            return {"error": f"Error loading CSV: {str(e)}"}

    def load_excel(self, file_path):
        try:
            excel_file = pd.ExcelFile(file_path)
            if len(excel_file.sheet_names) == 1:
                self.data = pd.read_excel(file_path)
            else:
                self.data = pd.read_excel(file_path, sheet_name=0)

            # Clean data
            self.data = self._clean_data(self.data)
            return self.get_data_summary()
        except Exception as e:
            print(f"Excel loading error: {e}")
            return {"error": f"Error loading Excel: {str(e)}"}

    def connect_sql(self, connection_string, query):
        try:
            conn = sqlite3.connect(connection_string)
            self.data = pd.read_sql_query(query, conn)
            conn.close()
            
            # Clean data
            self.data = self._clean_data(self.data)
            return self.get_data_summary()
        except Exception as e:
            print(f"SQL connection error: {e}")
            return {"error": f"Error connecting to database: {str(e)}"}

    def _clean_data(self, df):
        """Clean and preprocess data"""
        # Remove completely empty rows and columns
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        # Convert numeric strings to numbers where possible
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='ignore')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df

    def get_data_summary(self):
        if self.data is None:
            return {"error": "No data loaded"}

        try:
            summary = {
                "shape": list(self.data.shape),
                "columns": list(self.data.columns),
                "dtypes": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
                "missing_values": {col: int(self.data[col].isnull().sum()) for col in self.data.columns},
                "numeric_columns": list(self.data.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(self.data.select_dtypes(include=['object']).columns),
                "sample_data": self.data.head(10).fillna('N/A').to_dict('records')
            }

            if summary["numeric_columns"]:
                summary["statistics"] = self.data[summary["numeric_columns"]].describe().fillna(0).to_dict()

            return summary
        except Exception as e:
            print(f"Data summary error: {e}")
            return {"error": f"Error generating data summary: {str(e)}"}

    def perform_eda(self):
        if self.data is None:
            return {"error": "No data loaded"}

        try:
            ai_analyzer = AdvancedAIAnalyzer(self.data)
            comprehensive_results = ai_analyzer.comprehensive_analysis()

            results = {
                "correlation_analysis": {},
                "distribution_analysis": {},
                "outlier_analysis": {},
                "insights": [],
                "ai_analysis": comprehensive_results,
                "natural_language_insights": ai_analyzer.generate_natural_language_insights(),
                "advanced_statistics": ai_analyzer.perform_advanced_statistics()
            }

            numeric_cols = self.data.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 1:
                corr_matrix = self.data[numeric_cols].corr()
                results["correlation_analysis"] = {
                    "matrix": corr_matrix.fillna(0).to_dict(),
                    "high_correlations": self.find_high_correlations(corr_matrix)
                }

            for col in numeric_cols:
                col_data = self.data[col].dropna()
                if len(col_data) > 0:
                    results["distribution_analysis"][col] = {
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std()),
                        "skewness": float(col_data.skew()),
                        "kurtosis": float(col_data.kurtosis())
                    }

            traditional_insights = self.generate_insights()
            results["insights"] = traditional_insights

            return results
        except Exception as e:
            print(f"EDA error: {e}")
            traceback.print_exc()
            return {"error": f"Error performing EDA: {str(e)}"}

    def find_high_correlations(self, corr_matrix, threshold=0.7):
        high_corr = []
        try:
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val) and abs(corr_val) > threshold:
                        high_corr.append({
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "correlation": float(corr_val)
                        })
        except Exception as e:
            print(f"Correlation analysis error: {e}")
        
        return high_corr

    def generate_insights(self):
        insights = []

        if self.data is None:
            return insights

        try:
            # Missing values insight
            missing_pct = (self.data.isnull().sum() / len(self.data)) * 100
            high_missing = missing_pct[missing_pct > 20]
            if not high_missing.empty:
                insights.append(f"High missing values detected in: {', '.join(high_missing.index)}")

            # Skewness insight
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    skewness = self.data[col].skew()
                    if abs(skewness) > 2:
                        insights.append(f"{col} shows high skewness ({skewness:.2f}) - consider transformation")
                except:
                    continue

            # Outliers insight
            for col in numeric_cols:
                try:
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        outliers = self.data[(self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR))]
                        if len(outliers) > 0:
                            insights.append(f"{col} has {len(outliers)} potential outliers ({len(outliers)/len(self.data)*100:.1f}%)")
                except:
                    continue

        except Exception as e:
            print(f"Insights generation error: {e}")

        return insights

    def build_ml_models(self, target_column, model_type='auto'):
        """Build and train multiple ML models"""
        if self.data is None:
            return {"error": "No data loaded"}

        try:
            df = self.data.copy()
            
            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found"}

            # Determine problem type
            if model_type == 'auto':
                if df[target_column].dtype == 'object' or df[target_column].nunique() < 10:
                    problem_type = 'classification'
                else:
                    problem_type = 'regression'
            else:
                problem_type = model_type

            print(f"Problem type determined as: {problem_type}")

            # Prepare features
            feature_cols = []
            for col in df.columns:
                if col != target_column:
                    if df[col].dtype in ['int64', 'float64']:
                        feature_cols.append(col)
                    elif df[col].dtype == 'object' and df[col].nunique() < 20:
                        # One-hot encode categorical variables with few categories
                        encoded = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df, encoded], axis=1)
                        feature_cols.extend(encoded.columns.tolist())

            if len(feature_cols) == 0:
                return {"error": "No suitable features found for modeling"}

            # Prepare X and y
            X = df[feature_cols].fillna(df[feature_cols].mean() if problem_type == 'regression' else df[feature_cols].mode().iloc[0])
            y = df[target_column]

            # Handle missing values in target
            if y.isnull().sum() > 0:
                mask = ~y.isnull()
                X = X[mask]
                y = y[mask]

            if len(X) < 10:
                return {"error": "Not enough data points for modeling"}

            # Encode target if classification with string labels
            label_encoder = None
            if problem_type == 'classification' and y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if problem_type == 'classification' else None
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models_results = {}

            if problem_type == 'classification':
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                    'Decision Tree': DecisionTreeClassifier(random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                    'Naive Bayes': GaussianNB()
                }

                for name, model in models.items():
                    try:
                        if name in ['Logistic Regression', 'Naive Bayes']:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Feature importance
                        feature_importance = {}
                        if hasattr(model, 'feature_importances_'):
                            feature_importance = dict(zip(feature_cols, model.feature_importances_))
                        elif hasattr(model, 'coef_'):
                            feature_importance = dict(zip(feature_cols, np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)))

                        models_results[name] = {
                            'accuracy': float(accuracy),
                            'feature_importance': feature_importance
                        }

                    except Exception as e:
                        models_results[name] = {'error': str(e)}

            else:  # regression
                models = {
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Linear Regression': LinearRegression(),
                    'Decision Tree': DecisionTreeRegressor(random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
                }

                for name, model in models.items():
                    try:
                        if name == 'Linear Regression':
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)

                        # Feature importance
                        feature_importance = {}
                        if hasattr(model, 'feature_importances_'):
                            feature_importance = dict(zip(feature_cols, model.feature_importances_))
                        elif hasattr(model, 'coef_'):
                            feature_importance = dict(zip(feature_cols, np.abs(model.coef_)))

                        models_results[name] = {
                            'r2_score': float(r2),
                            'mse': float(mse),
                            'feature_importance': feature_importance
                        }

                    except Exception as e:
                        models_results[name] = {'error': str(e)}

            return {
                'problem_type': problem_type,
                'models': models_results,
                'feature_count': len(feature_cols),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }

        except Exception as e:
            print(f"ML modeling error: {e}")
            traceback.print_exc()
            return {"error": f"Error building ML models: {str(e)}"}

engine = DataIntelligenceEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        session['uploaded_filename'] = filename
        session['data_path'] = filepath

        # Load based on file type
        if filename.lower().endswith('.csv'):
            result = engine.load_csv(filepath)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            result = engine.load_excel(filepath)
        else:
            result = {'error': 'Unsupported file type. Please upload CSV or Excel files.'}

        return jsonify(result)

    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if engine.data is None:
            return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400

        print("Running EDA on DataFrame with shape:", engine.data.shape)
        results = engine.perform_eda()
        return jsonify(results)

    except Exception as e:
        print("Exception occurred during /analyze:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        if engine.data is None:
            return jsonify({"error": "No data uploaded yet. Please upload a CSV or Excel file first."}), 400

        print("Creating visualizations for DataFrame with shape:", engine.data.shape)
        analyzer = AdvancedAIAnalyzer(engine.data)
        visualizations = analyzer.generate_visualizations()
        
        print(f"Generated {len(visualizations)} visualizations")
        return jsonify(visualizations)
        
    except Exception as e:
        print(f"Visualization error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/sql-connect', methods=['POST'])
def sql_connect():
    try:
        data = request.get_json()
        connection_string = data.get('connection_string')
        query = data.get('query')
        
        if not connection_string or not query:
            return jsonify({'error': 'Connection string and query are required'}), 400
        
        result = engine.connect_sql(connection_string, query)
        
        if 'error' not in result:
            session['sql_connection'] = connection_string
            session['sql_query'] = query
        
        return jsonify(result)
    except Exception as e:
        print(f"SQL connect error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/advanced-analysis', methods=['POST'])
def advanced_analysis():
    try:
        if engine.data is None:
            return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400

        analyzer = AdvancedAIAnalyzer(engine.data)
        results = analyzer.comprehensive_analysis()
        insights = analyzer.generate_natural_language_insights()
        
        return jsonify({
            'analysis_results': results,
            'natural_language_insights': insights
        })
    except Exception as e:
        print(f"Advanced analysis error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate-workflow', methods=['POST'])
def generate_workflow():
    try:
        if engine.data is None:
            return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400

        workflow_generator = DataScienceWorkflowGenerator(engine.data)
        workflow = workflow_generator.generate_complete_workflow()
        
        return jsonify({'workflow': workflow})
    except Exception as e:
        print(f"Workflow generation error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if engine.data is None:
            return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400

        data = request.get_json()
        target_column = data.get('target_column')
        model_type = data.get('model_type', 'auto')
        
        if not target_column or target_column not in engine.data.columns:
            return jsonify({'error': 'Invalid target column. Please select a valid column.'}), 400

        # Build ML models
        results = engine.build_ml_models(target_column, model_type)
        return jsonify(results)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/export-report', methods=['POST'])
def export_report():
    try:
        if engine.data is None:
            return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400
        
        # Create PDF report
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Data Intelligence Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Timestamp
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Data Summary Section
        story.append(Paragraph("Data Summary", styles['Heading2']))
        data_summary = engine.get_data_summary()
        
        summary_data = [
            ['Metric', 'Value'],
            ['Number of Rows', str(data_summary['shape'][0])],
            ['Number of Columns', str(data_summary['shape'][1])],
            ['Numeric Columns', str(len(data_summary['numeric_columns']))],
            ['Categorical Columns', str(len(data_summary['categorical_columns']))],
            ['Total Missing Values', str(sum(data_summary['missing_values'].values()))]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Column Information
        story.append(Paragraph("Column Information", styles['Heading2']))
        
        col_data = [['Column Name', 'Data Type', 'Missing Values']]
        for col in data_summary['columns'][:15]:  # Limit to first 15 columns
            col_data.append([
                col,
                data_summary['dtypes'][col],
                str(data_summary['missing_values'][col])
            ])
        
        col_table = Table(col_data)
        col_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(col_table)
        story.append(Spacer(1, 20))
        
        # Statistical Analysis
        if data_summary.get('statistics'):
            story.append(Paragraph("Statistical Summary", styles['Heading2']))
            
            stats_data = [['Statistic'] + list(data_summary['statistics'].keys())[:5]]  # Limit columns
            
            for stat in ['mean', 'std', 'min', 'max']:
                if stat in list(data_summary['statistics'].values())[0]:
                    row = [stat.capitalize()]
                    for col in list(data_summary['statistics'].keys())[:5]:
                        value = data_summary['statistics'][col].get(stat, 0)
                        row.append(f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                    stats_data.append(row)
            
            stats_table = Table(stats_data)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(stats_table)
            story.append(Spacer(1, 20))
        
        # Insights Section
        try:
            insights = engine.generate_insights()
            if insights:
                story.append(Paragraph("Key Insights", styles['Heading2']))
                for i, insight in enumerate(insights[:10], 1):  # Limit to 10 insights
                    story.append(Paragraph(f"{i}. {insight}", styles['Normal']))
                    story.append(Spacer(1, 6))
                story.append(Spacer(1, 20))
        except:
            pass
        
        # Data Quality Assessment
        try:
            analyzer = AdvancedAIAnalyzer(engine.data)
            quality_results = analyzer.analyze_data_quality()
            
            story.append(Paragraph("Data Quality Assessment", styles['Heading2']))
            story.append(Paragraph(f"Overall Quality Score: {quality_results.get('summary_score', 'N/A')}/100", styles['Normal']))
            story.append(Paragraph(f"Completeness: {quality_results['completeness']['overall_score']:.1f}%", styles['Normal']))
            story.append(Paragraph(f"Duplicate Records: {quality_results['duplicates']['percentage']}%", styles['Normal']))
            story.append(Spacer(1, 20))
        except:
            pass
        
        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading2']))
        recommendations = [
            "1. Handle missing values before analysis",
            "2. Check for and remove duplicate records if necessary", 
            "3. Consider feature scaling for machine learning models",
            "4. Perform outlier detection and treatment",
            "5. Encode categorical variables for modeling",
            "6. Split data into training and testing sets",
            "7. Validate model performance using cross-validation"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
            story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f'data_intelligence_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"Export error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
