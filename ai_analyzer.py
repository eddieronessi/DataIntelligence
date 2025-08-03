import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import silhouette_score
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import warnings
warnings.filterwarnings('ignore')

class AdvancedAIAnalyzer:
    """Advanced AI-powered data analysis engine"""

    def __init__(self, data):
        self.data = data.copy()
        self.numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(data.select_dtypes(include=['object']).columns)
        self.insights = []

    def comprehensive_analysis(self):
        """Perform comprehensive AI-driven analysis"""
        results = {
            'data_quality': self.analyze_data_quality(),
            'statistical_insights': self.generate_statistical_insights(),
            'anomaly_detection': self.detect_anomalies(),
            'clustering_analysis': self.perform_clustering(),
            'feature_importance': self.analyze_feature_importance(),
            'recommendations': self.generate_recommendations(),
            'natural_language_insights': self.generate_natural_language_insights(),
            'advanced_statistics': self.perform_advanced_statistics()
        }
        return results

    def generate_visualizations(self):
        """Generate interactive visualizations using Plotly"""
        visualizations = {}
        
        try:
            # 1. Distribution plots for numeric columns
            for i, col in enumerate(self.numeric_columns[:4]):  # Limit to first 4 columns
                try:
                    col_data = self.data[col].dropna()
                    if len(col_data) > 0:
                        fig = px.histogram(
                            x=col_data, 
                            title=f'Distribution of {col}',
                            marginal="box",
                            nbins=30
                        )
                        fig.update_layout(
                            showlegend=False,
                            template="plotly_white",
                            height=400,
                            xaxis_title=col,
                            yaxis_title='Frequency'
                        )
                        visualizations[f'histogram_{col.replace(" ", "_")}'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                except Exception as e:
                    print(f"Error creating histogram for {col}: {e}")
                    continue

            # 2. Pie charts for categorical columns
            for i, col in enumerate(self.categorical_columns[:3]):  # Limit to first 3 categorical columns
                try:
                    value_counts = self.data[col].value_counts().head(10)  # Top 10 categories
                    if len(value_counts) > 1:
                        fig = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f'Distribution of {col}'
                        )
                        fig.update_layout(
                            height=400,
                            template="plotly_white"
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        visualizations[f'pie_{col.replace(" ", "_")}'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                except Exception as e:
                    print(f"Error creating pie chart for {col}: {e}")
                    continue

            # 3. Correlation heatmap
            if len(self.numeric_columns) > 1:
                try:
                    corr_matrix = self.data[self.numeric_columns].corr()
                    # Handle NaN values
                    corr_matrix = corr_matrix.fillna(0)
                    
                    fig = px.imshow(
                        corr_matrix,
                        title="Correlation Heatmap",
                        color_continuous_scale="RdBu_r",
                        aspect="auto",
                        text_auto=True
                    )
                    fig.update_layout(
                        height=500,
                        template="plotly_white"
                    )
                    fig.update_traces(texttemplate="%{z:.2f}")
                    visualizations['correlation_heatmap'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                except Exception as e:
                    print(f"Error creating correlation heatmap: {e}")

            # 4. Box plots for outlier detection
            if len(self.numeric_columns) > 0:
                try:
                    cols_to_plot = self.numeric_columns[:4]  # First 4 numeric columns
                    fig = go.Figure()
                    
                    for col in cols_to_plot:
                        col_data = self.data[col].dropna()
                        if len(col_data) > 0:
                            fig.add_trace(go.Box(
                                y=col_data,
                                name=col,
                                boxmean=True
                            ))
                    
                    fig.update_layout(
                        title="Box Plots for Outlier Detection",
                        yaxis_title="Values",
                        template="plotly_white",
                        height=400
                    )
                    visualizations['box_plots'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                except Exception as e:
                    print(f"Error creating box plots: {e}")

            # 5. Scatter plot for top 2 correlated variables
            if len(self.numeric_columns) >= 2:
                try:
                    corr_matrix = self.data[self.numeric_columns].corr()
                    corr_matrix = corr_matrix.fillna(0)
                    
                    # Find the pair with highest correlation (excluding diagonal)
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if not np.isnan(corr_val) and abs(corr_val) > 0.1:  # Lower threshold
                                corr_pairs.append((
                                    corr_matrix.columns[i],
                                    corr_matrix.columns[j],
                                    abs(corr_val)
                                ))
                    
                    if corr_pairs:
                        # Sort by correlation strength and take the top pair
                        corr_pairs.sort(key=lambda x: x[2], reverse=True)
                        var1, var2, _ = corr_pairs[0]
                        
                        # Clean data for scatter plot
                        scatter_data = self.data[[var1, var2]].dropna()
                        
                        if len(scatter_data) > 0:
                            fig = px.scatter(
                                scatter_data,
                                x=var1,
                                y=var2,
                                title=f'Scatter Plot: {var1} vs {var2}',
                                trendline="ols"
                            )
                            fig.update_layout(
                                template="plotly_white",
                                height=400
                            )
                            visualizations['scatter_plot'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                except Exception as e:
                    print(f"Error creating scatter plot: {e}")

            # 6. Missing values heatmap
            try:
                missing_data = self.data.isnull()
                if missing_data.sum().sum() > 0:
                    missing_matrix = missing_data.astype(int)
                    
                    # Sample data if too large
                    if len(missing_matrix) > 1000:
                        missing_matrix = missing_matrix.sample(1000, random_state=42)
                    
                    fig = px.imshow(
                        missing_matrix.T,
                        title="Missing Values Pattern",
                        color_continuous_scale=["white", "red"],
                        aspect="auto"
                    )
                    fig.update_layout(
                        xaxis_title="Row Index",
                        yaxis_title="Columns",
                        height=400,
                        template="plotly_white"
                    )
                    visualizations['missing_values'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            except Exception as e:
                print(f"Error creating missing values heatmap: {e}")

            # 7. Bar charts for categorical data
            for i, col in enumerate(self.categorical_columns[3:6]):  # Next 3 categorical columns
                try:
                    value_counts = self.data[col].value_counts().head(8)  # Top 8 categories
                    if len(value_counts) > 0:
                        fig = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=f'Top Categories in {col}',
                            labels={'x': col, 'y': 'Count'}
                        )
                        fig.update_layout(
                            template="plotly_white",
                            height=400,
                            xaxis_tickangle=-45
                        )
                        visualizations[f'bar_{col.replace(" ", "_")}'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                except Exception as e:
                    print(f"Error creating bar chart for {col}: {e}")
                    continue

            # 8. Data overview pie chart
            try:
                data_types = {
                    'Numeric': len(self.numeric_columns),
                    'Categorical': len(self.categorical_columns),
                    'Missing Values': self.data.isnull().sum().sum()
                }
                
                # Only include non-zero values
                data_types = {k: v for k, v in data_types.items() if v > 0}
                
                if data_types:
                    fig = px.pie(
                        values=list(data_types.values()),
                        names=list(data_types.keys()),
                        title='Data Overview'
                    )
                    fig.update_layout(
                        height=400,
                        template="plotly_white"
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    visualizations['data_overview_pie'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            except Exception as e:
                print(f"Error creating data overview pie chart: {e}")

        except Exception as e:
            print(f"General error in visualization generation: {e}")
            
        # Ensure we have at least one visualization
        if not visualizations and len(self.numeric_columns) > 0:
            try:
                col = self.numeric_columns[0]
                col_data = self.data[col].dropna()
                fig = px.histogram(
                    x=col_data, 
                    title=f'Simple Distribution of {col}'
                )
                fig.update_layout(template="plotly_white", height=400)
                visualizations['fallback_histogram'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            except:
                pass

        print(f"Generated {len(visualizations)} visualizations")
        return visualizations

    def analyze_data_quality(self):
        """Analyze data quality issues"""
        quality_report = {
            'completeness': {},
            'consistency': {},
            'validity': {},
            'duplicates': {},
            'summary_score': 0
        }

        try:
            total_cells = len(self.data) * len(self.data.columns)
            missing_cells = self.data.isnull().sum().sum()
            completeness_score = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0

            quality_report['completeness'] = {
                'overall_score': round(completeness_score, 2),
                'missing_by_column': {col: int(self.data[col].isnull().sum()) for col in self.data.columns},
                'missing_patterns': self.identify_missing_patterns()
            }

            duplicate_rows = self.data.duplicated().sum()
            quality_report['duplicates'] = {
                'count': int(duplicate_rows),
                'percentage': round((duplicate_rows / len(self.data)) * 100, 2) if len(self.data) > 0 else 0
            }

            consistency_issues = []
            for col in self.numeric_columns:
                try:
                    col_data = self.data[col].dropna()
                    if len(col_data) > 0:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:
                            outliers = col_data[(col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))]
                            if len(outliers) > 0:
                                consistency_issues.append({
                                    'column': col,
                                    'issue_type': 'outliers',
                                    'count': len(outliers),
                                    'percentage': round((len(outliers) / len(col_data)) * 100, 2)
                                })
                except:
                    continue

            quality_report['consistency']['issues'] = consistency_issues

            quality_score = (completeness_score +
                           (100 - quality_report['duplicates']['percentage']) +
                           max(0, 100 - len(consistency_issues) * 10)) / 3

            quality_report['summary_score'] = round(quality_score, 1)

        except Exception as e:
            print(f"Data quality analysis error: {e}")
            quality_report['summary_score'] = 50  # Default score

        return quality_report

    def identify_missing_patterns(self):
        """Identify patterns in missing data"""
        missing_patterns = []
        try:
            missing_matrix = self.data.isnull()

            for col in self.data.columns:
                if missing_matrix[col].sum() > 0:
                    correlation_with_missing = missing_matrix.corrwith(missing_matrix[col])
                    high_corr_missing = correlation_with_missing[
                        (correlation_with_missing > 0.5) & 
                        (correlation_with_missing < 1.0)
                    ].index.tolist()

                    if high_corr_missing:
                        missing_patterns.append({
                            'primary_column': col,
                            'related_columns': high_corr_missing
                        })
        except Exception as e:
            print(f"Missing patterns analysis error: {e}")

        return missing_patterns

    def generate_statistical_insights(self):
        """Generate advanced statistical insights"""
        insights = {
            'distribution_analysis': {},
            'correlation_insights': {},
            'statistical_tests': {}
        }

        try:
            for col in self.numeric_columns:
                try:
                    data_clean = self.data[col].dropna()
                    if len(data_clean) > 3:  # Need at least 3 data points
                        sample_size = min(5000, len(data_clean))
                        sample_data = data_clean.sample(sample_size, random_state=42) if len(data_clean) > sample_size else data_clean
                        
                        shapiro_stat, shapiro_p = stats.shapiro(sample_data)
                        
                        insights['distribution_analysis'][col] = {
                            'mean': float(data_clean.mean()),
                            'median': float(data_clean.median()),
                            'mode': float(data_clean.mode().iloc[0]) if len(data_clean.mode()) > 0 else None,
                            'std': float(data_clean.std()),
                            'skewness': float(data_clean.skew()),
                            'kurtosis': float(data_clean.kurtosis()),
                            'is_normal': bool(shapiro_p > 0.05),
                            'normality_p_value': float(shapiro_p),
                            'distribution_type': self.classify_distribution(data_clean)
                        }
                except Exception as e:
                    print(f"Statistical analysis error for {col}: {e}")
                    continue

            if len(self.numeric_columns) > 1:
                try:
                    corr_matrix = self.data[self.numeric_columns].corr()
                    corr_matrix = corr_matrix.fillna(0)
                    corr_pairs = []

                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if not np.isnan(corr_val):
                                corr_pairs.append({
                                    'var1': corr_matrix.columns[i],
                                    'var2': corr_matrix.columns[j],
                                    'correlation': float(corr_val),
                                    'strength': self.classify_correlation_strength(abs(corr_val))
                                })

                    corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

                    insights['correlation_insights'] = {
                        'strongest_correlations': corr_pairs[:10],
                        'correlation_summary': {
                            'strong_positive': int(len([c for c in corr_pairs if c['correlation'] > 0.7])),
                            'strong_negative': int(len([c for c in corr_pairs if c['correlation'] < -0.7])),
                            'moderate': int(len([c for c in corr_pairs if 0.3 <= abs(c['correlation']) <= 0.7])),
                            'weak': int(len([c for c in corr_pairs if abs(c['correlation']) < 0.3]))
                        }
                    }
                except Exception as e:
                    print(f"Correlation analysis error: {e}")

        except Exception as e:
            print(f"Statistical insights error: {e}")

        return insights

    def classify_distribution(self, data):
        """Classify the distribution type"""
        try:
            skewness = data.skew()
            kurtosis = data.kurtosis()

            if abs(skewness) < 0.5 and abs(kurtosis) < 3:
                return "Normal-like"
            elif skewness > 1:
                return "Right-skewed"
            elif skewness < -1:
                return "Left-skewed"
            elif kurtosis > 3:
                return "Heavy-tailed"
            elif kurtosis < -1:
                return "Light-tailed"
            else:
                return "Moderate asymmetry"
        except:
            return "Unknown"

    def classify_correlation_strength(self, corr_val):
        """Classify correlation strength"""
        if corr_val >= 0.9:
            return "Very Strong"
        elif corr_val >= 0.7:
            return "Strong"
        elif corr_val >= 0.5:
            return "Moderate"
        elif corr_val >= 0.3:
            return "Weak"
        else:
            return "Very Weak"

    def detect_anomalies(self):
        """Detect anomalies in the data"""
        anomaly_results = {}

        try:
            if len(self.numeric_columns) > 0:
                numeric_data = self.data[self.numeric_columns].dropna()

                if len(numeric_data) > 10:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_labels = iso_forest.fit_predict(numeric_data)
                    anomaly_indices = numeric_data.index[anomaly_labels == -1].tolist()

                    anomaly_results = {
                        'method': 'Isolation Forest',
                        'anomaly_count': int(np.sum(anomaly_labels == -1)),
                        'anomaly_percentage': round((np.sum(anomaly_labels == -1) / len(numeric_data)) * 100, 2),
                        'anomaly_indices': anomaly_indices[:50],  # Limit to first 50
                        'anomaly_summary': self.summarize_anomalies(numeric_data.loc[anomaly_indices])
                    }
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            anomaly_results = {
                'anomaly_count': 0,
                'anomaly_percentage': 0.0,
                'error': str(e)
            }

        return anomaly_results

    def summarize_anomalies(self, anomaly_data):
        """Summarize anomaly characteristics"""
        if len(anomaly_data) == 0:
            return {}

        summary = {}
        try:
            for col in anomaly_data.columns:
                col_data = anomaly_data[col]
                normal_data = self.data[col].dropna()

                summary[col] = {
                    'anomaly_mean': float(col_data.mean()) if len(col_data) > 0 else 0,
                    'normal_mean': float(normal_data.mean()) if len(normal_data) > 0 else 0,
                    'difference_from_normal': float(col_data.mean() - normal_data.mean()) if len(col_data) > 0 and len(normal_data) > 0 else 0,
                    'anomaly_std': float(col_data.std()) if len(col_data) > 1 else 0
                }
        except Exception as e:
            print(f"Anomaly summary error: {e}")

        return summary

    def perform_clustering(self):
        """Perform clustering analysis"""
        clustering_results = {}

        try:
            if len(self.numeric_columns) >= 2:
                cluster_data = self.data[self.numeric_columns].dropna()

                if len(cluster_data) > 10:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(cluster_data)

                    inertias = []
                    k_range = range(2, min(11, len(cluster_data)))

                    for k in k_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(scaled_data)
                        inertias.append(kmeans.inertia_)

                    # Find optimal k using elbow method
                    optimal_k = k_range[0]
                    if len(inertias) > 2:
                        for i in range(1, len(inertias) - 1):
                            if inertias[i - 1] - inertias[i] > inertias[i] - inertias[i + 1]:
                                optimal_k = k_range[i]
                                break

                    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(scaled_data)

                    cluster_analysis = {}
                    for cluster_id in range(optimal_k):
                        cluster_mask = cluster_labels == cluster_id
                        cluster_subset = cluster_data[cluster_mask]

                        cluster_analysis[f'Cluster_{cluster_id}'] = {
                            'size': int(np.sum(cluster_mask)),
                            'percentage': round((np.sum(cluster_mask) / len(cluster_data)) * 100, 2),
                            'characteristics': {
                                col: {
                                    'mean': float(cluster_subset[col].mean()),
                                    'std': float(cluster_subset[col].std()) if len(cluster_subset) > 1 else 0
                                } for col in self.numeric_columns
                            }
                        }

                    clustering_results = {
                        'optimal_clusters': int(optimal_k),
                        'cluster_analysis': cluster_analysis,
                        'silhouette_score': float(self.calculate_silhouette_score(scaled_data, cluster_labels))
                    }
        except Exception as e:
            print(f"Clustering analysis error: {e}")
            clustering_results = {'error': str(e)}

        return clustering_results

    def calculate_silhouette_score(self, data, labels):
        """Calculate silhouette score"""
        try:
            if len(set(labels)) > 1 and len(data) > 1:
                return silhouette_score(data, labels)
            else:
                return 0.0
        except:
            return 0.0

    def analyze_feature_importance(self):
        """Analyze feature importance using Random Forest"""
        try:
            df = self.data.copy()
            numeric_df = df.select_dtypes(include=[np.number]).dropna()

            if numeric_df.shape[1] < 2:
                return {'error': 'Not enough numeric features to analyze importance.'}

            X = numeric_df.iloc[:, :-1]
            y = numeric_df.iloc[:, -1]

            # Convert regression to classification if needed
            if y.nunique() > 10:
                y = pd.cut(y, bins=5, labels=False)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            importances = model.feature_importances_
            feature_names = X.columns
            importance_dict = dict(zip(feature_names, importances))

            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return sorted_importance
        except Exception as e:
            print(f"Feature importance error: {e}")
            return {'error': f'Feature importance analysis failed: {str(e)}'}

    def generate_recommendations(self):
        """Generate data science recommendations"""
        recommendations = {
            "data_preprocessing": [
                "Handle missing values using appropriate imputation strategies",
                "Remove or cap outliers based on domain knowledge",
                "Normalize or standardize numerical features for modeling",
                "Encode categorical variables using one-hot or label encoding"
            ],
            "feature_engineering": [
                "Create interaction features between important variables",
                "Apply feature selection to reduce dimensionality",
                "Consider polynomial features for non-linear relationships",
                "Use domain knowledge to create meaningful derived features"
            ],
            "modeling": [
                "Split data into training, validation, and test sets",
                "Try multiple algorithms and compare performance",
                "Use cross-validation for robust model evaluation",
                "Perform hyperparameter tuning for optimal results"
            ],
            "validation": [
                "Check for data leakage between training and test sets",
                "Validate model assumptions and residuals",
                "Test model performance on unseen data",
                "Monitor model performance over time"
            ]
        }
        return recommendations

    def generate_natural_language_insights(self):
        """Generate natural language insights about the data"""
        insights = []
        
        try:
            rows, cols = self.data.shape
            insights.append(f"Dataset contains {rows:,} rows and {cols} columns with {len(self.numeric_columns)} numeric and {len(self.categorical_columns)} categorical features.")
            
            # Missing values insight
            missing_pct = (self.data.isnull().sum() / len(self.data)) * 100
            total_missing = missing_pct.sum()
            if total_missing > 0:
                high_missing = missing_pct[missing_pct > 20]
                if not high_missing.empty:
                    insights.append(f"Significant missing values detected in {len(high_missing)} columns: {', '.join(high_missing.index[:3])}")
                else:
                    insights.append(f"Dataset has {total_missing:.1f}% total missing values across all columns.")
            
            # Correlation insight
            if len(self.numeric_columns) > 1:
                try:
                    corr_matrix = self.data[self.numeric_columns].corr()
                    corr_matrix = corr_matrix.fillna(0)
                    high_corr_pairs = []
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.8:
                                high_corr_pairs.append(f"{corr_matrix.columns[i]}-{corr_matrix.columns[j]}")
                    
                    if high_corr_pairs:
                        insights.append(f"Strong correlations (>0.8) found between: {', '.join(high_corr_pairs[:3])}")
                    else:
                        insights.append("No extremely strong correlations (>0.8) detected between numeric variables.")
                except:
                    pass
            
            # Data distribution insights
            for col in self.numeric_columns[:3]:  # Check first 3 numeric columns
                try:
                    col_data = self.data[col].dropna()
                    if len(col_data) > 0:
                        skewness = col_data.skew()
                        if abs(skewness) > 2:
                            direction = "right" if skewness > 0 else "left"
                            insights.append(f"Column '{col}' shows significant {direction} skewness ({skewness:.2f}), consider transformation.")
                except:
                    continue
            
            # Categorical insights
            for col in self.categorical_columns[:2]:  # Check first 2 categorical columns
                try:
                    unique_values = self.data[col].nunique()
                    total_values = len(self.data[col].dropna())
                    if unique_values == total_values:
                        insights.append(f"Column '{col}' appears to be a unique identifier with no repeated values.")
                    elif unique_values < 5:
                        insights.append(f"Column '{col}' has only {unique_values} unique categories, suitable for categorical encoding.")
                except:
                    continue
            
            # Data quality insight
            try:
                duplicates = self.data.duplicated().sum()
                if duplicates > 0:
                    insights.append(f"Found {duplicates} duplicate rows ({duplicates/len(self.data)*100:.1f}% of data).")
            except:
                pass
            
        except Exception as e:
            print(f"Natural language insights error: {e}")
            insights.append("Basic dataset loaded successfully. Ready for analysis.")

        return insights

    def perform_advanced_statistics(self):
        """Perform advanced statistical analysis"""
        stats_summary = {}

        try:
            for col in self.numeric_columns:
                try:
                    col_data = self.data[col].dropna()
                    if len(col_data) > 1:
                        stats_summary[col] = {
                            'min': float(col_data.min()),
                            'max': float(col_data.max()),
                            'variance': float(col_data.var()),
                            'range': float(col_data.max() - col_data.min()),
                            'iqr': float(stats.iqr(col_data)) if len(col_data) > 0 else 0,
                            'zscore_outliers': int(np.sum(np.abs(stats.zscore(col_data)) > 3)) if len(col_data) > 1 else 0
                        }
                except Exception as e:
                    print(f"Advanced stats error for {col}: {e}")
                    continue
        except Exception as e:
            print(f"Advanced statistics error: {e}")

        return stats_summary


class DataScienceWorkflowGenerator:
    """Generate comprehensive data science workflows"""
    
    def __init__(self, data):
        self.data = data.copy()
        self.numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(data.select_dtypes(include=['object']).columns)

    def generate_complete_workflow(self):
        """Generate a complete data science workflow"""
        workflow = {
            'data_exploration': self.generate_exploration_steps(),
            'data_preprocessing': self.generate_preprocessing_steps(),
            'feature_engineering': self.generate_feature_engineering_steps(),
            'modeling': self.generate_modeling_steps(),
            'evaluation': self.generate_evaluation_steps()
        }
        return workflow

    def generate_exploration_steps(self):
        """Generate data exploration steps"""
        steps = [
            {
                'step': 1,
                'action': 'Load and inspect data structure',
                'code_snippet': 'df.info(); df.describe(); df.head()',
                'purpose': 'Understand data types, missing values, and basic statistics'
            },
            {
                'step': 2,
                'action': 'Analyze missing values pattern',
                'code_snippet': 'df.isnull().sum(); sns.heatmap(df.isnull())',
                'purpose': 'Identify missing data patterns for proper handling'
            },
            {
                'step': 3,
                'action': 'Visualize data distributions',
                'code_snippet': 'df.hist(bins=20, figsize=(15, 10))',
                'purpose': 'Understand feature distributions and detect outliers'
            }
        ]
        return steps

    def generate_preprocessing_steps(self):
        """Generate preprocessing steps based on data characteristics"""
        steps = []
        
        # Missing values handling
        missing_pct = (self.data.isnull().sum() / len(self.data)) * 100
        if missing_pct.max() > 0:
            steps.append({
                'action': 'Handle missing values',
                'code_snippet': 'df.fillna(df.mean()) # for numeric\ndf.fillna(df.mode().iloc[0]) # for categorical',
                'priority': 'high',
                'impact': 'Essential for model training'
            })

        # Outlier detection
        if len(self.numeric_columns) > 0:
            steps.append({
                'action': 'Detect and handle outliers',
                'code_snippet': 'Q1 = df.quantile(0.25); Q3 = df.quantile(0.75); IQR = Q3 - Q1',
                'priority': 'medium',
                'impact': 'Improves model robustness'
            })

        # Categorical encoding
        if len(self.categorical_columns) > 0:
            steps.append({
                'action': 'Encode categorical variables',
                'code_snippet': 'pd.get_dummies(df) # or LabelEncoder()',
                'priority': 'high',
                'impact': 'Required for most ML algorithms'
            })

        # Feature scaling
        if len(self.numeric_columns) > 0:
            steps.append({
                'action': 'Scale numerical features',
                'code_snippet': 'StandardScaler().fit_transform(X)',
                'priority': 'medium',
                'impact': 'Important for distance-based algorithms'
            })

        return steps

    def generate_feature_engineering_steps(self):
        """Generate feature engineering recommendations"""
        steps = [
            {
                'action': 'Create interaction features',
                'code_snippet': 'df["feature1_x_feature2"] = df["feature1"] * df["feature2"]',
                'priority': 'low',
                'impact': 'Can capture non-linear relationships'
            },
            {
                'action': 'Feature selection',
                'code_snippet': 'SelectKBest(f_regression, k=10).fit_transform(X, y)',
                'priority': 'medium',
                'impact': 'Reduces overfitting and improves performance'
            },
            {
                'action': 'Dimensionality reduction',
                'code_snippet': 'PCA(n_components=0.95).fit_transform(X)',
                'priority': 'low',
                'impact': 'Useful for high-dimensional data'
            }
        ]
        return steps

    def generate_modeling_steps(self):
        """Generate modeling workflow steps"""
        steps = [
            {
                'action': 'Split data into train/validation/test sets',
                'code_snippet': 'train_test_split(X, y, test_size=0.2, random_state=42)',
                'priority': 'high',
                'impact': 'Essential for proper model evaluation'
            },
            {
                'action': 'Try multiple algorithms',
                'code_snippet': 'RandomForestClassifier(), LogisticRegression(), XGBClassifier()',
                'priority': 'high',
                'impact': 'Compare different approaches'
            },
            {
                'action': 'Hyperparameter tuning',
                'code_snippet': 'GridSearchCV(estimator, param_grid, cv=5)',
                'priority': 'medium',
                'impact': 'Optimize model performance'
            }
        ]
        return steps

    def generate_evaluation_steps(self):
        """Generate model evaluation steps"""
        steps = [
            {
                'action': 'Calculate performance metrics',
                'code_snippet': 'accuracy_score(), classification_report(), confusion_matrix()',
                'priority': 'high',
                'impact': 'Understand model performance'
            },
            {
                'action': 'Cross-validation',
                'code_snippet': 'cross_val_score(model, X, y, cv=5)',
                'priority': 'high',
                'impact': 'Ensure model generalization'
            },
            {
                'action': 'Feature importance analysis',
                'code_snippet': 'model.feature_importances_',
                'priority': 'medium',
                'impact': 'Understand which features drive predictions'
            }
        ]
        return steps
