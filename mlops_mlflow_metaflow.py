import logging
import sys
import warnings
from urllib.parse  import urlparse
import dagshub
import mlflow
import mlflow.sklearn
from metaflow import FlowSpec, step, Parameter, current
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

class RegressionFlow(FlowSpec):

    data_path = Parameter('data_path', default="Data/fact_visits_final_rev01.csv")

    @step
    def start(self):
        # initialize dagshub
        dagshub.init(repo_owner='sakthi-t', repo_name='hba1cmlops', mlflow=True)

        # set the MLFLOW tracking URI
        mlflow.set_tracking_uri("https://dagshub.com/sakthi-t/hba1cmlops.mlflow")

        self.remote_server_uri = "https://dagshub.com/sakthi-t/hba1cmlops.mlflow"
        mlflow.set_tracking_uri(self.remote_server_uri)

        self.tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        mlflow.start_run()
        self.next(self.load_data)

    @step
    def load_data(self):
        self.df_visits = pd.read_csv(self.data_path).copy()
        print(f"Data loaded with shape: {self.df_visits.shape}")
        mlflow.log_param("data_path", self.data_path)
        self.next(self.transform_data)
    

    @step
    def transform_data(self):
        self.data = self.df_visits[['patient_id', 'visited_date', 'sugar', 'hba1c']]

        # Converting visited_date to datetime
        self.data['visited_date'] = pd.to_datetime(self.data['visited_date'])

        # Extracting year, month, and day from visited_date
        self.data['year'] = self.data['visited_date'].dt.year
        self.data['month'] = self.data['visited_date'].dt.month
        self.data['day'] = self.data['visited_date'].dt.day

        # Dropping the original visited date column
        self.data = self.data.drop(columns=['visited_date'])


        # Split data into reference and current datasets
        self.reference_data = self.data.sample(n=1000, replace=False)
        self.current_data = self.data.drop(self.reference_data.index)
        
        self.next(self.define_features_target)

    @step
    def define_features_target(self):
        self.X = self.data.drop(columns=['hba1c'])
        self.y = self.data['hba1c']
        self.next(self.train_test_split)

    @step
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42)
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
        self.next(self.randomforest_param)


    @step
    def randomforest_param(self):
        self.n_estimators = 100
        mlflow.log_param("num of estimators", self.n_estimators)
        self.next(self.fit_randomforest)


    @step
    def fit_randomforest(self):
        # Train your RandomForest model
        self.X_train = np.array(self.X_train, copy=True) # code will fail without this line
        self.y_train = np.array(self.y_train, copy=True) # code will fail without this line
        self.model_rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        self.model_rf.fit(self.X_train, self.y_train)
        self.next(self.metrics_randomforest)

    @step
    def metrics_randomforest(self):  
        self.y_pred_rf = self.model_rf.predict(self.X_test)
        self.rmse_rf = mean_squared_error(self.y_test, self.y_pred_rf, squared=False)
        self.mae_rf = mean_absolute_error(self.y_test, self.y_pred_rf)
        self.r2_rf = r2_score(self.y_test, self.y_pred_rf)

        if self.tracking_url_type_store != "file":
            mlflow.sklearn.log_model(self.model_rf, "model", registered_model_name="random_forest_model")
        else:
            mlflow.sklearn.log_model(self.model_rf, "model")
        mlflow.log_metric("rmse_rf", self.rmse_rf)
        mlflow.log_metric("mae_rf", self.mae_rf)
        mlflow.log_metric("r2_rf", self.r2_rf)

        self.next(self.generate_reports)

    @step
    def generate_reports(self):

        # Add predictions to the reference and current datasets
        self.reference_data['prediction'] = self.model_rf.predict(self.reference_data.drop(columns=['hba1c']))
        self.current_data['prediction'] = self.model_rf.predict(self.current_data.drop(columns=['hba1c']))

        # Rename the 'hba1c' column to 'target' as required by Evidently metrics
        self.reference_data.rename(columns={'hba1c': 'target'}, inplace=True)
        self.current_data.rename(columns={'hba1c': 'target'}, inplace=True)

        
        # Evidently AI monitoring and reports
        # Data Drift Report
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=self.reference_data, current_data=self.current_data)
        drift_report.save_html("data_drift_report.html")
        
        # Regression Performance Report
        reg_report = Report(metrics=[RegressionPreset()])
        reg_report.run(reference_data=self.reference_data, current_data=self.current_data)
        reg_report.save_html("regression_performance_report.html")

        # Log reports to MLflow
        mlflow.log_artifact("data_drift_report.html")
        mlflow.log_artifact("regression_performance_report.html")

        self.next(self.end)
        
    
    @step
    def end(self):
        
        print(f"RandomForest - RMSE: {self.rmse_rf}, MAE: {self.mae_rf}, R2: {self.r2_rf}")
        mlflow.end_run()

if __name__ == "__main__":
    RegressionFlow()
