from metaflow import FlowSpec, step, Parameter, card, current
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class RegressionFlow(FlowSpec):

    data_path = Parameter('data_path', default="Data/fact_visits_final_rev01.csv")

    @card
    @step
    def start(self):
        self.next(self.load_data)

    @card
    @step
    def load_data(self):
        self.df_visits = pd.read_csv(self.data_path).copy()
        print(f"Data loaded with shape: {self.df_visits.shape}")
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
        self.next(self.elasticnet_param)

    @step
    def elasticnet_param(self):
        self.alpha = 0.5
        self.l1_ratio = 0.7
        self.next(self.fit_elasticnet)

    @step
    def fit_elasticnet(self):
        # Ensure the arrays are writable by copying them
        self.X_train = np.array(self.X_train, copy=True)
        self.y_train = np.array(self.y_train, copy=True)

        self.model_elastic = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        self.model_elastic.fit(self.X_train, self.y_train)
        self.next(self.predict_elasticnet)

    @step
    def predict_elasticnet(self):
        self.y_pred_elastic = self.model_elastic.predict(self.X_test)
        self.next(self.metrics_elasticnet)

    @step
    def metrics_elasticnet(self):
        self.rmse_elastic = mean_squared_error(self.y_test, self.y_pred_elastic, squared=False)
        self.mae_elastic = mean_absolute_error(self.y_test, self.y_pred_elastic)
        self.r2_elastic = r2_score(self.y_test, self.y_pred_elastic)
        self.next(self.randomforest_param)

    @step
    def randomforest_param(self):
        self.n_estimators = 100
        self.next(self.fit_randomforest)

    @step
    def fit_randomforest(self):
        self.X_train = np.array(self.X_train, copy=True)
        self.y_train = np.array(self.y_train, copy=True)

        self.model_rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        self.model_rf.fit(self.X_train, self.y_train)
        self.next(self.predict_randomforest)

    @step
    def predict_randomforest(self):
        self.y_pred_rf = self.model_rf.predict(self.X_test)
        self.next(self.metrics_randomforest)

    @step
    def metrics_randomforest(self):
        self.rmse_rf = mean_squared_error(self.y_test, self.y_pred_rf, squared=False)
        self.mae_rf = mean_absolute_error(self.y_test, self.y_pred_rf)
        self.r2_rf = r2_score(self.y_test, self.y_pred_rf)
        self.next(self.end)

    @card
    @step
    def end(self):
        print(f"ElasticNet - RMSE: {self.rmse_elastic}, MAE: {self.mae_elastic}, R2: {self.r2_elastic}")
        print(f"RandomForest - RMSE: {self.rmse_rf}, MAE: {self.mae_rf}, R2: {self.r2_rf}")

        # card_content = f"""
        # Regression Results

        ## ElasticNet
       # - RMSE: {self.rmse_elastic}
       # - MAE: {self.mae_elastic}
       # - R2: {self.r2_elastic}

        ## RandomForest
       # - RMSE: {self.rmse_rf}
       # - MAE: {self.mae_rf}
       # - R2: {self.r2_rf}
       # """
        # current.card.append(
         #   Markdown(card_content)
         #   )
        # current.card["results"] = card_content



if __name__ == "__main__":
    RegressionFlow()
