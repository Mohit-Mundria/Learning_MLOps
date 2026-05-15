# from fastapi import params
# from sklearn.model_selection import train_test_split
import json

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import ElasticNet
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.model_selection import cross_val_score
import os
import pandas as pd
import joblib
import optuna as opt
import mlflow
import dagshub 

from utility_code.utility_func import read_params



# def train_test_split_data(df:pd.DataFrame):
#     x_train,x_test,y_train,y_test=train_test_split(df.drop(columns=["ROI_Score"]), df["ROI_Score"], test_size=params['test_size'], random_state=params['random_state'])
#     return [x_train, x_test, y_train, y_test]



# def train_model(lst:list):
#     x_train, x_test, y_train, y_test=lst
#     model=XGBRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'], random_state=params['random_state'], sampling_method=params['sampling_method'], gamma=params['gamma'])
#     model.fit(x_train, y_train)
#     joblib.dump(model, params['model_path'])
    
    
# def main():
#     params=yaml.safe_load(open("params.yaml"))
#     dataset=pd.read_csv(params['processed_data_path'])
#     lst=train_test_split_data(dataset)
#     train_model(lst)
    
# if __name__=="__main__":
#     main()

def load_training_data(path:str):
    dataset=pd.read_csv(path)
    x_train=dataset.drop(columns=["ROI_Score"])
    y_train=dataset["ROI_Score"]
    return x_train, y_train

def load_testing_data(path:str):
    dataset=pd.read_csv(path)
    x_test=dataset.drop(columns=["ROI_Score"])
    y_test=dataset["ROI_Score"]
    return x_test, y_test

## We can also use optuna for hyperparameter tuning and model selection. Below is the code for that.
## We can add multiple models and their respetive hyperparameters in the objective function and let optuna slect the best model along with the best hyperprameters for that model.

def objective(trial, x_train, y_train):
    # params=read_params(r"D:\End to end project\Learning_MLOps\params.yaml")
    # x_train, y_train=load_training_data(params['preprocessing']['train_data_path'])
    regressor=trial.suggest_categorical("regressor", ["RandomForestRegressor", "XGBRegressor"])
    
    if regressor=="RandomForestRegressor":
        n_estimators=trial.suggest_int("n_estimators", 10, 200)
        max_depth=trial.suggest_int("max_depth", 5, 10)
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10)
        
        model=RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    elif regressor=="XGBRegressor":
        n_estimators=trial.suggest_int("n_estimators", 10, 200)
        max_depth=trial.suggest_int("max_depth", 2, 10)
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.03)
        sampling_method=trial.suggest_categorical("sampling_method", ["uniform", "gradient_based"])
        gamma=trial.suggest_float("gamma", 0, 1)
        
        model=XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, sampling_method=sampling_method, gamma=gamma)

    

    score=cross_val_score(model, x_train, y_train, cv=5, scoring="r2").mean()
    return score

def create_study(objective, x_train, y_train):
    study=opt.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=100)
    return [study.best_trial.params, study.best_trial.value]
    
def mlflow_tracking(model_parameters:dict):
    params=read_params("params.yaml")
    dagshub.init(repo_owner='mundriamohit100', repo_name='Learning_MLOps', mlflow=True)
    x_train, y_train=load_training_data(params['preprocessing']['train_data_path'])
    x_test, y_test=load_testing_data(params['preprocessing']['test_data_path'])
    model_name=model_parameters["regressor"]
    model_params={key: value for key, value in model_parameters.items() if key!="regressor"}
    mlflow.set_experiment("model_selection_and_hyperparameter_tuning_vs1")
    report=[]
    
    # model=model.set_params(**model_params)
    if model_name=="RandomForestRegressor":
        model=RandomForestRegressor(**model_params, random_state=42)
    elif model_name=="XGBRegressor":
        model=XGBRegressor(**model_params, random_state=42)
    
    
    model.fit(x_train, y_train)

    model_training_score= model.score(x_train, y_train)
    
    model_testing_score=cross_val_score(model, x_test, y_test, cv=5, scoring="r2").mean()
    report.append(model_training_score)
    report.append(model_testing_score)
    joblib.dump(model, params['model_training']['model_path'])
    
    with mlflow.start_run(run_name="Final_Model_MLOps_Project"):
        
        mlflow.log_param("model", model_name)
        mlflow.log_params(model_params)
        mlflow.log_metrics({"training_score": model_training_score, "testing_score": model_testing_score})
        mlflow.log_artifact(params['model_training']['model_path'])
        
        mlflow.sklearn.log_model(model, "model_object")
        os.makedirs("reports", exist_ok=True)
        metrics = {
        "training_r2": round(model_training_score, 4),
        "testing_r2":  round(model_testing_score, 4)
    }
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f)
        


def main():
    params=read_params("params.yaml")
    x_train, y_train=load_training_data(params['preprocessing']['train_data_path'])
    model_parameters, model_score=create_study(objective, x_train, y_train)
    mlflow_tracking(model_parameters)
    
if __name__=="__main__":
    main()
    
    
    
        
    
    
