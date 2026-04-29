from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd
import yaml
import joblib




def train_test_split_data(df:pd.DataFrame):
    x_train,x_test,y_train,y_test=train_test_split(df.drop(columns=["ROI_Score"]), df["ROI_Score"], test_size=params['test_size'], random_state=params['random_state'])
    return [x_train, x_test, y_train, y_test]



def train_model(lst:list):
    x_train, x_test, y_train, y_test=lst
    model=XGBRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'], random_state=params['random_state'], sampling_method=params['sampling_method'], gamma=params['gamma'])
    model.fit(x_train, y_train)
    joblib.dump(model, params['model_path'])
    
    
def main():
    params=yaml.safe_load(open("params.yaml"))
    dataset=pd.read_csv(params['processed_data_path'])
    lst=train_test_split_data(dataset)
    train_model(lst)
    
if __name__=="__main__":
    main()
    
    
