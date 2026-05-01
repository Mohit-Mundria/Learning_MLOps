import pandas as pd
import yaml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



def load_data(path:str)->pd.DataFrame:
    dataset=pd.read_csv(path)
    return dataset


def drop_col(df:pd.DataFrame, cols:list)->pd.DataFrame:
    for col in cols:
        df.drop(col, axis=1 , inplace=True)
    return df



def fill_nan(df:pd.DataFrame)->pd.DataFrame:
    for col in df.columns:
        if df[col].name == 'Failure_Reason':
            df[col].fillna("No failure", inplace=True)
        elif df[col].dtype=='object':
            df[col].fillna(df[col].mode(), inplace= True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)
    return df



def one_hot_encoding(df:pd.DataFrame)->pd.DataFrame:
    encoder=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoded=encoder.fit_transform(df[params['one_hot_encode_columns']])
    one_hot_encoded_df=pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(params['one_hot_encode_columns']))
    dataset=pd.concat([df.drop(columns=params['one_hot_encode_columns']), one_hot_encoded_df], axis=1)
    
    return dataset


def train_test_split_data(df:pd.DataFrame):
    training_data, testing_data=train_test_split(df, test_size=params['test_size'], random_state=params["random_state"])
    training_data.to_csv(params["train_data_path"], index=False)
    testing_data.to_csv(params["test_data_path"], index=False)

def main():
    # params=yaml.safe_load(open("params.yaml"))
    dataset=load_data(params['data_path'])
    dataset=drop_col(dataset, params['drop_columns'])
    dataset=fill_nan(dataset)
    dataset=one_hot_encoding(dataset)
    train_test_split_data(dataset)


if __name__ == "__main__":
    main()
