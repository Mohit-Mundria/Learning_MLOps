import pandas as pd
import yaml
from sklearn.preprocessing import OneHotEncoder



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

def main():
    # params=yaml.safe_load(open("params.yaml"))
    dataset=load_data(params['data_path'])
    dataset=drop_col(dataset, params['drop_columns'])
    dataset=fill_nan(dataset)
    dataset=one_hot_encoding(dataset)
    dataset.to_csv(params['processed_data_path'], index=False)


if __name__ == "__main__":
    main()
