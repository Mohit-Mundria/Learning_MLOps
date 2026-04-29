import pandas as pd 
from sklearn.model_selection import train_test_split
import os


def data_load(path:str):
    dataset=pd.read_csv(path)
    dataset.to_csv("D:\\End to end project\\Learning_MLOps\data_save\\raw\\raw_dataset.csv", index= False)
    return dataset


# def train_test_splitdata(dataset:pd.DataFrame):
#     input_columns=dataset.iloc[:,:-1]
#     output_columns=dataset.iloc[:,-1]
#     train_data_input,test_data_input,train_data_output,test_data_output=train_test_split(input_columns,output_columns,random_state=42, test_size=0.2)
#     path="D:\End to end project\Learning_MLOps\data_save"
#     train_data_input.to_csv(os.path.join(path, "train_data_input.csv"), index=False)
#     test_data_input.to_csv(os.path.join(path, "test_data_input.csv"), index=False)
#     train_data_output.to_csv(os.path.join(path, "train_data_output.csv"), index=False)
#     test_data_output.to_csv(os.path.join(path, "test_data_output.csv"), index=False)
    
    
def main():
    path="D:\\Downloads\\agentic_reality.csv"
    data_load(path)
    
    
    
if __name__=="__main__":
    main()