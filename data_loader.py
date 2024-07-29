from datasets import load_dataset
import pandas as pd

def load_data():
    dataset = load_dataset("TrainingDataPro/email-spam-classification")
    train_dataset = dataset['train']
    df = pd.DataFrame(train_dataset)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
