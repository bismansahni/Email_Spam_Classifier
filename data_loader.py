import pandas as pd

def load_data():
    # Load the datasets
    train_df = pd.read_csv('train_set.csv')
    val_df = pd.read_csv('val_set.csv')
    test_df = pd.read_csv('test_set.csv')
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = load_data()
    print("Training Set:")
    print(train_df.head())
    print("\nValidation Set:")
    print(val_df.head())
    print("\nTest Set:")
    print(test_df.head())
