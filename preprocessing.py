import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_dataframe(df):
    # Combine title and text
    df['combined_text'] = df['title'] + " " + df['text']
    df['combined_text'] = df['combined_text'].apply(preprocess_text)
    return df

if __name__ == "__main__":
    import data_loader
    train_df, val_df, test_df = data_loader.load_data()
    
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    test_df = preprocess_dataframe(test_df)
    
    print("Training Set:")
    print(train_df.head())
    print("\nValidation Set:")
    print(val_df.head())
    print("\nTest Set:")
    print(test_df.head())
