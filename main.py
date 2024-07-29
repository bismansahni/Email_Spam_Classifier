import data_loader
import preprocessing
import model

def main():
    # Load datasets
    train_df, val_df, test_df = data_loader.load_data()

    # Preprocess datasets
    train_df = preprocessing.preprocess_dataframe(train_df)
    val_df = preprocessing.preprocess_dataframe(val_df)
    test_df = preprocessing.preprocess_dataframe(test_df)

    # Vectorize datasets
    vectorizer = model.TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df['combined_text']).toarray()
    y_train = train_df['type'].apply(lambda x: 1 if x == 'spam' else 0).values

    X_val = vectorizer.transform(val_df['combined_text']).toarray()
    y_val = val_df['type'].apply(lambda x: 1 if x == 'spam' else 0).values

    X_test = vectorizer.transform(test_df['combined_text']).toarray()
    y_test = test_df['type'].apply(lambda x: 1 if x == 'spam' else 0).values

    # Train model
    trained_model = model.train_model(X_train, y_train, X_val, y_val)

    # Evaluate model
    accuracy = model.evaluate_model(trained_model, X_test, y_test)

    # Save model and vectorizer
    model.save_model_and_vectorizer(trained_model, vectorizer)

    # Print results
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
