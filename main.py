import data_loader
import preprocessing
import model

def main():
    # Load and preprocess data
    df = data_loader.load_data()
    df = preprocessing.preprocess_dataframe(df)
    
    # Train and evaluate the model
    trained_model, vectorizer, accuracy = model.train_model(df)
    
    # Print results
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
