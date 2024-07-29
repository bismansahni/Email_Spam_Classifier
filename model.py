from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib
import tensorflow as tf

def train_model(X_train, y_train, X_val, y_val):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=800, batch_size=10, validation_data=(X_val, y_val))

    return model

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy

def save_model_and_vectorizer(model, vectorizer):
    model.save('spam_classifier_model.h5')
    joblib.dump(vectorizer, 'vectorizer.pkl')

def load_model_and_vectorizer():
    model = keras.models.load_model('spam_classifier_model.h5')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

def predict_sample(model, vectorizer, sample_text):
    import preprocessing
    sample_text_processed = preprocessing.preprocess_text(sample_text)
    sample_vectorized = vectorizer.transform([sample_text_processed]).toarray()
    prediction = model.predict(sample_vectorized)
    return 'spam' if prediction >= 0.5 else 'not spam'

if __name__ == "__main__":
    import data_loader
    import preprocessing

    # Load datasets
    train_df, val_df, test_df = data_loader.load_data()

    # Preprocess datasets
    train_df = preprocessing.preprocess_dataframe(train_df)
    val_df = preprocessing.preprocess_dataframe(val_df)
    test_df = preprocessing.preprocess_dataframe(test_df)

    # Vectorize datasets
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df['combined_text']).toarray()
    y_train = train_df['type'].apply(lambda x: 1 if x == 'spam' else 0).values

    X_val = vectorizer.transform(val_df['combined_text']).toarray()
    y_val = val_df['type'].apply(lambda x: 1 if x == 'spam' else 0).values

    X_test = vectorizer.transform(test_df['combined_text']).toarray()
    y_test = test_df['type'].apply(lambda x: 1 if x == 'spam' else 0).values

    # Train model
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)

    # Save model and vectorizer
    save_model_and_vectorizer(model, vectorizer)

    print(f"Accuracy: {accuracy}")
