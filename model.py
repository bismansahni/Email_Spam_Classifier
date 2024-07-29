from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib
import tensorflow as tf

def train_model(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df['type'].apply(lambda x: 1 if x == 'spam' else 0).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)

    # Save the model and vectorizer
    model.save('spam_classifier_model.h5')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return model, vectorizer, accuracy

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

    df = data_loader.load_data()
    df = preprocessing.preprocess_dataframe(df)
    trained_model, vectorizer, accuracy = train_model(df)

    print(f"Accuracy: {accuracy}")
