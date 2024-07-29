import model

def test_sample(sample_text):
    # Load the pre-trained model and vectorizer
    trained_model, vectorizer = model.load_model_and_vectorizer()
    
    # Predict the class of the sample text
    prediction = model.predict_sample(trained_model, vectorizer, sample_text)
    print(f"Prediction for sample text: '{sample_text}' is: {prediction}")

if __name__ == "__main__":
    sample_text = "Hi James, Have you claim your complimentary gift yet? I've compiled in here a special astrology gift that predicts everything about you in the future? This is your enabler to take the correct actions now. >> Click here to claim your copy now >> Claim yours now, and thank me later. Love, Heather"
    test_sample(sample_text)
