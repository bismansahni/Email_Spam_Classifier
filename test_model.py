import model

def test_sample(sample_title, sample_text):
    # Load the pre-trained model and vectorizer
    trained_model, vectorizer = model.load_model_and_vectorizer()
    
    # Combine title and text
    combined_text = sample_title + " " + sample_text
    
    # Predict the class of the sample text
    prediction = model.predict_sample(trained_model, vectorizer, combined_text)
    print(f"Prediction for sample text: '{sample_text}' is: {prediction}")

if __name__ == "__main__":
    sample_title = "Affordable American MBA degree ($180month)"
    sample_text="Today more than ever you need to upskill and reskill as the global job market gets increasingly competitive. We hope these payment options will bring down any barriers standing in your way. Nexford prides itself on being the most affordable American university in the world. The Nexford academic model allows you to finish as fast as you want, and the faster you finish, the less you pay. Our tuition fees are calculated monthly ($180/month for an MBA), so the faster you complete your degree the less the degree will cost you. Most students complete our MBA in an average of 18 months (average total cost $3,240). Some completed in 12 to 14 months. Specialize in the world's hottest sectors like AI, Data Science, Cybersecurity, E-Commerce, etc. Upon graduation, you'll walk away with a specialization certificate on top of your Master's in Business Administration (MBA) degree. To find out more about applying to Nexford University, or anything else, sign up and we will be in touch with you. Nexford University is accredited by the Distance Education Accrediting Commission (DEAC). The DEAC is listed by the U.S. Department of Education as a recognized accrediting agency and is recognized by the Council for Higher Education Accreditation (CHEA) So what are you waiting for? Join a next-generation university today."
  
    test_sample(sample_title, sample_text)
