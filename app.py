from joblib import load
import neattext.functions as nfx

# Load trained model and vectorizer
model = load('emotion_model.pkl')
vectorizer = load('emotion_vectorizer.pkl')

def predict_emotion(text):
    cleaned = nfx.remove_stopwords(nfx.remove_punctuations(text)).lower()
    vect = vectorizer.transform([cleaned])
    return model.predict(vect)[0]

# CLI loop
print("Emotion Detector - Type 'exit' to quit")

while True:
    text = input("Enter a sentence: ")
    if text.lower() == 'exit':
        print("Goodbye!")
        break
    emotion = predict_emotion(text)
    print(f"Predicted Emotion: {emotion}\n")
