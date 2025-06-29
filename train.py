import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading dataset...")
df = pd.read_csv('dataset.csv')

print("Cleaning text...")
df['clean_text'] = df['text'].apply(nfx.remove_stopwords).apply(nfx.remove_punctuations).str.lower()

print("Plotting distribution...")
plt.figure(figsize=(8, 4))
sns.countplot(x='emotion', data=df, palette="Set2")
plt.title("Emotion Distribution")
plt.savefig("emotion_distribution.png")
plt.close()

print("Vectorizing...")
X = df['clean_text']
y = df['emotion']
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_vect = vectorizer.fit_transform(X)

print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, stratify=y, random_state=42)

print("Training model...")
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)

#  Print classification report
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Print accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {acc * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="coolwarm", values_format="d")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# Save model & vectorizer
print("\n Saving model and vectorizer...")
dump(model, 'emotion_model.pkl')
dump(vectorizer, 'emotion_vectorizer.pkl')

print(" Done. All files saved.")
