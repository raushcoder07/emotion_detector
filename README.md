# 😄 Emotion Detection from Text using Machine Learning

This project is a text-based emotion classification system that predicts emotional tone such as **happy**, **sad**, **angry**, **surprise**, **fear**, **netural**  using Python and machine learning techniques — without using any external APIs.

---

## 🚀 Features

- Detects 5 emotions from user input text:
  - `happy`, `sad`, `angry`, `surprise`, `fear`,`netural`
- Built with `Logistic Regression` and `TF-IDF`
- No internet or API required — fully local!
- Visual output:
  - Emotion distribution (`emotion_distribution.png`)
  - Confusion matrix (`confusion_matrix.png`)
- Modular, extensible, and easy to train with your own data

---

## 🛠️ Tech Stack

| Component     | Library        |
|---------------|----------------|
| ML Model      | `LogisticRegression` |
| Feature Extraction | `TfidfVectorizer` |
| Text Cleaning | `neattext` |
| Evaluation    | `scikit-learn`, `matplotlib`, `seaborn` |
| Model Saving  | `joblib` |
| Deployment    | CLI app (via `app.py`) |

---

 ## 📁 Project Structure
 emotion_detector/
│
├── app.py # Run this to test model predictions
├── train.py # Train model with dataset.csv
├── dataset.csv # Input text-emotion data
├── emotion_model.pkl # Saved ML model
├── emotion_vectorizer.pkl # Saved TF-IDF vectorizer
├── emotion_distribution.png # Emotion count plot
├── confusion_matrix.png # Confusion matrix after training
├── README.md # Project documentation


---

## 🔧 Setup Instructions

### 1. Clone or Download

```bash
git clone https://github.com/yourusername/emotion-detector.git
cd emotion-detector

2. Install Dependencies
    pip install -r requirements.txt
If you don’t have a requirements.txt, create one with:

3. Train the Model
Make sure dataset.csv is present and run:
   python train.py
This will train the model and create the .pkl files and plots.

4. Run the Predictor
     python app.py
Then input any sentence:

Enter a sentence: I'm feeling amazing today!
Predicted Emotion: happy

