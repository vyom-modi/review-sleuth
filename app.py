import re
from flask import Flask, render_template, request, session, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

app = Flask(__name__)
app.secret_key = 'irl'

# Load the Naive Bayes model
model = joblib.load('best_models/naive_bayes.pkl')

# Preprocess text function
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text)
    text = text.replace('READ MORE', '')
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r':\)|:\(|:\D|:\S', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_text = [word for word in words if word not in stop_words]
    filtered_text = [lemmatizer.lemmatize(word) for word in filtered_text]
    return " ".join(filtered_text)

@app.route("/", methods=["GET", "POST"])
def home():
    # Initialize predictions list in session if not already initialized
    if "predictions" not in session:
        session["predictions"] = []

    if request.method == "POST":
        # Retrieve review text from form
        review_text = request.form.get("review_text", "")
        
        # Check if the review text is empty
        if not review_text.strip():
            return render_template("index.html", predictions=session["predictions"], error_message="Please enter a review text.")
        
        # Preprocess the review text
        clean_text = preprocess_text(review_text)
        
        # Predict sentiment
        prediction = model.predict([clean_text])[0]

        # Append review and sentiment to predictions list in session
        session["predictions"].append((review_text, prediction))

        print("Session before appending:", session["predictions"])

        # Save the updated session
        session.modified = True

        print("Session after appending:", session["predictions"])
        
    # Render template with predictions
    return render_template("index.html", predictions=session["predictions"])

@app.route("/clear_session", methods=["POST"])
def clear_session():
    session.clear()  # Clear session data
    return jsonify({"message": "Session data cleared"}), 200

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
#     app.run(debug=True)
