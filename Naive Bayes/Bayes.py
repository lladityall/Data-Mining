import nltk

# Download the 'punkt_tab' resource if not already present
nltk.download('punkt_tab')
nltk.download('stopwords')
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Uncomment the first time to download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# Load dataset
df = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'])

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing
df['clean_message'] = df['message'].apply(preprocess_text)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_message'])
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label='spam'))
print("Recall:", recall_score(y_test, y_pred, pos_label='spam'))
print("F1 Score:", f1_score(y_test, y_pred, pos_label='spam'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict new message
new_message = ["Win a free iPhone now!"]
new_clean = [preprocess_text(msg) for msg in new_message]
new_vector = vectorizer.transform(new_clean)
new_pred = model.predict(new_vector)
print(f"\nPrediction for new message: '{new_message[0]}' => {new_pred[0]}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
