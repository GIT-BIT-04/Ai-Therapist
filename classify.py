import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, Normalizer
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

#Hi tests

file_path = '/Users/yadhu/PROJECTS/cape-THERAPIST/datasets/archive (5)/val.txt'
val_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])
file_path = '/Users/yadhu/PROJECTS/cape-THERAPIST/datasets/archive (5)/train.txt'
test_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])
file_path = '/Users/yadhu/PROJECTS/cape-THERAPIST/datasets/archive (5)/test.txt'
train_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])

train_df.info()
print('-----------------------------------------------------------------------')
test_df.info()
print('-----------------------------------------------------------------------')
val_df.info()

print(val_df['Emotion'].value_counts())

percentage_appearance = val_df['Emotion'].value_counts(normalize=True) * 100
print(percentage_appearance)
print('-----------------------------------------------------------------------')
percentage_appearance = train_df['Emotion'].value_counts(normalize=True) * 100
print(percentage_appearance)
print('-----------------------------------------------------------------------')
percentage_appearance = test_df['Emotion'].value_counts(normalize=True) * 100
print(percentage_appearance)


val_df = val_df.drop(val_df[(val_df['Emotion'] == 'surprise') | (val_df['Emotion'] == 'love')].index)
test_df = test_df.drop(test_df[(test_df['Emotion'] == 'surprise') | (test_df['Emotion'] == 'love')].index)
train_df = train_df.drop(train_df[(train_df['Emotion'] == 'surprise') | (train_df['Emotion'] == 'love')].index)
print('------------------------- dropped columns ---------------------------------')
print(train_df['Emotion'].value_counts())

print('------------------------- dropped columns ---------------------------------')
print(test_df['Emotion'].value_counts())

print('------------------------- dropped columns ---------------------------------')
print(val_df['Emotion'].value_counts())

val_df['text_length'] = val_df['Text'].apply(len)
idx_max_length = val_df['text_length'].idxmax()
emotion = val_df.loc[idx_max_length, 'Emotion']
txt = val_df.loc[idx_max_length, 'Text']
print('the text is: ' + txt)
print('the emotion is: ' + emotion)
print('------------------------------------------------------------------')
nltk.download('stopwords')
print(stopwords.words('english'))

class TextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, lower=False, stem=False):
        self.lower = lower
        self.stem = stem
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def text_processing(text):
            processed_text = re.sub('[^a-zA-Z]', ' ', text) # remove any non-alphabet characters
            if self.lower:
                processed_text = processed_text.lower()
            processed_text = processed_text.split()
            if self.stem:
                ps = PorterStemmer()
                processed_text = [ps.stem(word) for word in processed_text if word not in set(stopwords.words('english'))]
            processed_text = ' '.join(processed_text)
            return processed_text
        
        return [text_processing(text) for text in X]

def plot_confusion_matrices(train_true, train_pred, val_true, val_pred, test_true, test_pred, labels):
    # Create confusion matrices
    train_conf_matrix = confusion_matrix(train_true, train_pred)
    val_conf_matrix = confusion_matrix(val_true, val_pred)
    test_conf_matrix = confusion_matrix(test_true, test_pred)

    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Train Confusion Matrix
    sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Train Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Validation Confusion Matrix
    sns.heatmap(val_conf_matrix, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Validation Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    # Test Confusion Matrix
    sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=labels, yticklabels=labels, ax=axes[2])
    axes[2].set_title('Test Confusion Matrix')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')

    plt.tight_layout()
    plt.show()
print('--------------------- Random Forest Classifier -------------------------------')
from sklearn.ensemble import RandomForestClassifier

text_processor = TextProcessor(lower=True, stem=False)
vectorizer = CountVectorizer(max_features=3000)
RF = RandomForestClassifier(
    n_estimators=50, random_state=42, n_jobs=-1, verbose=1
)

pipeline = Pipeline([
    ("text_processing", text_processor), # Text processing step
    ("vectorizer", CountVectorizer()),   # CountVectorizer step
    ("classifier", RF)  # RandomForestClassifier step
])

pipeline.fit(train_df['Text'], train_df['Emotion'])  

process_pip = Pipeline(
    [
        ("text_processing", text_processor),
    ]
)

x_train_procceced = process_pip.fit_transform(train_df['Text'])
x_test_procceced = process_pip.transform(test_df['Text'])
x_val_procceced = process_pip.transform(val_df['Text'])


train_pred = pipeline.predict(x_train_procceced)
train_accuracy = accuracy_score(train_df['Emotion'], train_pred)
print("Train set accuracy:", train_accuracy)

# Validation set accuracy
val_pred = pipeline.predict(x_val_procceced)
val_accuracy = accuracy_score(val_df['Emotion'], val_pred)
print("Validation set accuracy:", val_accuracy)

# Test set accuracy
test_pred = pipeline.predict(x_test_procceced)
test_accuracy = accuracy_score(test_df['Emotion'], test_pred)
print("Test set accuracy:", test_accuracy)

RF_acc = test_accuracy

plot_confusion_matrices(
    train_df['Emotion'], train_pred,
    val_df['Emotion'], val_pred,
    test_df['Emotion'], test_pred,
    pipeline.classes_
)

print('---------------------------- SVC -----------------------------------')

from sklearn.svm import SVC
text_processor = TextProcessor(lower=True, stem=False)
vectorizer = CountVectorizer(max_features=3000)
svm = SVC(kernel="linear",gamma=1, C=.5, random_state=42)
svm_pipeline = Pipeline(
    [
        ("text_processing", text_processor),
        ("vectorizer", vectorizer),
        ("svm", svm),
    ]
)
svm_pipeline.fit(train_df['Text'], train_df['Emotion']) 

train_pred = svm_pipeline.predict(x_train_procceced)
train_accuracy = accuracy_score(train_df['Emotion'], train_pred)
print("Train set accuracy:", train_accuracy)

# Validation set accuracy
val_pred = svm_pipeline.predict(x_val_procceced)
val_accuracy = accuracy_score(val_df['Emotion'], val_pred)
print("Validation set accuracy:", val_accuracy)

# Test set accuracy
test_pred = svm_pipeline.predict(x_test_procceced)
test_accuracy = accuracy_score(test_df['Emotion'], test_pred)
print("Test set accuracy:", test_accuracy)

svm_acc= test_accuracy

plot_confusion_matrices(
    train_df['Emotion'], train_pred,
    val_df['Emotion'], val_pred,
    test_df['Emotion'], test_pred,
    pipeline.classes_
)

print('--------------------------- Logistic Regression ---------------------------')
from sklearn.linear_model import LogisticRegression
text_processor = TextProcessor(lower=True, stem=False)

vectorizer = CountVectorizer(max_features=3000)

logistics = LogisticRegression(random_state=42, max_iter=1000)

logs_pipeline = Pipeline([
    ("text_processing", text_processor), # Text processing step
    ("vectorizer", CountVectorizer()),   # CountVectorizer step
    ("classifier", logistics)  # RandomForestClassifier step
])

logs_pipeline.fit(train_df['Text'], train_df['Emotion'])

train_pred = logs_pipeline.predict(x_train_procceced)
train_accuracy = accuracy_score(train_df['Emotion'], train_pred)
print("Train set accuracy:", train_accuracy)

# Validation set accuracy
val_pred = logs_pipeline.predict(x_val_procceced)
val_accuracy = accuracy_score(val_df['Emotion'], val_pred)
print("Validation set accuracy:", val_accuracy)

# Test set accuracy
test_pred = logs_pipeline.predict(x_test_procceced)
test_accuracy = accuracy_score(test_df['Emotion'], test_pred)
print("Test set accuracy:", test_accuracy)

logs_acc = test_accuracy

plot_confusion_matrices(
    train_df['Emotion'], train_pred,
    val_df['Emotion'], val_pred,
    test_df['Emotion'], test_pred,
    pipeline.classes_
)

print('---------------------------- MultiNomial NB ------------------------------------')
from sklearn.naive_bayes import MultinomialNB
text_processor = TextProcessor(lower=True, stem=True)

vectorizer = CountVectorizer(max_features=3000)

MNB = MultinomialNB()

MNB_pipeline = Pipeline([
    ("text_processing", text_processor), # Text processing step
    ("vectorizer", CountVectorizer()),   # CountVectorizer step
    ("classifier", MNB)  # RandomForestClassifier step
])

MNB_pipeline.fit(train_df['Text'], train_df['Emotion'])

train_pred = MNB_pipeline.predict(x_train_procceced)
train_accuracy = accuracy_score(train_df['Emotion'], train_pred)
print("Train set accuracy:", train_accuracy)

# Validation set accuracy
val_pred = MNB_pipeline.predict(x_val_procceced)
val_accuracy = accuracy_score(val_df['Emotion'], val_pred)
print("Validation set accuracy:", val_accuracy)

# Test set accuracy
test_pred = MNB_pipeline.predict(x_test_procceced)
test_accuracy = accuracy_score(test_df['Emotion'], test_pred)
print("Test set accuracy:", test_accuracy)

MNB_acc = test_accuracy

plot_confusion_matrices(
    train_df['Emotion'], train_pred,
    val_df['Emotion'], val_pred,    
    test_df['Emotion'], test_pred,
    pipeline.classes_
)


print('----------------------------- Gradient Boosting Classifier ------------------------------')
from sklearn.ensemble import GradientBoostingClassifier
text_processor = TextProcessor(lower=True, stem=False)

vectorizer = CountVectorizer(max_features=3000)

GB = GradientBoostingClassifier()

GB_pipeline = Pipeline([
    ("text_processing", text_processor), # Text processing step
    ("vectorizer", CountVectorizer()),   # CountVectorizer step
    ("classifier", GB)  # RandomForestClassifier step
])

GB_pipeline.fit(train_df['Text'], train_df['Emotion'])

train_pred = GB_pipeline.predict(x_train_procceced)
train_accuracy = accuracy_score(train_df['Emotion'], train_pred)
print("Train set accuracy:", train_accuracy)

# Validation set accuracy
val_pred = GB_pipeline.predict(x_val_procceced)
val_accuracy = accuracy_score(val_df['Emotion'], val_pred)
print("Validation set accuracy:", val_accuracy)

# Test set accuracy
test_pred = GB_pipeline.predict(x_test_procceced)
test_accuracy = accuracy_score(test_df['Emotion'], test_pred)
print("Test set accuracy:", test_accuracy)

GB_acc = test_accuracy


plot_confusion_matrices(
    train_df['Emotion'], train_pred,
    val_df['Emotion'], val_pred,
    test_df['Emotion'], test_pred,
    pipeline.classes_
)

accuracies = {
    "Multi Naive Bayes": MNB_acc,
    "SVM": svm_acc,
    "Random Forest": RF_acc,
    "Logistic Regression": logs_acc,
    "Gradient Boosting": GB_acc
}

# Sort accuracies in descending order
sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)

# Extract model names and accuracies
model_names = [model[0] for model in sorted_accuracies]
accuracy_values = [model[1] for model in sorted_accuracies]
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=model_names, y=accuracy_values, palette="YlGnBu")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy of Models")
plt.xticks(rotation=45, ha='right')

# Annotate each bar with its accuracy value
for i, v in enumerate(accuracy_values):
    ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()


print('-------------------- voting classifier ----------------------')
from sklearn.ensemble import VotingClassifier

estimators=[
        ("RFC", pipeline),
        ("Logistics Regression", logs_pipeline),
        ("Gradient Boosting",GB_pipeline),
        ("SVM", svm_pipeline)]


voting_classifier = VotingClassifier(estimators, voting='hard')
voting_classifier.fit(train_df['Text'], train_df['Emotion'])

train_pred = voting_classifier.predict(x_train_procceced)
train_accuracy = accuracy_score(train_df['Emotion'], train_pred)
print("Train set accuracy:", train_accuracy)

# Validation set accuracy
val_pred = voting_classifier.predict(x_val_procceced)
val_accuracy = accuracy_score(val_df['Emotion'], val_pred)
print("Validation set accuracy:", val_accuracy)

# Test set accuracy
test_pred = voting_classifier.predict(x_test_procceced)
test_accuracy = accuracy_score(test_df['Emotion'], test_pred)
print("Test set accuracy:", test_accuracy)

model_accuracy = test_accuracy

print('---------------------- Model Accuracy -----------------------')

print('Our Machine Learning model has an accuracy of {:.2f}%'.format(model_accuracy * 100))

custom_text = "it feels like my career is slipping away from me and i don’t know how to fix it"
predicted_emotion = voting_classifier.predict([custom_text])
print("Predicted Emotion:", predicted_emotion[0])

custom_text = "I feel overwhelmed with sorrow"
predicted_emotion = voting_classifier.predict([custom_text])
print("Predicted Emotion:", predicted_emotion[0])