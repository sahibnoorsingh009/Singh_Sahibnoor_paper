# -*- coding: utf-8 -*-
"""OLID_dat_training .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17Bg5_BnvkJWncmzj9kDspreF-9r1NUxD
"""

import os
import pandas as pd
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import nltk

from nltk.tokenize import word_tokenize

# Download NLTK tokenizer
nltk.download('punkt_tab')
# Load Data
file_path = '/OLID.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    raise FileNotFoundError(f"File not found: {file_path}")


# Ensure column exists
if "cleaned_tweet" not in df.columns:
    raise KeyError("Column 'cleaned_tweet' not found in dataset")

# Drop missing values
df = df.dropna(subset=["cleaned_tweet"])


# Tokenize text
df["tokenized"] = df["cleaned_tweet"].apply(lambda x: word_tokenize(str(x).lower()))

# Train Word2Vec model
word2vec_model = Word2Vec(
    sentences=df["tokenized"],
    vector_size=100,  # Each word is represented by a 100-dimensional vector
    window=5,         # Context window size
    min_count=2,      # Ignore words with fewer than 2 occurrences
    sg=1,             # Skip-gram model (1=skip-gram, 0=CBOW)
    workers=4         # Use multiple CPU cores
)

# Save model for later use
word2vec_model.save("word2vec_model.bin")

import numpy as np
from sklearn.decomposition import PCA

# Extract word vectors
words = list(word2vec_model.wv.index_to_key)  # Get all unique words
word_vectors = np.array([word2vec_model.wv[word] for word in words])

# Apply PCA (reduce to 2D)
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

# Plot word vectors
plt.figure(figsize=(12, 8))
plt.scatter(word_vectors_2d[:30, 0], word_vectors_2d[:30, 1], marker="o", color="blue")

# Add word labels
for i, word in enumerate(words[:30]):  # Plot first 50 words only
    plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]), fontsize=12)

plt.title("Word2Vec Visualization using PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(False)
plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# Load Data
file_path = '/OLID.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    raise FileNotFoundError(f"File not found: {file_path}")

# Clean and tokenize words
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and special characters
    words = text.split()  # Tokenize by space
    return words

# Apply text preprocessing to the 'cleaned_tweet' column
df['tokens'] = df['cleaned_tweet'].astype(str).apply(preprocess_text)

# Flatten list of all words
all_words = [word for tokens in df['tokens'] for word in tokens]

# Count word frequencies
word_counts = Counter(all_words)
top_words = word_counts.most_common(20)  # Get top 20 words

# Plot word frequency
words, counts = zip(*top_words)
plt.figure(figsize=(12, 6))
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 20 Most Frequent Words in Cleaned Tweets")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud



# 2. Most Frequent Word
df['cleaned_tweet'] = df['cleaned_tweet'].astype(str).fillna("")

all_words = " ".join(df['cleaned_tweet'])
word_counts = Counter(all_words.split())
most_common_words = word_counts.most_common(20)

words, counts = zip(*most_common_words)
plt.figure(figsize=(10,5))
sns.barplot(x=list(words), y=list(counts))
plt.xticks(rotation=45)
plt.title("Most Frequent Words")
plt.xlabel("Words")
plt.ylabel("Count")
plt.show()

# 1. Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df['subtask_a'])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 3. Word Cloud
df['cleaned_tweet'] = df['cleaned_tweet'].astype(str).fillna("")

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Most Common Words")
plt.show()

from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = "/OLID.csv"
df = pd.read_csv(file_path)

# Filter tweets based on label
tweets_0 = " ".join(df[df["subtask_a"] == "NOT"]["cleaned_tweet"].dropna())
tweets_1 = " ".join(df[df["subtask_a"] == "OFF"]["cleaned_tweet"].dropna())

# Generate word frequency counts
word_counts_0 = Counter(tweets_0.split())
word_counts_1 = Counter(tweets_1.split())

# Get the most common words for each category
top_n = 40 # Number of top words to display

common_words_0 = word_counts_0.most_common(top_n)
common_words_1 = word_counts_1.most_common(top_n)

# Convert to DataFrame for visualization
df_words_0 = pd.DataFrame(common_words_0, columns=["Word", "Count"])
df_words_1 = pd.DataFrame(common_words_1, columns=["Word", "Count"])

# Plot bar graphs
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.barh(df_words_0["Word"], df_words_0["Count"], color="blue")
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.xlabel("Frequency")
plt.title("Top Words in Tweets Labeled as 'NOT' (0)")

plt.subplot(1, 2, 2)
plt.barh(df_words_1["Word"], df_words_1["Count"], color="red")
plt.gca().invert_yaxis()
plt.xlabel("Frequency")
plt.title("Top Words in Tweets Labeled as 'OFF' (1)")

plt.tight_layout()
plt.show()

from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the dataset
file_path = "/OLID.csv"
df = pd.read_csv(file_path)

# Filter tweets based on label
tweets_0 = " ".join(df[df["subtask_a"] == "NOT"]["cleaned_tweet"].dropna())
tweets_1 = " ".join(df[df["subtask_a"] == "OFF"]["cleaned_tweet"].dropna())

# Generate word frequency counts
word_counts_0 = Counter(tweets_0.split())
word_counts_1 = Counter(tweets_1.split())

# Get the most common words for each category
top_n = 15  # Number of top words to display

common_words_0 = word_counts_0.most_common(top_n)
common_words_1 = word_counts_1.most_common(top_n)

# Convert to DataFrame for visualization
df_words_0 = pd.DataFrame(common_words_0, columns=["Word", "Count"])
df_words_1 = pd.DataFrame(common_words_1, columns=["Word", "Count"])

# Plot bar graphs
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.barh(df_words_0["Word"], df_words_0["Count"], color="blue")
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.xlabel("Frequency")
plt.title("Top Words in Tweets Labeled as 'NOT' (0)")

plt.subplot(1, 2, 2)
plt.barh(df_words_1["Word"], df_words_1["Count"], color="red")
plt.gca().invert_yaxis()
plt.xlabel("Frequency")
plt.title("Top Words in Tweets Labeled as 'OFF' (1)")

plt.tight_layout()
plt.show()

# LDA Topic Modeling
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X_text = vectorizer.fit_transform(df['cleaned_tweet'].dropna())

# Define the number of topics
num_topics = 5

# Train LDA model
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(X_text)

# Get the top words for each topic
words = vectorizer.get_feature_names_out()
topic_words = {}

for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [words[i] for i in topic.argsort()[-10:]]  # Top 10 words per topic
    topic_words[f"Topic {topic_idx+1}"] = top_words

# Convert to DataFrame for visualization
topic_df = pd.DataFrame(topic_words)

# Display the topics
print(topic_df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

# Load Data (Ensure 'cleaned_tweet' column exists in df)
df['cleaned_tweet'] = df['cleaned_tweet'].astype(str).fillna("")  # Convert to string and handle NaN

# Convert Tweets into Numerical Representation (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['cleaned_tweet'])

# Apply K-Means Clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_tfidf)

# Get Cluster Counts
plt.figure(figsize=(6, 4))
sns.countplot(x=df['cluster'])
plt.title("Number of Tweets per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# Get Top Words in Each Cluster
def get_top_words(cluster_num, n_words=10):
    cluster_indices = np.where(df['cluster'] == cluster_num)[0]
    words = " ".join(df.iloc[cluster_indices]['cleaned_tweet'].dropna().astype(str)).split()
    common_words = Counter(words).most_common(n_words)
    return [word for word, count in common_words]

# Display Top Words in Each Cluster
for i in range(num_clusters):
    print(f"\n🔹 Top words in Cluster {i}: {get_top_words(i)}")

# Assign Cluster Labels (Manual Step)
cluster_mapping = {0: "Liberal", 1: "Conservative"}  # Modify based on analysis
df['political_leaning'] = df['cluster'].map(cluster_mapping)

# Show Sample Results
print("\n🔹 Sample Clustered Tweets:")
print(df[['cleaned_tweet', 'political_leaning']].sample(10))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

# Load Data (Ensure 'cleaned_tweet' column exists in df)
df['cleaned_tweet'] = df['cleaned_tweet'].astype(str).fillna("")  # Convert to string and handle NaN

# Convert Tweets into Numerical Representation (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['cleaned_tweet'])

from scipy.cluster.hierarchy import linkage, fcluster
import scipy.cluster.hierarchy as sch

# Perform hierarchical clustering
Z = linkage(X_tfidf.toarray(), method='ward')

# Choose number of clusters
df['cluster'] = fcluster(Z, t=2, criterion='maxclust')  # 2 clusters


# Get Cluster Counts
plt.figure(figsize=(6, 4))
sns.countplot(x=df['cluster'])
plt.title("Number of Tweets per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# Get Top Words in Each Cluster
def get_top_words(cluster_num, n_words=10):
    cluster_indices = np.where(df['cluster'] == cluster_num)[0]
    words = " ".join(df.iloc[cluster_indices]['cleaned_tweet'].dropna().astype(str)).split()
    common_words = Counter(words).most_common(n_words)
    return [word for word, count in common_words]

# Display Top Words in Each Cluster
for i in range(num_clusters):
    print(f"\n🔹 Top words in Cluster {i}: {get_top_words(i)}")

# Assign Cluster Labels (Manual Step)
cluster_mapping = {0: "Liberal", 1: "Conservative"}  # Modify based on analysis
df['political_leaning'] = df['cluster'].map(cluster_mapping)

# Show Sample Results
print("\n🔹 Sample Clustered Tweets:")
print(df[['cleaned_tweet', 'political_leaning']].sample(10))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Ensure the 'cleaned_tweet' column exists
df['cleaned_tweet'] = df['cleaned_tweet'].astype(str)

# Convert Tweets into TF-IDF Features
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['cleaned_tweet'])

# Apply K-Means Clustering (2 clusters: Liberal & Conservative)
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_tfidf)

# Get Top Words in Each Cluster (to Identify Labels)
def get_top_words(cluster_num, n_words=10):
    cluster_indices = np.where(df['cluster'] == cluster_num)[0]
    words = " ".join(df.iloc[cluster_indices]['cleaned_tweet']).split()
    common_words = Counter(words).most_common(n_words)
    return [word for word, count in common_words]

# Display Top Words in Each Cluster
for i in range(num_clusters):
    print(f"\n🔹 Top words in Cluster {i}: {get_top_words(i)}")

# Assign Cluster Labels Based on Common Words (Manual Step)
# Modify based on your analysis of common words in each cluster
cluster_mapping = {0: "Liberal", 1: "Conservative"}  # Adjust if needed
df['political_leaning'] = df['cluster'].map(cluster_mapping)

# Convert Labels to Numeric (0 = Liberal, 1 = Conservative)
df['political_label'] = df['political_leaning'].map({"Liberal": 0, "Conservative": 1})

# Show Sample Labeled Data
print("\n🔹 Sample Labeled Tweets:")
print(df[['cleaned_tweet', 'political_leaning']].sample(10))

# Save the labeled dataset for supervised learning
df.to_csv("labeled_tweets.csv", index=False)
print("\n✅ Labeled dataset saved as 'labeled_tweets.csv'!")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Ensure Labels are Encoded (0 = Liberal, 1 = Conservative)
label_encoder = LabelEncoder()
df['political_label'] = label_encoder.fit_transform(df['political_leaning'])  # Convert to numeric labels

# Prepare Data
X = df['cleaned_tweet']
y = df['political_label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert Text to Features (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train XGBoost Model
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric="logloss", use_label_encoder=False)
model.fit(X_train_tfidf, y_train)

# Predictions & Performance
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"🔹 Model Accuracy: {acc:.2f}")

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Function to Predict New Tweets
def predict_political_tweet(tweet):
    tweet_tfidf = tfidf.transform([tweet])  # Convert tweet to TF-IDF features
    prediction = model.predict(tweet_tfidf)[0]  # Predict class
    return label_encoder.inverse_transform([prediction])[0]  # Convert back to label

# Example Usage
new_tweet = "TRUMP IS A leader of USA"
print(f"🔹 Predicted Political Leaning: {predict_political_tweet(new_tweet)}")

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Data
file_path = '/OLID.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    raise FileNotFoundError(f"File not found: {file_path}")

# Encode Labels
label_encoder = preprocessing.LabelEncoder()
df['subtask_a'] = label_encoder.fit_transform(df['subtask_a'])

# Handle Missing Values
df = df.dropna(subset=['cleaned_tweet'])
df['cleaned_tweet'] = df['cleaned_tweet'].astype(str)  # Ensure all values are strings

# Train-Test Split
X = df['cleaned_tweet']
y = df['subtask_a']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert Text to Features (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)  # Limit features for better performance
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Convert Sparse Matrix to Dense for XGBoost
X_train_tfidf = X_train_tfidf
X_test_tfidf = X_test_tfidf

# Train XGBoost Model
model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', tree_method="hist" )
model.fit(X_train_tfidf, y_train)

# Predictions & Performance
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"🔹 XGBoost Model Accuracy: {acc:.2f}")

# Print Confusion Matrix & Classification Report
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load Data
file_path = '/OLID.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    raise FileNotFoundError(f"File not found: {file_path}")

# Encode Labels
label_encoder = preprocessing.LabelEncoder()
df['subtask_a'] = label_encoder.fit_transform(df['subtask_a'])

# Handle Missing Values
df = df.dropna(subset=['cleaned_tweet'])
df['cleaned_tweet'] = df['cleaned_tweet'].astype(str)  # Ensure all values are strings

# Train-Test Split
X = df['cleaned_tweet']
y = df['subtask_a']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert Text to Bag-of-Words
cv = CountVectorizer(max_features=2000)
X_train_bow = cv.fit_transform(X_train).toarray()
X_test_bow = cv.transform(X_test).toarray()  # Transform test data
print(X_train_bow[2])

#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

model = RandomForestClassifier()
model.fit(X_train_bow, y_train)

y_pred = model.predict(X_test_bow)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")
print(confusion_matrix(y_test, y_pred))

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Data
file_path = '/OLID.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    raise FileNotFoundError(f"File not found: {file_path}")

# Encode Labels
label_encoder = preprocessing.LabelEncoder()
df['subtask_a'] = label_encoder.fit_transform(df['subtask_a'])

# Handle Missing Values
df = df.dropna(subset=['cleaned_tweet'])
df['cleaned_tweet'] = df['cleaned_tweet'].astype(str)  # Ensure all values are strings

# Train-Test Split
X = df['cleaned_tweet']
y = df['subtask_a']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert Text to Bag-of-Words
cv = CountVectorizer()
X_train_bow = cv.fit_transform(X_train).toarray()
X_test_bow = cv.transform(X_test).toarray()  # Transform test data
print(X_train_bow[2])

# Nave_bayes

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix

model = MultinomialNB()
model.fit(X_train_bow, y_train)

y_pred = model.predict(X_test_bow)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")
print(confusion_matrix(y_test, y_pred))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load Data
file_path = '/OLID.csv'  # Change this to your file path
df = pd.read_csv(file_path)

# Encode Labels
label_encoder = LabelEncoder()
df['subtask_a'] = label_encoder.fit_transform(df['subtask_a'])

# Handle Missing Values
df = df.dropna(subset=['cleaned_tweet'])
df['cleaned_tweet'] = df['cleaned_tweet'].astype(str)  # Ensure all values are strings

# Train-Test Split
X = df['cleaned_tweet']
y = df['subtask_a']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert Text to Features (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)  # Limit features for better performance
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize Naive Bayes Model
nb_model = MultinomialNB()

# Calculate Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    nb_model, X_train_tfidf, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate Mean and Standard Deviation for Plot
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot Learning Curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue')
plt.plot(train_sizes, test_mean, label='Validation Accuracy', color='green')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='green', alpha=0.2)
plt.title("Learning Curve (Naive Bayes)")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Load Data (Make sure your dataframe 'df' is already loaded)
X = df['cleaned_tweet']
y = df['subtask_a']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert Text to Features (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train XGBoost Model with Evaluation Monitoring
eval_set = [(X_train_tfidf, y_train), (X_test_tfidf, y_test)]
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric="logloss", use_label_encoder=False)

model.fit(X_train_tfidf, y_train, eval_set=eval_set, verbose=True)

# Extract Accuracy for Training and Validation
results = model.evals_result()
train_errors = results['validation_0']['logloss']
test_errors = results['validation_1']['logloss']

# Plot Learning Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_errors) + 1), train_errors, label='Training LogLoss')
plt.plot(range(1, len(test_errors) + 1), test_errors, label='Validation LogLoss')
plt.xlabel('Boosting Iterations')
plt.ylabel('Log Loss')
plt.title('XGBoost Learning Curve')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing

# Load Data
df = pd.read_csv('/OLID.csv')  # Change this to your actual file path

# Encode Labels
label_encoder = preprocessing.LabelEncoder()
df['subtask_a'] = label_encoder.fit_transform(df['subtask_a'])

# Handle Missing Values
df = df.dropna(subset=['cleaned_tweet'])
df['cleaned_tweet'] = df['cleaned_tweet'].astype(str)

# Train-Test Split
X = df['cleaned_tweet']
y = df['subtask_a']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert Text to Features (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Define Classifier
model = RandomForestClassifier(n_estimators=100, random_state=1)

# Compute learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_tfidf, y_train, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)  # 10 points from 10% to 100% of data
)

# Calculate mean and std deviation
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.plot(train_sizes, test_mean, 'o-', color="red", label="Validation accuracy")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve (Random Forest)")
plt.legend(loc="best")
plt.show()