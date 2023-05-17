import nltk
from nltk.corpus import movie_reviews
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy

# Step 1: Prepare the data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Step 2: Define feature extraction function
def extract_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

# Step 3: Create feature set and split into training and testing data
all_words = nltk.FreqDist(movie_reviews.words())
word_features = list(all_words.keys())[:2000]
feature_set = [(extract_features(doc), category) for (doc, category) in documents]
train_set = feature_set[:1500]
test_set = feature_set[1500:]

# Step 4: Train the classifier
classifier = NaiveBayesClassifier.train(train_set)

# Step 5: Evaluate the classifier
accuracy_score = accuracy(classifier, test_set)
print("Accuracy:", accuracy_score)

# Step 6: Classify new reviews
new_reviews = [
    "This movie was fantastic!",
    "I didn't like the acting in this film.",
    "The plot was confusing, but the visuals were stunning."
]
for review in new_reviews:
    features = extract_features(review.split())
    sentiment = classifier.classify(features)
    print(review)
    print("Sentiment:", sentiment)
