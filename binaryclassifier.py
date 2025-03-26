import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the datasets
dad_df = pd.read_csv("dad_jokes.csv") # dad joke dataset
non_dad_df = pd.read_csv("non_dadjoke.csv") # non-dad joke dataset

non_dad_df.columns = non_dad_df.columns.str.lower()

dad_df["label"] = 1  # 1 is dad jokes
non_dad_df["label"] = 0  # 0 is non-dad jokes

# Combine the two datasets
data = pd.concat([dad_df, non_dad_df], ignore_index=True)

print("Data:")
print(data.head())

# Split train and test data
train, test = train_test_split(data, test_size=0.1, random_state=9)

# Print the number of samples in train and test sets
print("Rows in train set:", train.shape[0])
print("Rows in the test set:", test.shape[0])

vec_ques = TfidfVectorizer()
vec_ans = TfidfVectorizer()

X_train_ques = vec_ques.fit_transform(train["question"])
X_train_ans = vec_ans.fit_transform(train["answer"])

X_test_ques = vec_ques.transform(test["question"])
X_test_ans = vec_ans.transform(test["answer"])

# Hstack used to combine vectorized question and answer 
X_train_combined = hstack([X_train_ques, X_train_ans])
X_test_combined = hstack([X_test_ques, X_test_ans])

y_train = train["label"]
y_test = test["label"]

print("Distribution in the train set:", y_train.value_counts())

# Train
model = LogisticRegression()
model.fit(X_train_combined, y_train)

y_pred = model.predict(X_test_combined)

# Evaluation
print(classification_report(y_test, y_pred))
