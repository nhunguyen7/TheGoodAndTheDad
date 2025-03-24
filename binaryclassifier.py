import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the datasets
dad_df = pd.read_csv("[DADJOKE]") # dad joke dataset
non_dad_df = pd.read_csv("[NORMALJOKE]") # non-dad joke dataset

dad_df["label"] = 1  # 1 is dad jokes
non_dad_df["label"] = 0  # 0 is non-dad jokes

# Combine the two datasets
data = pd.concat([dad_df, non_dad_df], ignore_index=True)

# Split train and test data
test_size = int(0.1 * len(data))
train, test  = train_test_split(data, test_size=test_size, random_state=9)

vec_ques = TfidfVectorizer()
vec_ans = TfidfVectorizer()

X_train_ques = vec_ques.fit_transform(data["question"])
X_train_ans = vec_ans.fit_transform(data["answer"])

X_test_ques = vectorizer_ques.transform(test["question"])
X_test_ans = vectorizer_ans.transform(test["answer"])

# Hstack used to combine vectorized question and answer 
X_train_combined = hstack([X_train_ques, X_train_ans])
X_test_combined = hstack([X_test_ques, X_test_ans])

y_train = train["label"]
y_test = test["label"]

# Train
model = LogisticRegression()
model.fit(X_train_combined, y_train)

y_pred = model.predict(X_test_combined)

# Evaluation
print(classification_report(y_test, y_pred))