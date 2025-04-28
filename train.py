import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 📂 Load the dataset
df = pd.read_csv(r"C:\train\train.csv")  # Ensure train.csv is in the same folder
df = df.fillna('')  # Fill missing values

# 📝 Combine 'author' & 'title' for better analysis
df["content"] = df["author"] + " " + df["title"]

# 🔹 Separate Features (X) and Labels (Y)
X = df["content"]
Y = df["label"]  # Assuming labels are 0 (real) and 1 (fake)

# ✨ Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# 🔀 Split Data into Train/Test Sets
X_train, X_test, Y_train, Y_test = train_test_split(X_vectorized, Y, test_size=0.2, stratify=Y, random_state=2)

# 🚀 Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=2)
model.fit(X_train, Y_train)

# 📊 Evaluate Accuracy
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# 💾 Save Model & Vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("🎉 Model & Vectorizer saved successfully!")
