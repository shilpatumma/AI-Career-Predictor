import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv('../data/career_data.csv')

# -----------------------------
# 2. PREPROCESSING
# -----------------------------

# Convert text skills → numeric
vectorizer = CountVectorizer()
X_skills = vectorizer.fit_transform(df['skills'])

# Encode education
edu_encoder = LabelEncoder()
df['education_encoded'] = edu_encoder.fit_transform(df['education'])

# Combine features
import numpy as np
X = np.hstack((X_skills.toarray(),
               df[['experience', 'education_encoded']].values))

# -----------------------------
# 3. TARGETS
# -----------------------------

# Job role (classification)
role_encoder = LabelEncoder()
y_role = role_encoder.fit_transform(df['job_role'])

# Salary (regression)
y_salary = df['salary']

# -----------------------------
# 4. TRAIN MODELS
# -----------------------------

# 🎯 Classification Model
clf = RandomForestClassifier()
clf.fit(X, y_role)

# 💰 Regression Model
reg = LinearRegression()
reg.fit(X, y_salary)

# 🔵 Clustering Model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# -----------------------------
# 5. SAVE MODELS
# -----------------------------

with open('../models/classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('../models/regressor.pkl', 'wb') as f:
    pickle.dump(reg, f)

with open('../models/kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

with open('../models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('../models/edu_encoder.pkl', 'wb') as f:
    pickle.dump(edu_encoder, f)

with open('../models/role_encoder.pkl', 'wb') as f:
    pickle.dump(role_encoder, f)

print("✅ Training complete and models saved!")