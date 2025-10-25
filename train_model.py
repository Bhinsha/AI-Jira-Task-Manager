import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# -- TRAINING part (you can run this once and then comment it out / separate)  

df = pd.read_csv('C:/Users/bhinsha/OneDrive/Desktop/aitask/jira_dataset.csv')

    # assume columns: “description”, “type”, “priority”, “assignee”
df = df.dropna(subset=['clean_summary', 'issue_type', 'priority', 'task_assignee'])
    
le_type = LabelEncoder()
le_priority = LabelEncoder()
le_assignee = LabelEncoder()
df['type_enc'] = le_type.fit_transform(df['issue_type'])
df['priority_enc'] = le_priority.fit_transform(df['priority'])
df['assignee_enc'] = le_assignee.fit_transform(df['task_assignee'])
    
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df['clean_summary'])
    
y = np.vstack([
        df['type_enc'],
        df['priority_enc'],
        df['assignee_enc']
    ]).T
    
X_train, X_test = X, X  # for simplicity; you should split in real code
y_train, y_test = y, y
    
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train, y_train)
    
    # save artifacts
joblib.dump(tfidf, 'tfidf_model.pkl')
joblib.dump(clf, 'task_model.pkl')
joblib.dump(le_type, 'le_type.pkl')
joblib.dump(le_priority, 'le_priority.pkl')
joblib.dump(le_assignee, 'le_assignee.pkl')
print("Model training complete.")

