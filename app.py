from flask import Flask, render_template, request, jsonify
import joblib

# -- LOAD artifacts
tfidf = joblib.load('tfidf_model.pkl')
clf   = joblib.load('task_model.pkl')
le_type     = joblib.load('le_type.pkl')
le_priority = joblib.load('le_priority.pkl')
le_assignee = joblib.load('le_assignee.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    description = request.form.get('description', '')
    if not description:
        return render_template('index.html', error="Please enter a task description.")
    
    X_new = tfidf.transform([description])
    preds = clf.predict(X_new)[0]
    type_pred     = le_type.inverse_transform([preds[0]])[0]
    priority_pred = le_priority.inverse_transform([preds[1]])[0]
    assignee_pred = le_assignee.inverse_transform([preds[2]])[0]
    
    return render_template('index.html',
                           description=description,
                           predicted_type=type_pred,
                           predicted_priority=priority_pred,
                           predicted_assignee=assignee_pred)

if __name__ == '__main__':
    app.run(debug=True)
