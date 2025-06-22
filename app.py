from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

model = tf.keras.models.load_model('model/exercise_model.h5')
scaler = joblib.load('model/scaler.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

routines = {
    'muscle_gain_normal': ["벤치프레스 4세트", "스쿼트 4세트", "데드리프트 3세트"],
    'muscle_gain_overweight': ["레그프레스 3세트", "체스트 프레스 3세트", "레그 익스텐션 3세트"],
    'weight_loss_obese': ["인터벌 러닝 20분", "버피 4세트", "점핑잭 3세트"],
    'weight_loss_normal': ["조깅 30분", "마운틴 클라이머 3세트", "사이클 20분"],
    'flexibility_all': ["요가 30분", "코브라 자세", "스트레칭 루틴"]
}

def get_bmi_category(weight, height):
    bmi = weight / ((height / 100) ** 2)
    if bmi >= 30:
        return 'obese'
    elif bmi >= 25:
        return 'overweight'
    else:
        return 'normal'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    age = int(request.form['age'])
    gender = request.form['gender']
    goal = request.form['goal']
    experience = request.form['experience']
    height = float(request.form['height'])
    weight = float(request.form['weight'])

    bmi_category = get_bmi_category(weight, height)

    input_df = pd.DataFrame([[age, gender, goal, experience]],
                            columns=['age', 'gender', 'goal', 'experience'])

    for col in ['gender', 'goal', 'experience']:
        input_df[col] = label_encoders[col].transform(input_df[col])

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    predicted_goal = np.argmax(prediction, axis=1)
    inv_goal = label_encoders['goal'].inverse_transform(predicted_goal)[0]

    if inv_goal == 'flexibility':
        routine_key = 'flexibility_all'
    else:
        routine_key = f"{inv_goal}_{bmi_category}"

    recommended_routine = routines.get(routine_key, ["추천 운동 루틴 없음"])

    # 세션처럼 값 전달 (파라미터를 URL에 포함)
    return render_template('result.html', 
                           age=age,
                           gender=gender,
                           goal=goal,
                           predicted_goal=inv_goal,
                           experience=experience,
                           bmi_category=bmi_category,
                           height=height,
                           weight=weight,
                           routine=recommended_routine)

@app.route('/feedback', methods=['POST'])
def feedback():
    user_feedback = {
        "age": request.form['age'],
        "gender": request.form['gender'],
        "height": request.form['height'],
        "weight": request.form['weight'],
        "goal": request.form['goal'],
        "predicted_goal": request.form['predicted_goal'],
        "experience": request.form['experience'],
        "bmi_category": request.form['bmi_category'],
        "feedback_text": request.form.get('feedback_text', ''),
        "satisfaction": request.form.get('satisfaction', 'none')
    }

    feedback_df = pd.DataFrame([user_feedback])
    if not os.path.exists("feedback.csv"):
        feedback_df.to_csv("feedback.csv", index=False, encoding='utf-8-sig')
    else:
        feedback_df.to_csv("feedback.csv", mode='a', header=False, index=False, encoding='utf-8-sig')

    return "<h3>피드백 감사합니다! 프로그램이 더 똑똑해지는 데 도움이 됩니다 :)</h3>"

if __name__ == '__main__':
    app.run(debug=True)
