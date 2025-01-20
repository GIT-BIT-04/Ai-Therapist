from flask import Flask, request, jsonify, render_template
import pickle
from classify import voting_classifier
from flask_cors import CORS
import pandas as pd
import os
import datetime
from surprise import Dataset, Reader, SVD

app = Flask(__name__, static_folder="FrontEnd")
CORS(app)

# Load the hybrid recommender model
with open("hybrid_recommender_model.pkl", "rb") as hybrid_model_file:
    hybrid_model = pickle.load(hybrid_model_file)

# Function to get recommendations
def get_recommendations(user_input):
    predicted_emotion = voting_classifier.predict([user_input])[0]
    targeted_issue = predicted_emotion
    print(f"Detected Emotion: {predicted_emotion}")

    user_id = 1  # Hardcoded for now
    top_n = 5

    content_df = hybrid_model["content_df"]
    algo = hybrid_model["collaborative_filtering_model"]

    def get_content_based_recommendations():
        relevant_solutions = content_df[
            content_df['Targeted Issue'].str.contains(targeted_issue, case=False, na=False)
        ]
        return relevant_solutions.sort_values(by='Solution ID').head(top_n)['Solution Name'].tolist()

    def get_collaborative_filtering_recommendations():
        testset = algo.trainset.build_anti_testset()
        testset = filter(lambda x: x[0] == user_id, testset)
        predictions = algo.test(testset)
        predictions.sort(key=lambda x: x.est, reverse=True)
        recommendations = [prediction.iid for prediction in predictions[:top_n]]
        return recommendations

    content_recommendations = get_content_based_recommendations()
    collaborative_recommendations = get_collaborative_filtering_recommendations()

    hybrid_recommendations = list(set(content_recommendations + collaborative_recommendations))
    print(f"Recommendations: {hybrid_recommendations[:top_n]}")
    return predicted_emotion, hybrid_recommendations[:top_n]

# Save feedback
def save_feedback(user_id, solution_id, user_response, rating):
    feedback_file = "feedback.csv"
    if os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)
    else:
        feedback_df = pd.DataFrame(columns=["Interaction ID", "User ID", "Solution ID", "User Response", "Rating", "Timestamp"])

    interaction_id = len(feedback_df) + 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_feedback = pd.DataFrame([[interaction_id, user_id, solution_id, user_response, rating, timestamp]],
                                columns=["Interaction ID", "User ID", "Solution ID", "User Response", "Rating", "Timestamp"])
    feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    feedback_df.to_csv(feedback_file, index=False)
    print(f"Feedback for Solution ID {solution_id} saved with a rating of {rating}.")

# Retrain models
def retrain_models():
    feedback_file = "feedback.csv"
    if not os.path.exists(feedback_file):
        return "No feedback data available for retraining."

    feedback_df = pd.read_csv(feedback_file)
    global hybrid_model

    # Update collaborative filtering data
    data = hybrid_model["content_df"].copy()
    feedback_data = feedback_df[["User ID", "Solution ID", "Rating"]]
    data = pd.concat([data, feedback_data], ignore_index=True).drop_duplicates()

    reader = Reader(rating_scale=(1, 5))
    data_cf = Dataset.load_from_df(data[["User ID", "Solution ID", "Rating"]], reader)
    trainset = data_cf.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    # Save the updated model
    hybrid_model["collaborative_filtering_model"] = algo
    with open("hybrid_recommender_model.pkl", "wb") as hybrid_model_file:
        pickle.dump(hybrid_model, hybrid_model_file)
    return "Models retrained successfully."

@app.route("/process", methods=["POST"])
def process_message():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        if not user_message.strip():
            return jsonify({"response": "Please provide a valid input."}), 400
        
        detected_emotion, recommendations = get_recommendations(user_message)
        
        if recommendations:
            response_text = (
                f"We detected that you might be feeling '{detected_emotion}'.\n"
                f"Here are some therapies we suggest you to follow:\n"
                + "\n".join(f"- {rec}" for rec in recommendations)
            )
        else:
            response_text = (
                f"We detected that you might be feeling '{detected_emotion}', "
                "but we couldn't find any recommendations for your input."
            )
        return jsonify({"response": response_text, "emotion": detected_emotion, "recommendations": recommendations})
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        solution_id = data.get("solution_id")
        user_response = data.get("user_response")
        rating = data.get("rating")

      
        
        save_feedback(user_id, solution_id, user_response, rating)
        return jsonify({"response": "Feedback saved successfully."})
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        message = retrain_models()
        return jsonify({"response": message})
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=5500, debug=False)
