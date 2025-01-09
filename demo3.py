import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import datetime
import os
import pickle
from classify import predicted_emotion
from classify import voting_classifier

# 1. Data Loading and Initialization
data = pd.read_csv("solution.csv")
user_ids = [random.randint(1, 10) for _ in range(len(data))]
data['User ID'] = user_ids
data['Rating'] = data['Effectiveness Rating']

content_df = data[['Solution ID', 'Solution Name', 'Type', 'Description', 'Targeted Issue']].copy()
content_df['Content'] = content_df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

tfidf_vectorizer = TfidfVectorizer()
content_matrix = tfidf_vectorizer.fit_transform(content_df['Content'])

reader = Reader(rating_scale=(1, 5))
data_cf = Dataset.load_from_df(data[['User ID', 'Solution ID', 'Rating']], reader)
algo = SVD()
trainset = data_cf.build_full_trainset()
algo.fit(trainset)

# Save the trained collaborative filtering model, TF-IDF vectorizer, and the entire hybrid model
with open("collaborative_filtering_model.pkl", "wb") as cf_model_file:
    pickle.dump(algo, cf_model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

with open("hybrid_recommender_model.pkl", "wb") as hybrid_model_file:
    pickle.dump({
        "collaborative_filtering_model": algo,
        "tfidf_vectorizer": tfidf_vectorizer,
        "content_df": content_df
    }, hybrid_model_file)

# 2. Recommendation Engine Functions
def get_content_based_recommendations(targeted_issue, top_n):
    relevant_solutions = content_df[content_df['Targeted Issue'].str.contains(targeted_issue, case=False, na=False)]
    if relevant_solutions.empty:
        print(f"No solutions found for the targeted issue: {targeted_issue}")
        return []
    recommendations = relevant_solutions.sort_values(by='Solution ID').head(top_n)['Solution ID'].tolist()
    return recommendations

def get_collaborative_filtering_recommendations(user_id, top_n):
    testset = trainset.build_anti_testset()
    testset = filter(lambda x: x[0] == user_id, testset)
    predictions = algo.test(testset)
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommendations = [prediction.iid for prediction in predictions[:top_n]]
    return recommendations

def get_hybrid_recommendations(user_id, targeted_issue, top_n):
    content_based_recommendations = get_content_based_recommendations(targeted_issue, top_n)
    print(f"Content-based recommendations for Targeted Issue '{targeted_issue}': {content_based_recommendations}")

    collaborative_filtering_recommendations = get_collaborative_filtering_recommendations(user_id, top_n)
    print(f"Collaborative filtering recommendations for User ID {user_id}: {collaborative_filtering_recommendations}")

    hybrid_recommendations = list(set(content_based_recommendations + collaborative_filtering_recommendations))
    valid_recommendations = [rec for rec in hybrid_recommendations if rec in content_df['Solution ID'].values]
    print(f"Hybrid recommendations after filtering: {valid_recommendations}")
    return valid_recommendations[:top_n]

def update_dataset_with_feedback():
    feedback_file = "feedback.csv"
    if os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)
        global data
        new_ratings = feedback_df[['User ID', 'Solution ID', 'Rating']]
        data = pd.concat([data, new_ratings], ignore_index=True).drop_duplicates()
        print("Dataset updated with feedback.")
    else:
        print("No feedback data found to update the dataset.")

def retrain_collaborative_filtering():
    global algo, trainset
    reader = Reader(rating_scale=(1, 5))
    updated_cf_data = Dataset.load_from_df(data[['User ID', 'Solution ID', 'Rating']], reader)
    trainset = updated_cf_data.build_full_trainset()
    algo.fit(trainset)
    print("Collaborative filtering model retrained with updated data.")

    # Save the updated collaborative filtering model
    with open("collaborative_filtering_model.pkl", "wb") as cf_model_file:
        pickle.dump(algo, cf_model_file)

def refresh_content_based_data():
    global content_df, content_matrix
    content_df = data[['Solution ID', 'Solution Name', 'Type', 'Description', 'Targeted Issue']].copy()
    content_df['Content'] = content_df.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    content_matrix = tfidf_vectorizer.fit_transform(content_df['Content'])
    print("Content-based data and TF-IDF matrix refreshed.")

    # Save the updated TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
        pickle.dump(tfidf_vectorizer, vectorizer_file)

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

def is_first_user():
    feedback_file = "feedback.csv"
    return not os.path.exists(feedback_file) or len(pd.read_csv(feedback_file)) == 0

# 5. Main Workflow for Subsequent Users
if __name__ == "__main__":
    user_id = 2  # Example User ID, this would be dynamically determined in a real application
    text = "i want to die"
    predicted_emotion = voting_classifier.predict([text])[0]  # Predict emotion from NLP model
    targeted_issue = predicted_emotion  # Use the predicted emotion from NLP as the targeted issue
    top_n = 5  # Number of recommendations to generate

    print("----------------------------------------------------------------------------------------------")
    print("-------------------- EMOTION --------------------------")
    print("===================", predicted_emotion, "========================")
    print(f"Subsequent user detected with emotion: {targeted_issue}. Updating dataset and retraining models...")

    # Update dataset with feedback (assuming the feedback is gathered elsewhere)
    update_dataset_with_feedback()

    # Retrain the collaborative filtering model
    retrain_collaborative_filtering()

    # Refresh content-based recommendation data
    refresh_content_based_data()

    # Get hybrid recommendations based on the emotion predicted and the targeted issue
    recommendations = get_hybrid_recommendations(user_id, targeted_issue, top_n)

    # Ensure Solution Name and ID mapping is correct
    recommendation_data = content_df[content_df['Solution ID'].isin(recommendations)][['Solution ID', 'Solution Name']]
    solution_names = recommendation_data['Solution Name'].tolist()
    solution_ids = recommendation_data['Solution ID'].tolist()

    # Display the valid recommendations to the user
    if solution_names:
        print(f"Updated recommendations for User {user_id}: {solution_names}")
    else:
        print("No valid recommendations found.")

    # Collect feedback from the user with predefined options
    if solution_names:
        solution_name = solution_names[0]  # Pick the first one as an example or allow user to choose.
        print("--------------------------------------- Recommendations -------------------------------------------")
        print(f"Recommendation: {solution_name} (Solution ID: {solution_ids[0]})")
        print("----------------------------------------------//---------------------------------------------------------")
        # Predefined options for feedback
        feedback_options = ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"]
        print("Please provide feedback for the solution:")
        for i, option in enumerate(feedback_options, 1):
            print(f"{i}. {option}")
        print("------------------------------------//--------------------------------------")
        # User selects feedback option
        feedback_choice = int(input("Enter the number corresponding to your feedback: "))
        
        # Map the feedback choice to a text value
        user_response = feedback_options[feedback_choice - 1]
        
        # Ask for rating (1-5)
        rating = int(input("Please provide a rating between 1 and 5: "))
        
        # Find the Solution ID corresponding to the selected Solution Name
        solution_id = solution_ids[0]  # Use the ID of the first recommended solution

        # Save feedback with the correct Solution ID
        save_feedback(user_id, solution_id, user_response, rating)
