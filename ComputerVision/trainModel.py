from detect_track_pose import record_pose_data

labels = ["idle", "kick", "punch"]

# Test model
import xgboost as xgb
loaded_model = xgb.XGBClassifier()
loaded_model.load_model('Model/xgboost_model.json')

def predict_action(pose_vector):
    # returns the predicted action label
    prediction = loaded_model.predict(pose_vector.reshape(1, -1))[0]
    return labels[prediction]

#gets the keypoint vector out of the loop, to feed in a model in the future
for pose_vector in record_pose_data(mode_test=True, prediction_callback=predict_action):
    # feed pose_vector to your model in real time
    print(pose_vector)
    print("-------------------")