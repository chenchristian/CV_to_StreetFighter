from detect_track_pose import record_pose_data

#gets the keypoint vector out of the loop, to feed in a model in the future
for pose_vector in record_pose_data(mode_test=True):
    # feed pose_vector to your model in real time
        print(pose_vector)
        print("______________________")
