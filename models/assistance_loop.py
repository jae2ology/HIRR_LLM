import os
import json
import numpy as np
import time

from perception_module import PerceptionModule
from activity_recognition import ActivityRecognizer
from baseline_HIRR import HIRRBase, UserAction

dataset_path = 'data/merged_json/homer_dataset_complete.json'
X_train_path = 'data/merged_json/train/X_train_raw.json'
X_test_path = 'data/merged_json/test/X_test_raw.json'

X = []
y = []
dataset = []

perception_module = PerceptionModule(False)
activity_model = ActivityRecognizer(X, y)
HIRR_base = HIRRBase(dataset)

# main loop
while True:
    current_objs = []

    # get rgbd data from the camera and detect object being picked up from YOLO model
    detected_labels = perception_module.get_detected_objects()
    if len(detected_labels) == 0:
        continue

    current_object = detected_labels[0]
    current_objs.append(current_object)

    print(f"User picked up {current_object}")

    # check if user is done with task already
    while True:
        perception_module_2 = PerceptionModule(False)
        detected_labels_2 = perception_module.get_detected_objects()
        if detected_labels_2[0] != current_object:
            # user is doing the task on their own. observe
            current_objs.append(detected_labels_2)
            current_object = detected_labels_2[0]
            print(f"User picked up {current_object}")
            time.sleep(2)
        else:
            break

    # predict what activity that was
    current_activity = activity_model.predict(current_objs)

    user_action = UserAction(activity=current_activity, objects_in_activity=current_objs, completed=False)

    # check if the activity is recognized:
    if HIRR_base.can_assist_activity(user_action):
        # then check if it has been completed
        if HIRR_base.can_assist_objects(user_action):
            # continue to object prediction
            next_object = HIRR_base.evaluate_next_step(user_action)
            if next_object == 'END':
                print("Nothing to assist with. Task over!")
                break
            # send to robot to assist
            break

        else:
            print("Cannot assist at the moment")
            continue

    else:
        print("Does not recognize activity")
        continue

    time.sleep(0.5)
