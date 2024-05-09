import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model.fine_tune_keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model import PointHistoryClassifier
from app import calc_bounding_rect, calc_landmark_list, pre_process_landmark, pre_process_point_history, draw_bounding_rect, draw_landmarks, draw_info_text, draw_point_history

def load_keypoint_classifier_labels():
    """
    Load labels for the keypoint classifier from a CSV file.
    
    Returns:
        list: A list of labels from the keypoint classifier CSV file.
    """
    with open('model/fine_tune_keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in reader]
    return keypoint_classifier_labels

def load_point_history_classifier_labels():
    """
    Load labels for the point history classifier from a CSV file.
    
    Returns:
        list: A list of labels from the point history classifier CSV file.
    """
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in reader]
    return point_history_classifier_labels

# Initialize classifiers and other necessary components
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

# Load labels (assuming these functions or data are available globally)
keypoint_classifier_labels = load_keypoint_classifier_labels()
point_history_classifier_labels = load_point_history_classifier_labels()

# Initialize MediaPipe hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)



def process_frame(frame):
    """
    Process a single frame for hand gesture recognition.
    
    Args:
        frame (numpy.ndarray): The current video frame.
    
    Returns:
        numpy.ndarray: The processed frame with annotations.
    """
    # Mirror the frame for better user interaction
    # frame = cv.flip(frame, 1)
    # Convert color to RGB (as required by MediaPipe)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True
    output_image = frame.copy()
    label = ""

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Calculate bounding box
            brect = calc_bounding_rect(output_image, hand_landmarks)
            # Calculate landmarks
            landmark_list = calc_landmark_list(output_image, hand_landmarks)
            # Pre-process landmarks
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            pre_processed_point_history_list = pre_process_point_history(output_image, point_history)

            # Classify gesture based on keypoints
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == 2:  # Assume 2 is pointing gesture
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])


            # Classify gesture based on point history
            finger_gesture_id = 0
            point_history_len = len(pre_processed_point_history_list)
            if point_history_len == (history_length * 2):
                finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

            # Update gesture history
            finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(finger_gesture_history).most_common()

            # Drawing
            output_image = draw_bounding_rect(True, output_image, brect)
            output_image = draw_landmarks(output_image, landmark_list)

            label = point_history_classifier_labels[most_common_fg_id[0][0]]
    else:
        point_history.append([0, 0])

    output_image = draw_point_history(output_image, point_history)
    return output_image, label



