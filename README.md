## Gesture-Nauts - Invokes the spirit of exploration and adventure in the realm of gesture-based technology

### Table of Contents

1. [Description](#description)

2. [Demo](#demo)

3. [Usage](#usage)

4. [Contributors](#contributors)

5. [Division of Labor](#division-of-labor)

6. [Related Work](#related-work)

7. [Reflections](#reflections)

### Description

The idea of our project is to implement a 3D gesture recognition system that provides corresponding animations upon recognition.

The current backdrop is characterized by the frequent use of smart devices to simplify our lives. Various new technologies leveraging AI emphasize interaction with humans to unlock new functionalities. Gesture recognition, historically a fundamental cognitive function in humans, is now being automated through the rise of AI technologies. This advancement in computer technology is gradually implemented in various smart devices, such as smartphones, smart home systems, and virtual reality (VR) setups. This integration enables more intuitive and efficient ways for users to interact with technology, allowing for gesture-based commands that can control everything from music playback to complex navigation within virtual environments. As a result, gesture recognition is not only redefining the boundaries of human-computer interaction but is also paving the way for more personalized and seamless digital experiences.

### DataSet

[HaGRID - HAnd Gesture Recognition Image Dataset](https://www.kaggle.com/datasets/kapitanov/hagrid)

In our project, we use the subsample of the above dataset as the whole dataset is too big and slow for us to train. And the subsample, the feature generated from the mp_hands model, the keypoint classifier, point history classifier checkpoint can be found [here](https://drive.google.com/drive/folders/1EXHr-K1pcXEE_w2RdjumaqSaftNocv1W?usp=drive_link).

### Demo

Todo: add demo images

| Index | class           |
| ----- | --------------- |
| 0     | call            |
| 1     | dislike         |
| 2     | fist            |
| 3     | four            |
| 4     | like            |
| 5     | mute            |
| 6     | ok              |
| 7     | one             |
| 8     | stop_inverted   |
| 9     | rock            |
| 10    | peace_inverted  |
| 11    | stop            |
| 12    | palm            |
| 13    | peace           |
| 14    | three           |
| 15    | three2          |
| 16    | two_up          |
| 17    | two_up_inverted |

### Usage

#### Run locally

```
python app.py
```

--device
Specifying the camera device number (Default：0)
--width
Width at the time of camera capture (Default：960)
--height
Height at the time of camera capture (Default：540)
--use_static_image_mode
Whether to use static_image_mode option for MediaPipe inference (Default：Unspecified)
--min_detection_confidence
Detection confidence threshold (Default：0.5)
--min_tracking_confidence
Tracking confidence threshold (Default：0.5)

#### Deployed page

Todo:
We are working on this.

#### Package Requirements

- mediapipe 0.8.1

- OpenCV 3.4.2 or Later

- Tensorflow 2.3.0 or Later

- tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)

- scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix)

- matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)

We also provide our `requirement.txt` file. So all you need to do is:

```
pip install -r requirements.txt
```

There might be some differences between Mac and Windows for `TensorFlow` packages, so when there is a conflict, you have to resolve it manually.

### Contributor

- [@Chen Wei (cwei24)](https://github.com/MRSA-J)

- [@Kangyu Zhu (kzhu37)](https://github.com/)

- [@Zhongzheng Xu (zxu169)](https://github.com/lebretou)

### Division of labor

We plan on working equally across X aspects of the project:

1. Preprocess the data: Together

2. Design the model: Together

3. Model Architecture: Chen Wei

- Feature Generator (Use mediapipe's hands pretrained model)
- Keypoint classifier
- Point history classifier

4. Evaluation and and Visualization: Kangyu Zhu
5. Fine-tune the model, change the structure and parameters: Together
6. Model Training: Chen Wei
7. Deployment of the model: Zhongzheng Xu
8. Write the report and make the poster: Together

### Related Work

- [hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe/blob/main/README.md)

- [MediaPipe](https://developers.google.com/mediapipe)

### Reflections

Our project ultimately turned out to be ok and our model works as expected. It can generate captions that are acceptable and coherent although not being perfect. <br>
If we have more time, we could implement ...
