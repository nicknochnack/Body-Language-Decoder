{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 0. Install and Import Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install mediapipe opencv-python pandas scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import mediapipe as mp # Import mediapipe\n",
    "import cv2 # Import opencv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mp_drawing = mp.solutions.drawing_utils # Drawing helpers\n",
    "mp_holistic = mp.solutions.holistic # Mediapipe Solutions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Make Some Detections"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "# Initiate holistic model\n",
    "with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:\n",
    "    \n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        \n",
    "        # Recolor Feed\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        image.flags.writeable = False        \n",
    "        \n",
    "        # Make Detections\n",
    "        results = holistic.process(image)\n",
    "        # print(results.face_landmarks)\n",
    "        \n",
    "        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks\n",
    "        \n",
    "        # Recolor image back to BGR for rendering\n",
    "        image.flags.writeable = True   \n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "        \n",
    "        # 1. Draw face landmarks\n",
    "        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),\n",
    "                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)\n",
    "                                 )\n",
    "        \n",
    "        # 2. Right hand\n",
    "        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "\n",
    "        # 3. Left Hand\n",
    "        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "\n",
    "        # 4. Pose Detections\n",
    "        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "                        \n",
    "        cv2.imshow('Raw Webcam Feed', image)\n",
    "\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "results.face_landmarks.landmark[0].visibility"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Capture Landmarks & Export to CSV\n",
    "<!--<img src=\"https://i.imgur.com/8bForKY.png\">-->\n",
    "<!--<img src=\"https://i.imgur.com/AzKNp7A.png\">-->"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import csv\n",
    "import os\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)\n",
    "num_coords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "landmarks = ['class']\n",
    "for val in range(1, num_coords+1):\n",
    "    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "landmarks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('coords.csv', mode='w', newline='') as f:\n",
    "    csv_writer = csv.writer(f, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)\n",
    "    csv_writer.writerow(landmarks)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class_name = \"Wakanda Forever\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "# Initiate holistic model\n",
    "with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:\n",
    "    \n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        \n",
    "        # Recolor Feed\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        image.flags.writeable = False        \n",
    "        \n",
    "        # Make Detections\n",
    "        results = holistic.process(image)\n",
    "        # print(results.face_landmarks)\n",
    "        \n",
    "        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks\n",
    "        \n",
    "        # Recolor image back to BGR for rendering\n",
    "        image.flags.writeable = True   \n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "        \n",
    "        # 1. Draw face landmarks\n",
    "        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),\n",
    "                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)\n",
    "                                 )\n",
    "        \n",
    "        # 2. Right hand\n",
    "        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "\n",
    "        # 3. Left Hand\n",
    "        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "\n",
    "        # 4. Pose Detections\n",
    "        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "        # Export coordinates\n",
    "        try:\n",
    "            # Extract Pose landmarks\n",
    "            pose = results.pose_landmarks.landmark\n",
    "            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())\n",
    "            \n",
    "            # Extract Face landmarks\n",
    "            face = results.face_landmarks.landmark\n",
    "            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())\n",
    "            \n",
    "            # Concate rows\n",
    "            row = pose_row+face_row\n",
    "            \n",
    "            # Append class name \n",
    "            row.insert(0, class_name)\n",
    "            \n",
    "            # Export to CSV\n",
    "            with open('coords.csv', mode='a', newline='') as f:\n",
    "                csv_writer = csv.writer(f, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)\n",
    "                csv_writer.writerow(row) \n",
    "            \n",
    "        except:\n",
    "            pass\n",
    "                        \n",
    "        cv2.imshow('Raw Webcam Feed', image)\n",
    "\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Train Custom Model Using Scikit Learn"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.1 Read in Collected Data and Process"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('coords.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[df['class']=='Sad']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop('class', axis=1) # features\n",
    "y = df['class'] # target value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.2 Train Machine Learning Classification Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import make_pipeline \n",
    "from sklearn.preprocessing import StandardScaler \n",
    "\n",
    "from sklearn.linear_model import LogisticRegression, RidgeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pipelines = {\n",
    "    'lr':make_pipeline(StandardScaler(), LogisticRegression()),\n",
    "    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),\n",
    "    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),\n",
    "    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fit_models = {}\n",
    "for algo, pipeline in pipelines.items():\n",
    "    model = pipeline.fit(X_train, y_train)\n",
    "    fit_models[algo] = model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fit_models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fit_models['rc'].predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.3 Evaluate and Serialize Model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score # Accuracy metrics \n",
    "import pickle "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for algo, model in fit_models.items():\n",
    "    yhat = model.predict(X_test)\n",
    "    print(algo, accuracy_score(y_test, yhat))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fit_models['rf'].predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('body_language.pkl', 'wb') as f:\n",
    "    pickle.dump(fit_models['rf'], f)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Make Detections with Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('body_language.pkl', 'rb') as f:\n",
    "    model = pickle.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cap = cv2.VideoCapture(0)\n",
    "# Initiate holistic model\n",
    "with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:\n",
    "    \n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        \n",
    "        # Recolor Feed\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        image.flags.writeable = False        \n",
    "        \n",
    "        # Make Detections\n",
    "        results = holistic.process(image)\n",
    "        # print(results.face_landmarks)\n",
    "        \n",
    "        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks\n",
    "        \n",
    "        # Recolor image back to BGR for rendering\n",
    "        image.flags.writeable = True   \n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "        \n",
    "        # 1. Draw face landmarks\n",
    "        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),\n",
    "                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)\n",
    "                                 )\n",
    "        \n",
    "        # 2. Right hand\n",
    "        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "\n",
    "        # 3. Left Hand\n",
    "        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "\n",
    "        # 4. Pose Detections\n",
    "        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, \n",
    "                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),\n",
    "                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)\n",
    "                                 )\n",
    "        # Export coordinates\n",
    "        try:\n",
    "            # Extract Pose landmarks\n",
    "            pose = results.pose_landmarks.landmark\n",
    "            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())\n",
    "            \n",
    "            # Extract Face landmarks\n",
    "            face = results.face_landmarks.landmark\n",
    "            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())\n",
    "            \n",
    "            # Concate rows\n",
    "            row = pose_row+face_row\n",
    "            \n",
    "#             # Append class name \n",
    "#             row.insert(0, class_name)\n",
    "            \n",
    "#             # Export to CSV\n",
    "#             with open('coords.csv', mode='a', newline='') as f:\n",
    "#                 csv_writer = csv.writer(f, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)\n",
    "#                 csv_writer.writerow(row) \n",
    "\n",
    "            # Make Detections\n",
    "            X = pd.DataFrame([row])\n",
    "            body_language_class = model.predict(X)[0]\n",
    "            body_language_prob = model.predict_proba(X)[0]\n",
    "            print(body_language_class, body_language_prob)\n",
    "            \n",
    "            # Grab ear coords\n",
    "            coords = tuple(np.multiply(\n",
    "                            np.array(\n",
    "                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, \n",
    "                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))\n",
    "                        , [640,480]).astype(int))\n",
    "            \n",
    "            cv2.rectangle(image, \n",
    "                          (coords[0], coords[1]+5), \n",
    "                          (coords[0]+len(body_language_class)*20, coords[1]-30), \n",
    "                          (245, 117, 16), -1)\n",
    "            cv2.putText(image, body_language_class, coords, \n",
    "                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)\n",
    "            \n",
    "            # Get status box\n",
    "            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)\n",
    "            \n",
    "            # Display Class\n",
    "            cv2.putText(image, 'CLASS'\n",
    "                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)\n",
    "            cv2.putText(image, body_language_class.split(' ')[0]\n",
    "                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)\n",
    "            \n",
    "            # Display Probability\n",
    "            cv2.putText(image, 'PROB'\n",
    "                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)\n",
    "            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))\n",
    "                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)\n",
    "            \n",
    "        except:\n",
    "            pass\n",
    "                        \n",
    "        cv2.imshow('Raw Webcam Feed', image)\n",
    "\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "tuple(np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, \n",
    "results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), [640,480]).astype(int))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "mediapipe",
   "language": "python",
   "name": "mediapipe"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
