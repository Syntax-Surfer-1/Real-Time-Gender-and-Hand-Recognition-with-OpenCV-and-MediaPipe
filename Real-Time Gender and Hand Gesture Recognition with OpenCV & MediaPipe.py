import cv2
import time
import mediapipe as mp
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# Set up TensorFlow to use GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and ready for use.")

# Initialize Mediapipe face detection and hands modules
def init_mediapipe():
    mp_face_detection = mp.solutions.face_detection
    mp_hands = mp.solutions.hands
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return face_detection, hands, mp_hands

# Build the gender prediction model
def build_gender_model():
    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the pre-trained gender model
def load_gender_model():
    model = build_gender_model()
    # Here you should load the model weights if you have a pre-trained model saved
    # For example: model.load_weights('path_to_pretrained_model_weights.h5')
    return model

# Detect faces using Mediapipe and predict gender using DeepFace
def detect_faces_and_gender(face_detection, img, model):
    results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            faces.append((x, y, w, h))
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the face region
            face_img = img[y:y+h, x:x+w]
            if face_img.size != 0:
                # Predict gender
                face_img_resized = cv2.resize(face_img, (224, 224))  # Ensure the face image is the correct size
                face_img_normalized = face_img_resized / 255.0  # Normalize the face image
                face_img_batch = np.expand_dims(face_img_normalized, axis=0)  # Add batch dimension
                gender_predictions = model.predict(face_img_batch)
                gender_label = 'Woman' if gender_predictions[0][0] > 0.5 else 'Man'
                gender_text = f"{gender_label}"

                # Display the gender prediction
                cv2.putText(img, gender_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    return faces

# Detect and draw hands in the given image
def detect_hands(rgb_img, img, hands, mp_hands, left_count, right_count):
    results = hands.process(rgb_img)
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            if hand_label == "Left":
                left_count[0] = draw_hand_landmarks(img, hand_landmarks, hand_handedness, mp_hands, left_count[0])
            else:
                right_count[0] = draw_hand_landmarks(img, hand_landmarks, hand_handedness, mp_hands, right_count[0])

# Draw hand landmarks and annotations
def draw_hand_landmarks(img, hand_landmarks, hand_handedness, mp_hands, finger_count):
    landmark_points = []
    h, w, _ = img.shape
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        landmark_points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    hand_label = hand_handedness.classification[0].label
    hand_text = "Left Hand" if hand_label == "Left" else "Right Hand"
    cv2.putText(img, hand_text, (landmark_points[0][0], landmark_points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        cv2.line(img, landmark_points[start_idx], landmark_points[end_idx], (0, 0, 255), 2)

    extended_fingers = count_extended_fingers(hand_landmarks, mp_hands)
    return extended_fingers

# Count the number of extended fingers
def count_extended_fingers(hand_landmarks, mp_hands):
    tips_ids = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    mcp_ids = [
        mp_hands.HandLandmark.THUMB_CMC,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]

    count = 0
    for tip_id, mcp_id in zip(tips_ids, mcp_ids):
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[mcp_id].y:
            count += 1

    return count

# Main function to run the video processing
def main():
    face_detection, hands, mp_hands = init_mediapipe()
    gender_model = load_gender_model()

    cap = cv2.VideoCapture(0)
    mirror_view = True
    start_time = time.time()
    frame_count = 0
    left_count = [0]
    right_count = [0]

    while True:
        ret, img = cap.read()
        if not ret:
            break

        if mirror_view:
            img = cv2.flip(img, 1)

        faces = detect_faces_and_gender(face_detection, img, gender_model)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detect_hands(rgb_img, img, hands, mp_hands, left_count, right_count)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(img, f'FPS: {fps:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        total_fingers = left_count[0] + right_count[0]
        cv2.putText(img, f'Left Fingers: {left_count[0]}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f'Right Fingers: {right_count[0]}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f'Total Fingers: {total_fingers}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Hello', img)

        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
