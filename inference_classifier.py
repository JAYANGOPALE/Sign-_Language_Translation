import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load trained model
model_dict = pickle.load(open('rf_model.pickle', 'rb'))
model = model_dict['model']

# Label map (update if you have a bigger set)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Adjust to your label list
labels_dict = {i: label for i, label in enumerate(labels)}

# Webcam
cap = cv2.VideoCapture(0)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Sentence builder
sentence = ""
predicted_character = ""

while True:
    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            if x_ and y_:
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                x1 = int(min(x_) * W) - 20
                y1 = int(min(y_) * H) - 20
                x2 = int(max(x_) * W) + 20
                y2 = int(max(y_) * H) + 20

                if len(data_aux) == 42:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Draw box and prediction
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # Display sentence above frame
    cv2.rectangle(frame, (0, 0), (W, 40), (255, 255, 255), -1)
    cv2.putText(frame, "Sentence: " + sentence, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('Hand Sign Prediction', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):         # Quit
        break
    elif key == 13:             # Enter key → Confirm letter
        if predicted_character:
            sentence += predicted_character
    elif key == 8:              # Backspace key → Delete last character
        sentence = sentence[:-1]
    elif key == 32:             # Space key
        sentence += ' '

cap.release()
cv2.destroyAllWindows()
