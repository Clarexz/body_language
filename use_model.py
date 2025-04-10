import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def use_trained_model():
    # Cargar el modelo entrenado
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Iniciar la cámara
    cap = cv2.VideoCapture(1)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolorizar la imagen a RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Dibujar landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            try:
                def extract_landmarks(landmarks, expected_count):
                    if landmarks:
                        return list(np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks]).flatten())
                    else:
                        # Si no se detectaron landmarks, rellena con ceros
                        return [0.0] * (expected_count * 4)

                # Extraer puntos de referencia
                pose_row = extract_landmarks(results.pose_landmarks.landmark if results.pose_landmarks else None, 33)
                face_row = extract_landmarks(results.face_landmarks.landmark if results.face_landmarks else None, 468)
                left_hand_row = extract_landmarks(
                    results.left_hand_landmarks.landmark if results.left_hand_landmarks else None, 21)
                right_hand_row = extract_landmarks(
                    results.right_hand_landmarks.landmark if results.right_hand_landmarks else None, 21)

                # Concatenar todo
                row = pose_row + face_row + left_hand_row + right_hand_row

                # Convertir a DataFrame con columnas del modelo
                X = pd.DataFrame([row], columns=model.feature_names_in_)

                # Hacer la predicción
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                confidence = body_language_prob[np.argmax(body_language_prob)]

                # Coordenadas de la oreja izquierda
                coords = tuple(np.multiply(
                    np.array((
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y
                    )), [640, 480]).astype(int))

                # Mostrar resultados en pantalla
                cv2.rectangle(image, (coords[0], coords[1] + 5),
                              (coords[0] + len(body_language_class) * 20, coords[1] - 30), (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

                # Status box
                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                cv2.putText(image, f"Estado: {body_language_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Probabilidad: {confidence:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error al hacer predicciones: {e}")

            # Mostrar la imagen en una ventana
            cv2.imshow('Predicciones', image)

            # Salir si se presiona la tecla 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()