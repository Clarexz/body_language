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
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Hacer predicciones
            try:
                # Extraer puntos de referencia de la pose
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extraer puntos de referencia de la cara
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                # Concatenar filas
                row = pose_row + face_row
                
                # Convertir a DataFrame con nombres de columnas
                X = pd.DataFrame([row], columns=model.feature_names_in_)  # Usar los nombres de características del modelo
                
                # Hacer la predicción
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
        
                # Obtener la probabilidad de la clase predicha
                confidence = body_language_prob[np.argmax(body_language_prob)]
                
                # Coordenadas de la oreja izquierda
                coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
            
                cv2.rectangle(image, 
                        (coords[0], coords[1]+5), 
                        (coords[0]+len(body_language_class)*20, coords[1]-30), 
                        (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
    
                # Mostrar resultados en la pantalla
                cv2.putText(image, f"Clase: {body_language_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Confianza: {confidence:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
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