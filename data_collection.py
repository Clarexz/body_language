import cv2
import mediapipe as mp
import csv
import numpy as np
import os  # Importar el módulo os para verificar si el archivo existe

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def capture_data():
    class_name = input("Ingresa el nombre de la clase (por ejemplo, 'Feliz', 'Triste'): ")
    cap = cv2.VideoCapture(1)  # Iniciar la cámara
    
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return
    
    # Iniciar el modelo Holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()  # Leer un frame de la cámara
            
            if not ret:
                print("Error: No se puede leer el frame")
                break

            # Recolorizar la imagen a RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
                
            # Hacer detecciones
            results = holistic.process(image)
                
            # Recolorizar la imagen de nuevo a BGR
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

            # Mostrar el frame en una ventana
            cv2.imshow("Cámara", image)

            # Salir si se presiona la tecla "q"
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            # Verificar si el archivo CSV existe
            if not os.path.exists('coords.csv'):
                # Crear el archivo CSV con los encabezados
                num_coords = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)
                landmarks = ['class']
                for val in range(1, num_coords + 1):
                    landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
                
                with open('coords.csv', mode='w', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(landmarks)
            
            # Guardar landmarks en CSV
            try:
                # Extraer puntos de referencia de la pose
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extraer puntos de referencia de la cara
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                # Concatenar filas
                row = pose_row + face_row
                
                # Agregar el nombre de la clase al principio del row
                row.insert(0, class_name)
                
                # Escribir en el archivo CSV
                with open("coords.csv", mode="a", newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
            except:
                print("Error al guardar los datos")

        cap.release()  # Liberar la cámara
        cv2.destroyAllWindows()  # Cerrar todas las ventanas