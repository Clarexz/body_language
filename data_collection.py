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
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            # Mostrar el frame en una ventana
            cv2.imshow("Camara", image)

            # Salir si se presiona la tecla "q"
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # Verificar si el archivo CSV existe
            if not os.path.exists('coords.csv'):
                # Inicializamos el conteo total de landmarks
                num_coords = 0
                landmarks = ['Estado']

                # Pose
                if results.pose_landmarks:
                    num_coords += len(results.pose_landmarks.landmark)
                else:
                    num_coords += 33  # valor fijo para pose

                # Face
                if results.face_landmarks:
                    num_coords += len(results.face_landmarks.landmark)
                else:
                    num_coords += 468  # valor fijo para face

                # Manos (siempre se añaden aunque no estén presentes)
                num_coords += 21 * 2  # 21 por cada mano

                for val in range(1, num_coords + 1):
                    landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']

                with open('coords.csv', mode='w', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(landmarks)

            # Guardar landmarks en CSV
            try:
                row = []

                # Pose
                if results.pose_landmarks:
                    pose = results.pose_landmarks.landmark
                    row += list(np.array([[l.x, l.y, l.z, l.visibility] for l in pose]).flatten())
                else:
                    row += [0.0] * 33 * 4  # 33 landmarks * 4 valores

                # Face
                if results.face_landmarks:
                    face = results.face_landmarks.landmark
                    row += list(np.array([[l.x, l.y, l.z, l.visibility] for l in face]).flatten())
                else:
                    row += [0.0] * 468 * 4

                # Mano izquierda
                if results.left_hand_landmarks:
                    left_hand = results.left_hand_landmarks.landmark
                    row += list(np.array([[l.x, l.y, l.z, l.visibility] for l in left_hand]).flatten())
                else:
                    row += [0.0] * 21 * 4

                # Mano derecha
                if results.right_hand_landmarks:
                    right_hand = results.right_hand_landmarks.landmark
                    row += list(np.array([[l.x, l.y, l.z, l.visibility] for l in right_hand]).flatten())
                else:
                    row += [0.0] * 21 * 4

                # Clase
                row.insert(0, class_name)

                with open("coords.csv", mode="a", newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

            except Exception as e:
                print(f"Error al guardar los datos: {e}")

        cap.release()  # Liberar la cámara
        cv2.destroyAllWindows()  # Cerrar todas las ventanas