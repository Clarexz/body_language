import firebase_admin
from firebase_admin import credentials, db
import time
import threading
import atexit

# Configuration
UPDATE_INTERVAL = 0.5  # seconds
SERVICE_ACCOUNT_FILE = "serviceAccountKey.json"
DATABASE_URL = "https://emociones-8d92c-default-rtdb.firebaseio.com/"

# State management
firebase_app = None
update_thread = None
should_run = False  # Start as False, set to True when initialized
current_state = ""
last_update = 0
data_lock = threading.Lock()


def initialize_firebase():
    global firebase_app, update_thread, should_run

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
            firebase_app = firebase_admin.initialize_app(cred, {
                'databaseURL': DATABASE_URL,
                'databaseAuthVariableOverride': {'uid': 'server-admin', 'admin': True}
            })

        # Activa el thread de updates
        should_run = True
        update_thread = threading.Thread(target=update_worker, daemon=True)
        update_thread.start()

        return True
    except Exception as e:
        print(f"Firebase init error: {e}")
        return False


def update_worker():
    global last_update

    while should_run:
        if time.time() - last_update >= UPDATE_INTERVAL:
            try:
                print(f"Intentando actualizar Firebase con: {current_state}")  # ← Nuevo
                ref = db.reference('/Emociones')
                ref.update({
                    'encendido': True,
                    'estadoActual': current_state
                })
                print("¡Actualización exitosa!")  # ← Nuevo
                last_update = time.time()
            except Exception as e:
                print(f"Update failed: {str(e)}")
        time.sleep(0.1)


def set_emotion_state(state):
    global current_state
    with data_lock:
        current_state = state

def cleanup():
    global should_run

    should_run = False  # Detiene el thread de updates

    # Espera brevemente si el thread está activo
    if update_thread and isinstance(update_thread, threading.Thread) and update_thread.is_alive():
        update_thread.join(timeout=0.3)

    # Intenta el cleanup solo si Firebase está inicializado
    if firebase_app:
        try:
            # Crea una conexión nueva y rápida (sin reusar la existente)
            cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
            temp_app = firebase_admin.initialize_app(cred, {
                'databaseURL': DATABASE_URL,
                'databaseAuthVariableOverride': {'uid': 'server-admin'}
            }, name='temp_cleanup_app')

            # Usa set() con timeout corto
            ref = db.reference('/Emociones', app=temp_app)
            ref.set({'encendido': False, 'estadoActual': ""}, timeout=2.0)
            # Elimina la app temporal
            firebase_admin.delete_app(temp_app)
        except Exception:
            pass  # Ignora cualquier error durante el cleanup

# Register cleanup handler
atexit.register(cleanup)