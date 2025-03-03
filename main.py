from data_collection import capture_data
from train_model import trains_and_save_model
from use_model import use_trained_model

def main():
    print("Selecciona una opción:")
    print("1. Capturar datos y entrenar modelo")
    print("2. Usar modelo entrenado")
    choice = input("Opción: ")

    if choice == '1':
        capture_data()
        trains_and_save_model()
    elif choice == '2':
        use_trained_model()
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()