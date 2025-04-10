import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def trains_and_save_model():
    df = pd.read_csv('coords.csv')
    X = df.drop("Estado", axis=1)
    y = df['Estado']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Guardar el modelo entrenado
    with open('body_language.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Modelo entrenado y guardado como 'body_language.pkl")