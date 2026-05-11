import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

mensajes = [
    "Gana dinero rápido",
    "Haz clic para reclamar premio",
    "Oferta limitada compra ahora",
    "Has ganado un iphone",
    "Hola profesor adjunto la tarea",
    "Nos vemos mañana",
    "Te mando el reporte",
    "Reunión a las 5 pm",
    "Recibe dinero gratis",
    "Tu cuenta fue seleccionada",
    "Promoción exclusiva",
]

categorias = [
    "spam",
    "spam",
    "spam",
    "spam",
    "no spam",
    "no spam",
    "no spam",
    "no spam"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(mensajes)

modelo = MLPClassifier(
    hidden_layer_sizes=(5,),
    max_iter=1000,
    random_state=1
)

modelo.fit(X, categorias)

print("Modelo entrenado correctamente.")

while True:

    mensaje = input("\nEscribe un mensaje: ")

    if mensaje.lower() == "salir":
        break

    mensaje_vectorizado = vectorizer.transform([mensaje])

    resultado = modelo.predict(mensaje_vectorizado)

    print("Clasificación:", resultado[0])