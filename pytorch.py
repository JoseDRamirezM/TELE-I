# PREREQUISITOS

import torch  # cargar el core de la librería
from torch import nn  # cargar modelo de redes neuronales
from torch.utils.data import DataLoader  # Módulo para cargar datos
from torchvision import datasets  # Módulo para cargar datos
from torchvision.transforms import ToTensor  # Módulo para transformar datos
import matplotlib.pyplot as plt  # Módulo para visualizar el Dataset

# Cargar datos de entrenamiento de pytorch
datos_entrenamiento = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Cargar datos de prueba de pytorch
datos_prueba = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Estas son las clases en las
# que se clasifican las imágenes
clases_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# VISUALIZAR DATASET
# Se utiliza matplotlib para visualizar las imágenes del Dataset
figura = plt.figure(figsize=(10, 10))  # Definir el tamaño del canvas
cols, filas = 8, 8  # Filas y columnas
for i in range(1, cols * filas + 1):  # Llenar el canvas con las imagenes
    # Luego tomamos imagenes aleatorias
    muestra_id = torch.randint(len(datos_entrenamiento), size=(1,)).item()
    img, label = datos_entrenamiento[muestra_id]
    figura.add_subplot(filas, cols, i)
    plt.title(clases_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# Primer hiperparámetro
# Imagenes utilizadas en una iteración
# Iteración : numero de veces que se actualizan parámetros del modelo
tam_lote = 64

# Objeto que almacena los datos de entrenamiento
entrenamiento_dataloader = DataLoader(datos_entrenamiento, batch_size=tam_lote)
# Dataloader para datos de prueba
prueba_dataloader = DataLoader(datos_prueba, batch_size=tam_lote)


# Definir un dispositivo CPU o GPU

# Se elige un dispositivo de procesamiento
# GPU NVIDIA o CPU
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando {dispositivo}")


# DEFINIR EL MODELO

class NeuralNetwork(nn.Module):  # clase del modelo
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # capa para normalizar imagenes a pixeles
        # La capa secuencial inserta otras capas en el mismo orden
        self.linear_relu_stack = nn.Sequential(
            # Combinación de funciones de activación
            # que crean asociaciones complejas entre
            # las neuronas

            # Transformaciones lineales
            nn.Linear(28*28, 512),
            # Transformaciones no lineales
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    # Esta funcion se llama cuando se le dan
    # entradas al modelo

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Se pasa el modelo al dispositivo de procesamiento
modelo = NeuralNetwork().to(dispositivo)
print(modelo)

# OPTIMIZAR LOS PARÁMETROS

# Función de pérdida para calcular el error
# Calula la distancia entre el resultado obtenido
# Y el esperado
fn_perdida = nn.CrossEntropyLoss()
# Algoritmo de optimización de
# gradiente descendente y segundo hiperparámetro
# Hiperparametro "learning rate"
# Define que tanto modificar los parametros
# En este caso los pesos (transformaciones) de cada neurona
optimizador = torch.optim.SGD(modelo.parameters(), lr=1e-3)


# Entrenar modelo


def train(dataloader, modelo, fn_perdida, optimizador):
    tam = len(dataloader.dataset)  # cargar datos de entrenamiento
    modelo.train()  # entrenar el modelo
    for lote, (X, y) in enumerate(dataloader):
        # Obtener entradas y salidas
        X, y = X.to(dispositivo), y.to(dispositivo)

        # Calcular la predicción para cierta entrada (imágen)
        pred = modelo(X)
        perdida = fn_perdida(pred, y)  # calcular el error

        # Propagación inversa o Backpropagation para actualizar los pesos de
        # las neuronas
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        # Imprimir el valor actual de la función de pérdida
        if lote % 100 == 0:
            perdida, actual = perdida.item(), lote * len(X)
            print(f"perdida: {perdida:>7f}  [{actual:>5d}/{tam:>5d}]")

# Probar el modelo


def test(dataloader, modelo, fn_perdida):
    tam = len(dataloader.dataset)
    num_lotes = len(dataloader)  # numero de lotes
    modelo.eval()  # evaluar el modelo
    test_perdida, correcto = 0, 0
    with torch.no_grad():  # desactivar el gradiente
        for X, y in dataloader:
            X, y = X.to(dispositivo), y.to(dispositivo)  # entradas y salidas
            pred = modelo(X)  # Clasificar una entrada
            # Función de pérdida acumulada
            test_perdida += fn_perdida(pred, y).item()
            # Acumulado de aciertos en la clasificación
            correcto += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_perdida /= num_lotes
    correcto /= tam
    # Calculas promedios y métricas del resultado de la prueba
    print(
        f"Test Error: \n Precision: {(100*correcto):>0.1f}%, Perdida promedio: {test_perdida:>8f} \n")


# Codigo controlador


epocas = 80  # tercer hiperparámetro
# Iteraciones de aprendizaje
# Llamar las funciones para entrenar y probar el modelo
for t in range(epocas):
    print(f"Epoca {t+1}\n-------------------------------")
    train(entrenamiento_dataloader, modelo, fn_perdida, optimizador)
    test(prueba_dataloader, modelo, fn_perdida)
print("Hecho")


torch.save(modelo.state_dict(), "model.pth")
print("Modelo guardado en model.pth")

# Usar el modelo para realizar predicciones

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

# Cada clase correspondiente a las imágenes del dataset
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

modelo.eval()  # evaluar el modelo con la entrada actual

correctos = 0
# Probar las predicciones con 1000 imagenes del dataset
for i in range(0, 1000):
    # la imagen y su clase
    x, y = datos_entrenamiento[i][0], datos_prueba[i][1]
    with torch.no_grad():
        pred = modelo(x)  # Dar la imagen al modelo
        predic, actual = classes[pred[0].argmax(0)], classes[y]
        if predic == actual:
            correctos += 1
        # Determinar si la predicción fue correcta
        print(f'Prediccion: "{predic}", Actual: "{actual}"')

print(f'Total correctos: "{correctos}"')
