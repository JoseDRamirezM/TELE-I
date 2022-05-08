# PREREQUISITOS

import torch  # cargar el core de la librería
from torch import nn  # cargar modelo de redes neuronales
from torch.utils.data import DataLoader  # Módulo para cargar datos
from torchvision import datasets  # Módulo para cargar datos
from torchvision.transforms import ToTensor  # Módulo para transformar datos
import matplotlib.pyplot as plt  # Módulo para visualizar el Dataset

# Cargar datos de entrenamiento de pytorch
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Cargar datos de prueba de pytorch
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Estas son las clases en las
# que se clasifican las imágenes
labels_map = {
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
figure = plt.figure(figsize=(8, 8))
cols, rows = 9, 9
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# Primer hiperparámetro
batch_size = 64

# Primer hiperparámetro
train_dataloader = DataLoader(training_data, batch_size=batch_size)
# Dataloader para datos de prueba
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# Definir un dispositivo CPU o GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


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


# Montar el modelo en el dispositivo
model = NeuralNetwork().to(device)
print(model)

# OPTIMIZAR LOS PARÁMETROS

# Función de pérdida para calcular el error
loss_fn = nn.CrossEntropyLoss()
# Algoritmo de optimización de
# gradiente descendente y segundo hiperparámetro
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Entrenar modelo


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # cargar datos de entrenamiento
    model.train()  # entrenar el modelo
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # Obtener entradas y salidas

        # Calcular la predicción del error
        pred = model(X)
        loss = loss_fn(pred, y)  # calcular el error

        # Propagación inversa o Backpropagation para actualizar los pesos de
        # las neuronas
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Imprimir el valor actual de la función de pérdida
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Probar el modelo


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)  # tamaño del lote
    model.eval()  # evaluar el modelo
    test_loss, correct = 0, 0
    with torch.no_grad():  # desactivar el gradiente
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # entradas y salidas
            pred = model(X)  # Clasificar una entrada
            # Función de pérdida acumulada
            test_loss += loss_fn(pred, y).item()
            # Acumulado de aciertos en la clasificación
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    # Calculas promedios y métricas del resultado de la prueba
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Codigo controlador

epochs = 5  # tercer hiperparámetro
# Llamar las funciones para entrenar y probar el modelo
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

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

model.eval()  # evaluar el modelo con la entrada actual
x, y = test_data[0][0], test_data[0][1]  # la imagen y su clase
with torch.no_grad():
    pred = model(x)  # Dar la imagen al modelo
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    # Determinar si la predicción fue correcta
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
