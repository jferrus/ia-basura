import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

def main():
    # 1. Configuración de rutas y parámetros
    data_dir = './datasets/Garbage classification/Garbage classification'
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # Detectar dispositivo (GPU si está disponible)
    # Detecta si es un Mac con GPU (MPS), si no, usa CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Usando dispositivo: {device}")

    # 2. Transformaciones de imagen (Data Augmentation para entrenamiento)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Cargar Dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    classes = full_dataset.classes
    num_classes = len(classes)
    print(f"Clases detectadas: {classes}")

    # Dividir en Entrenamiento (80%) y Validación (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 4. Definir el Modelo (ResNet18 - Ligero y eficiente)
    print("Configurando el modelo...")
    model = models.resnet18(pretrained=True)
    
    # Congelar capas base
    for param in model.parameters():
        param.requires_grad = False
    
    # Cambiar la capa final para nuestras clases de basura
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # 5. Función de pérdida y Optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # 6. Ciclo de entrenamiento
    print("Iniciando entrenamiento...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validación
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f} - Val Accuracy: {100 * correct / total:.2f}%")

    # 7. Guardar el modelo entrenado
    torch.save(model.state_dict(), 'garbage_classifier2.pth')
    print("Modelo guardado como garbage_classifier2.pth")

if __name__ == "__main__":
    main()