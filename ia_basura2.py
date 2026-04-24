import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from rembg import remove
import os

def predict_garbage(image_path, model_path):
    # 1. Definir las clases
    class_names = ['Cartón', 'Vidrio', 'Metal', 'Papel', 'Plástico', 'Resto']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Reconstruir la arquitectura ConvNeXt
    model = models.convnext_tiny(weights=None)

    # Según tu error, el modxelo guardado tiene una capa extra (posiblemente un Sequential) 
    # en la posición 2 del clasificador.
    num_ftrs = model.classifier[2].in_features

    # Reemplazamos la capa lineal simple por una que coincida con tus "Unexpected keys"
    # Esto crea la estructura classifier.2.0 que pide tu archivo .pth
    model.classifier[2] = nn.Sequential(
        nn.Linear(num_ftrs, len(class_names))
    )

    # 3. Cargar los pesos guardados
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("✓ Modelo ConvNeXt cargado correctamente (ajustado a Sequential).")
    except Exception as e:
        print(f"X Error al cargar: {e}")
        return
    
    # 4. Transformaciones
    # ConvNeXt suele entrenarse con 224x224, pero es más sensible a la normalización
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 5. Cargar y procesar la imagen
    if not os.path.exists(image_path):
        print(f"Archivo no encontrado: {image_path}")
        return

    image = Image.open(image_path).convert('RGB')
    
    # Eliminar fondo y poner fondo blanco
    img_no_bg = remove(image)
    fondo_blanco = Image.new("RGBA", img_no_bg.size, (255, 255, 255, 255))
    image_final = Image.alpha_composite(fondo_blanco, img_no_bg).convert("RGB")

    image_tensor = inference_transform(image_final).unsqueeze(0).to(device)
    
    # 6. Predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # 7. Agrupación por contenedores
    resumen_contenedores = {
        "Azul 🔵 (Papel/Cartón)": (probabilities[0] + probabilities[3]).item(),
        "Amarillo 🟡 (Envases/Metal)": (probabilities[2] + probabilities[4]).item(),
        "Verde 🟢 (Vidrio)": probabilities[1].item(),
        "Gris ⚫ (Resto)": probabilities[5].item()
    }

    # 8. Mostrar resultados
    print("-" * 40)
    print(f"{'CLASE':<15} | {'PROBABILIDAD':>12}")
    print("-" * 40)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} | {probabilities[i].item() * 100:>11.2f}%")
    
    print("-" * 40)
    print("RESUMEN POR CONTENEDOR:")
    for cont, valor in resumen_contenedores.items():
        print(f"{cont:<25}: {valor * 100:>6.2f}%")

    # 9. Decisión final
    ganador_cont = max(resumen_contenedores, key=resumen_contenedores.get)
    confianza_final = resumen_contenedores[ganador_cont]

    print("-" * 40)
    print(f"DECISIÓN: {ganador_cont}")
    print(f"CONFIANZA: {confianza_final * 100:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    # Asegúrate de que la carpeta existe
    if not os.path.exists("./imagenes"):
        os.makedirs("./imagenes")
        
    while True:
        nombre_foto = input("Nombre de la foto (o 'salir'): ")
        if nombre_foto.lower() == 'salir':
            break
            
        path = f"./imagenes/{nombre_foto}.jpg"
        predict_garbage(path, 'garbage_convnext.pth')