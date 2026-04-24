import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from rembg import remove
import os

def predict_garbage(image_path, model_path):
    # 1. Definir las clases (Deben estar en el mismo orden que tus carpetas)
    # ImageFolder las ordena alfabéticamente por defecto
    class_names = ['Cartón', 'Vidrio', 'Metal', 'Papel', 'Plástico', 'Resto']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contendedores = ['Azul 🔵', 'Verde 🟢', 'Amarillo 🟡', 'Azul 🔵', 'Amarillo 🟡', 'Gris ⚫']

    # 2. Reconstruir la arquitectura EXACTAMENTE como en el entrenamiento
    model = models.resnet18(weights=None) # No necesitamos pesos preentrenados aquí
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names)) # Aquí es donde debe dar 6
    
    # 3. Cargar los pesos guardados
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("✓ Modelo cargado correctamente.")
    except Exception as e:
        print(f"X Error al cargar: {e}")
        return

    # 4. Transformaciones (Usamos las de validación: Resize + Normalize)
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
    img_no_bg = remove(image)

    # 3. Crear un fondo blanco sólido
    # (Muchos modelos de basura fallan si el fondo es negro o transparente)
    fondo_blanco = Image.new("RGBA", img_no_bg.size, (255, 255, 255, 255))
        
        # 4. Pegar el objeto sobre el fondo blanco
    image_final = Image.alpha_composite(fondo_blanco, img_no_bg).convert("RGB")

    image_tensor = inference_transform(image_final).unsqueeze(0).to(device)
    
    # 6. Predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # 7. Agrupación por contenedores
    # Índices: 0:Cartón, 1:Vidrio, 2:Metal, 3:Papel, 4:Plástico, 5:Resto
    # Azul = Cartón (0) + Papel (3)
    # Amarillo = Metal (2) + Plástico (4)
    
    prob_azul = (probabilities[0] + probabilities[3]).item()
    prob_amarillo = (probabilities[2] + probabilities[4]).item()
    prob_verde = probabilities[1].item()
    prob_blanco = probabilities[5].item()

    # Creamos un diccionario para mapear el contenedor con su suma
    resumen_contenedores = {
        "Azul 🔵 (Papel/Cartón)": prob_azul,
        "Amarillo 🟡 (Envases/Metal)": prob_amarillo,
        "Verde 🟢 (Vidrio)": prob_verde,
        "Blanco ⚪️ (Resto)": prob_blanco
    }

    # 8. Mostrar resultados detallados
    print("-" * 35)
    print("DESGLOSE POR CLASE INDIVIDUAL:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<10}: {probabilities[i].item() * 100:>6.2f}%")
    
    print("-" * 35)
    print("PROBABILIDAD POR CONTENEDOR:")
    for cont, valor in resumen_contenedores.items():
        print(f"{cont:<25}: {valor * 100:>6.2f}%")

    # 9. Determinar el ganador basado en la suma
    ganador_cont = max(resumen_contenedores, key=resumen_contenedores.get)
    confianza_final = resumen_contenedores[ganador_cont]

    print("-" * 35)
    print(f"DECISIÓN FINAL: {ganador_cont}")
    print(f"CONFIANZA TOTAL: {confianza_final * 100:.2f}%")
    print("-" * 35)

if __name__ == "__main__":
    while(True):
        nombre_foto = input("Introduce el nombre de la foto: ")

        IMAGEN_TEST = f"./imagenes/{nombre_foto}.jpg"
        predict_garbage(IMAGEN_TEST, 'garbage_classifier.pth')
    