import os
import cv2
import torch
from ultralytics import YOLO

def main():
    # 1. Charger ton modèle personnalisé
    model = YOLO("best.pt") 
    
    # 2. Définir la source (Remplace par l'IP de ton téléphone ou ta vidéo)
    source_flux = "http://10.132.234.243:8080/video" 

    # 3. Créer le dossier pour stocker les photos uniques si de n'existe pas
    dossier_sauvegarde = "plaques_enregistrees"
    os.makedirs(dossier_sauvegarde, exist_ok=True)

    # Liste pour garder en mémoire les plaques déjà photographiées
    plaques_deja_sauvegardees = set()

    print("🎥 Lancement de la détection (Une seule photo par voiture)...")
    print("Appuie sur 'Q' pour quitter.")

    choix_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4. Lancer le tracking (suivi)
    results = model.track(
        source=source_flux, 
        stream=True, 
        device=choix_device,    
        show=True,             
        conf=0.6              # Seuil à 60% pour éviter les fausses détections
    )

    # Boucle pour analyser le flux image par image
    for r in results:
        # On vérifie si YOLO a trouvé des plaques dans l'image actuelle
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()  # Coordonnées des rectangles
            ids = r.boxes.id.cpu().numpy().astype(int)  # Identifiants uniques des voitures
            
            # Pour chaque plaque détectée sur l'image
            for box, track_id in zip(boxes, ids):
                # ⭐ LE FILTRE : Si cet ID de voiture n'a JAMAIS été photographié
                if track_id not in plaques_deja_sauvegardees:
                    
                    # 1. On récupère les coordonnées du rectangle de la plaque
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 2. On découpe la plaque dans l'image d'origine (Crop)
                    image_origine = r.orig_img
                    img_plaque = image_origine[y1:y2, x1:x2]
                    
                    # 3. On vérifie que la découpe n'est pas vide
                    if img_plaque.size > 0:
                        # On crée un nom de fichier unique avec l'ID de la voiture
                        nom_fichier = os.path.join(dossier_sauvegarde, f"voiture_ID_{track_id}.jpg")
                        
                        # Enregistrement de la photo sur le PC
                        cv2.imwrite(nom_fichier, img_plaque)
                        print(f"📸 Nouvelle voiture détectée ! Photo enregistrée : {nom_fichier}")
                        
                        # On ajoute l'ID à notre liste pour NE PLUS la photographier
                        plaques_deja_sauvegardees.add(track_id)

if __name__ == "__main__":
    main()