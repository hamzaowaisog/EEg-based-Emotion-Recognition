import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTModel, ViTFeatureExtractor

def load_vit_model():
    model_name = "google/vit-base-patch16-224-in21k"
    model = ViTModel.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor

def extract_vit_features(face_img_path, model, feature_extractor):
    image = Image.open(face_img_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Global feature vector


if __name__ == "__main__":
    vit_model, vit_extractor = load_vit_model()
    face_image_path = r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Pytorch_Retinaface-master\output\face.jpg"  # Output from RetinaFace
    features = extract_vit_features(face_image_path, vit_model, vit_extractor)
    print("Extracted Features Shape:", features.shape)
