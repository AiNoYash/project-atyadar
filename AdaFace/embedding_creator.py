import os
import torch
import torchvision.transforms as transforms
from PIL import Image
# Assuming you have downloaded the weights and can import their model architecture
from AdaFace.net import build_model 

def load_adaface_model(weight_path):
    model = build_model('ir_50') 
    statedict = torch.load(weight_path)['state_dict']
    model_statedict = {key.replace('model.', ''): val for key, val in statedict.items()}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def create_embeddings(image_folder, model):
    # AdaFace expects 112x112 aligned cropped faces, normalized
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    embeddings = {}
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                feat, _ = model(tensor)
                embeddings[filename] = feat.squeeze().numpy()
                
    return embeddings

if __name__ == "__main__":
    model = load_adaface_model('adaface_ir50_ms1mv2.ckpt')
    db_embeddings = create_embeddings('indian_faces_50', model)
    torch.save(db_embeddings, 'database_embeddings.pt')