import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from AdaFace.net import build_model

class AdaFaceMatcher:
    def __init__(self, weight_path='adaface_ir50_ms1mv2.ckpt', database_path='database_embeddings.pt'):
        """
        Initializes the matcher, loading the model and database embeddings into memory
        for fast, repeated API inference.
        """
        self.weight_path = weight_path
        self.database_path = database_path
        self.model = self._load_model(weight_path)
        self.showGuidance = False
        
        self.bestMatch = None
        self.bestSim = -1
        
        self.prevMatch = None
        self.prevSim = -1
        
        if os.path.exists(database_path):
            self.db_embeddings = torch.load(database_path, weights_only=False)
        else:
            self.db_embeddings = {}
            print(f"Warning: {database_path} not found. Use create_database_embeddings first.")

    def _load_model(self, weight_path):
        model = build_model('ir_50')
        statedict = torch.load(weight_path, weights_only=False)['state_dict']
        model_statedict = {key.replace('model.', ''): val for key, val in statedict.items()}
        model.load_state_dict(model_statedict, strict=False)
        model.eval()
        return model

    @staticmethod
    def get_embedding(image_path, model):
        """
        Static method to extract an embedding from a single image.
        """
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            feat, _ = model(tensor)
            return feat.squeeze()

    @staticmethod
    def create_database_embeddings(image_folder, model, output_path='database_embeddings.pt'):
        """
        Static method to process a folder of images and save their embeddings.
        """
        embeddings = {}
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_folder, filename)
                try:
                    feat = AdaFaceMatcher.get_embedding(img_path, model)
                    embeddings[filename] = feat.numpy()
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
                    
        torch.save(embeddings, output_path)
        print(f"Saved {len(embeddings)} embeddings to {output_path}")
        return embeddings

    def find_best_match(self, test_image_path):
        """
        Instance method used by your API endpoint to find the closest match.
        """
        if not self.db_embeddings:
            raise ValueError("Database embeddings are not loaded.")
            
        test_feat = self.get_embedding(test_image_path, self.model)
        
        best_match = None
        highest_sim = -1.0
        
        for filename, db_feat in self.db_embeddings.items():
            db_tensor = torch.tensor(db_feat)
            sim = F.cosine_similarity(test_feat.unsqueeze(0), db_tensor.unsqueeze(0)).item()
            
            if sim > highest_sim:
                highest_sim = sim
                best_match = filename
        
        
        if self.showGuidance:
            self.prevMatch = self.bestMatch
            self.prevSim = self.bestSim
            
            self.bestMatch = best_match
            self.bestSim = highest_sim
        
                
        return best_match, highest_sim
    
    
    def guide(self):
        """
        Instance method used to turn on/off guidance.
        """
        self.showGuidance = not self.showGuidance
    
    
    def giveGuidance(self):
        """
        Instance method used to get the guidance object based on previous inputs.
        """
        if not self.showGuidance:
            print(f"Warning: Turn on guidance mode to use it.")
            return

        
        improvedResults = self.bestSim >= self.prevSim
        targetChanged = self.bestMatch != self.prevMatch
        
        guidanceObj = {
            "improvedResults": improvedResults,
            "targetChanged": targetChanged,
            "currentSim": self.bestSim,
            "prevSim": self.prevSim,
            "bestMatch": self.bestMatch,
            "prevMatch": self.prevMatch,
            }

        return guidanceObj