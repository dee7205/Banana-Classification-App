import torch
import torch.nn as nn
from torchvision import models

# ==========================================
# 1. DEFINE RESNET50 ARCHITECTURE
# ==========================================
resnet50_base = models.resnet50(weights=None) 

class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            *list(resnet50_base.children())[:-1]
        )
        INPUT_FEATURES = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_FEATURES, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. DEFINE VGG19 ARCHITECTURE
# ==========================================
vgg19_base = models.vgg19(weights=None)

class VGG19Transfer(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = vgg19_base.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ==========================================
# 3. CONVERT AND SAVE
# ==========================================
def save_complete_model(model_class, pth_path, save_name):
    print(f"------------------------------------------------")
    print(f"📂 Processing: {save_name}")
    
    try:
        # Initialize empty model
        model = model_class(num_classes=4)
        
        # Load weights with SECURITY CHECK DISABLED (weights_only=False)
        print(f"   Loading weights from: {pth_path}")
        
        # --- THE FIX IS HERE ---
        state_dict = torch.load(pth_path, map_location=torch.device('cpu'), weights_only=False) 
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Trace
        print("   Tracing model architecture...")
        dummy_input = torch.rand(1, 3, 224, 224)
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Save
        traced_model.save(f"models/{save_name}")
        print(f"✅ Success! Saved as: models/{save_name}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{pth_path}'")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Ensure these match your filenames
    path_to_resnet_pth = "models/resnet50.pth" 
    path_to_vgg_pth    = "models/vgg19.pth"
    
    # Run conversion
    save_complete_model(ResNet50Transfer, path_to_resnet_pth, "resnet50_complete.pt")
    save_complete_model(VGG19Transfer, path_to_vgg_pth, "vgg19_complete.pt")