import torch
import torch.nn as nn

# 1. DEFINE THE CLASS (Matches your training code)
class GiMaTagCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(GiMaTagCNN, self).__init__()
        
        # --- Block 1 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        # --- Block 2 ---
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )

        # --- Block 3 ---
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.40)
        )

        flattened_size = 64 * 28 * 28

        # --- Fully Connected ---
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(0.55),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 2. LOAD & SAVE WITH FIX
if __name__ == "__main__":
    model = GiMaTagCNN(num_classes=4)
    
    # Ensure this matches your exact filename
    pth_file_path = "models/gimatag.pth" 
    
    try:
        print(f"📂 Loading weights from: {pth_file_path}")
        
        # --- THE FIX: ADD weights_only=False ---
        state_dict = torch.load(pth_file_path, map_location=torch.device('cpu'), weights_only=False)
        
        model.load_state_dict(state_dict)
        model.eval()
        
        print("✅ Weights loaded successfully!")
        
        # Trace and Save
        print("   Tracing model architecture...")
        example_input = torch.rand(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model, example_input)
        
        # Save to 'models' folder directly
        traced_script_module.save("models/gimatag_complete.pt")
        
        print("🎉 SUCCESS! Saved as 'models/gimatag_complete.pt'")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file at {pth_file_path}")
    except Exception as e:
        print(f"❌ Error: {e}")