import gradio as gr
import torch
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
LABELS = {0: 'Overripe', 1: 'Ripe', 2: 'Rotten', 3: 'Unripe'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 2. LOAD COMPLETE MODELS
# ---------------------------------------------------------
models_dict = {}

def load_complete_model(filename, key_name):
    try:
        print(f"📂 Loading {key_name}...")
        # Load the TorchScript model (structure + weights)
        model = torch.jit.load(f"models/{filename}", map_location=device)
        model.eval()
        models_dict[key_name] = model
        print(f"✅ {key_name} Ready")
    except Exception as e:
        print(f"❌ {key_name} Failed: {e}")

# Load models (Make sure these .pt files exist in your 'models' folder!)
load_complete_model("gimatag_complete.pt", "GiMaTag")
load_complete_model("vgg19_complete.pt", "VGG19")
load_complete_model("resnet50_complete.pt", "ResNet50")

# ---------------------------------------------------------
# 3. PREDICTION LOGIC
# ---------------------------------------------------------
def predict(image, model_name):
    if image is None: return None
    if model_name not in models_dict: 
        return {f"Error: {model_name} not loaded": 0.0}
    
    image = Image.fromarray(image).convert('RGB')
    tensor = transform_pipeline(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = models_dict[model_name](tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# ---------------------------------------------------------
# 4. PROFESSIONAL UI (CSS & HTML)
# ---------------------------------------------------------

# Custom CSS for the "Banana Theme"
custom_css = """
.container { max-width: 1200px; margin: auto; padding-top: 20px; }
.header-container { 
    text-align: center; 
    padding: 30px; 
    background: linear-gradient(135deg, #fff8e1 0%, #ffffff 100%); 
    border-radius: 15px; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
    margin-bottom: 25px;
    border-bottom: 4px solid #f1c40f;
}
.logo-img { height: 90px; margin-bottom: 15px; }
.app-title { font-size: 32px; font-weight: 800; color: #2c3e50; margin: 0; }
.app-subtitle { font-size: 18px; color: #7f8c8d; margin-top: 5px; font-weight: 400; }
.college-text { font-size: 12px; font-weight: bold; color: #f39c12; letter-spacing: 1px; margin-top: 15px; text-transform: uppercase; }

/* Card Styling for Input/Output */
.gr-group { 
    background-color: white; 
    border-radius: 12px; 
    box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
    padding: 20px; 
    border: 1px solid #eee;
}

/* Button Styling */
#analyze-btn { 
    background: linear-gradient(to right, #f1c40f, #f39c12); 
    border: none; 
    color: white; 
    font-weight: bold; 
    font-size: 16px; 
    height: 50px;
    border-radius: 8px;
    transition: 0.3s;
}
#analyze-btn:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 5px 15px rgba(241, 196, 15, 0.4); 
}

/* Footer Styling */
.footer { text-align: center; margin-top: 40px; color: #95a5a6; font-size: 12px; }
"""

# HTML Content for Header
header_html = """
<div class="header-container">
    <h1 class="app-title">Sweet Spot</h1>
    <h3 class="app-subtitle">Banana Ripeness Classification System</h3>
    <p class="college-text">College of Information and Computing</p>
</div>
"""

# Build the App
with gr.Blocks(title="Sweet Spot | USeP", css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.HTML(header_html)
    
    with gr.Row():
        # --- LEFT COLUMN: INPUTS ---
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🍌 Image Analysis")
                input_image = gr.Image(label="Upload Banana Image", type="numpy", height=320)
                
                gr.Markdown("### 🧠 Model Settings")
                model_selector = gr.Dropdown(
                    choices=["VGG19", "ResNet50", "GiMaTag"], 
                    value="VGG19", 
                    label="Select Architecture",
                    info="Choose the neural network to perform the analysis."
                )
                
                analyze_btn = gr.Button("✨ Analyze Ripeness", elem_id="analyze-btn")

        # --- RIGHT COLUMN: RESULTS ---
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📊 Prediction Results")
                # Label provides the progress bars automatically
                output_label = gr.Label(num_top_classes=4, label="Confidence Scores", show_label=False)
            
            with gr.Accordion("ℹ️ About the Classes", open=True):
                gr.Markdown("""
                * **Unripe:** Green skin, firm texture.
                * **Ripe:** Yellow skin, optimal sweetness.
                * **Overripe:** Brown spots, soft texture.
                * **Rotten:** Mostly black/brown, unsafe to eat.
                """)

    # Footer
    gr.Markdown(
        """
        <div class="footer">
        <b>Researchers:</b> Dave Shanna Marie E. Gigawin, Waken Cean C. Maclang, Allan C. Tagle <br>
        Modelling and Simulation Course Learning Evidence • 2025
        </div>
        """
    )

    # Logic
    analyze_btn.click(fn=predict, inputs=[input_image, model_selector], outputs=output_label)

# Launch with assets allowed
if __name__ == "__main__":
    demo.launch(allowed_paths=["."], share=True)