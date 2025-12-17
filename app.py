import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
LABELS = {0: 'Overripe', 1: 'Ripe', 2: 'Rotten', 3: 'Unripe'}

# Define the transform pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models (dictionary approach for easy switching)
models = {}
def load_models():
    model_names = ["VGG19", "ResNet50", "GiMaTag_CNN"]
    for name in model_names:
        try:
            # Assumes files are named vgg19.pth, resnet50.pth, etc.
            filename = f"{name.lower()}.pth"
            model = torch.load(filename, map_location=device)
            model.eval()
            models[name] = model
            print(f"Loaded {name}")
        except:
            print(f"⚠️ Model {name} not found. Upload {filename} to fix.")

load_models()

# ---------------------------------------------------------
# 2. PREDICTION FUNCTION
# ---------------------------------------------------------
def predict(image, model_name):
    if image is None:
        return None
    
    if model_name not in models:
        return {"Error": "Model file not found. Please check deployment."}
    
    # Process Image
    image = Image.fromarray(image).convert('RGB')
    input_tensor = transform_pipeline(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        model = models[model_name]
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
    
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# ---------------------------------------------------------
# 3. PROFESSIONAL UI LAYOUT (Using gr.Blocks)
# ---------------------------------------------------------

header_html = """
<div style="display: flex; justify-content: center; align-items: center; text-align: center; padding-bottom: 20px;">
    <div style="display: flex; flex-direction: column; align-items: center;">
        <img src="file/assets/CIC_Logo.png" alt="CIC Logo" style="height: 100px; margin-bottom: 10px;">
        
        <h1 style="margin: 0; font-size: 28px; color: #2C3E50;">Sweet Spot</h1>
        <h3 style="margin: 5px 0; font-weight: normal; color: #555;">Banana Ripeness Classification System</h3>
        <p style="margin: 0; font-size: 14px; font-weight: bold; color: #888;">
            COLLEGE OF INFORMATION AND COMPUTING
        </p>
    </div>
</div>
<hr>
"""
# Custom CSS to make it look cleaner
css = """
.container { max-width: 900px; margin: auto; }
.output-label { font-weight: bold; font-size: 16px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="Sweet Spot | USeP") as demo:
    
    # 1. Header Section
    gr.HTML(header_html)
    
    # 2. Author Credits (Markdown)
    with gr.Row():
        gr.Markdown(
            """
            <div style="text-align: center; font-size: 14px;">
            **Researchers:** Dave Shanna Marie E. Gigawin, Waken Cean C. Maclang, Allan C. Tagle<br>
            *Modelling and Simulation Course Learning Evidence*
            </div>
            """
        )

    # 3. The Main App Interface
    with gr.Row():
        # Left Column: Inputs
        with gr.Column():
            input_image = gr.Image(label="Upload Banana Image", type="numpy", height=300)
            model_selector = gr.Dropdown(
                choices=["VGG19", "ResNet50", "GiMaTag_CNN"], 
                value="VGG19", 
                label="Select Architecture"
            )
            analyze_btn = gr.Button("Analyze Ripeness", variant="primary")
        
        # Right Column: Outputs
        with gr.Column():
            output_label = gr.Label(num_top_classes=4, label="Prediction Results")
    
    # 4. Footer / Paper Abstract
    with gr.Accordion("About the Models", open=False):
        gr.Markdown(
            """
            ### Abstract & Methodology
            This application uses deep learning to classify bananas into **Unripe, Ripe, Overripe, or Rotten**.
            
            * **VGG19:** Transfer learning model. Highest accuracy (**97.97%**) but slower inference.
            * **ResNet50:** Transfer learning model. Good balance of accuracy and speed.
            * **GiMaTag CNN:** A custom 8-layer CNN designed by the researchers. Fastest inference (**61 fps**).
            """
        )

    # Link the button to the function
    analyze_btn.click(fn=predict, inputs=[input_image, model_selector], outputs=output_label)

# Launch
if __name__ == "__main__":
    demo.launch(allowed_paths=["assets/"])