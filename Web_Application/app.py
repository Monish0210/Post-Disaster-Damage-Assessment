import os
from flask import Flask, render_template, request, url_for
from PIL import Image
import torch
from torchvision import transforms
from model import SiamUnet
import json
import uuid
import numpy as np
import sys

sys.modules['__main__'] = sys.modules['model']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_PATH'] = os.path.join(os.path.dirname(__file__), 'siamUnet.pt')
app.config['DEFAULT_GSD'] = 0.5
app.config['ORIGINAL_IMAGE_SIZE'] = 1024
app.config['MODEL_IMAGE_SIZE'] = 256

model = None
device = torch.device('cpu')

transform = transforms.Compose([
    transforms.Resize((app.config['MODEL_IMAGE_SIZE'], app.config['MODEL_IMAGE_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DAMAGE_CLASSES = {
    0: 'Background',
    1: 'No Damage',
    2: 'Minor Damage',
    3: 'Major Damage',
    4: 'Destroyed'
}
CLASS_COLORS_RGB = [
    (0, 0, 0),       # 0: Background (Black)
    (0, 255, 0),     # 1: No Damage (Green)
    (255, 165, 0),   # 2: Minor Damage (Orange)
    (128, 0, 128),   # 3: Major Damage (Purple)
    (255, 0, 0)      # 4: Destroyed (Red)
]


def load_model_if_needed():
    global model
    if model is None:
        print("Loading model...")
        try:
            model_path = app.config['MODEL_PATH']
            if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found at {model_path}")
            model = torch.load(model_path, map_location=device, weights_only=False)
            model.eval()
            print("Model loaded.")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            model = "not_found"

@app.before_request
def before_request():
    load_model_if_needed()

@app.route('/', methods=['GET', 'POST'])
def index():
    if model == "not_found":
        return f"Model file '{app.config['MODEL_PATH']}' not found or failed to load.", 500

    if request.method == 'POST':
        if 'pre_image' not in request.files or 'post_image' not in request.files:
            return "Missing pre or post image file", 400
            
        pre_image_file = request.files['pre_image']
        post_image_file = request.files['post_image']

        if pre_image_file.filename == '' or post_image_file.filename == '':
            return "No selected file", 400
        
        try:
            gsd = float(request.form.get('gsd', app.config['DEFAULT_GSD']))
        except ValueError:
            gsd = app.config['DEFAULT_GSD']
            
        
        original_area_per_pixel = gsd * gsd
        scale_factor_squared = (app.config['ORIGINAL_IMAGE_SIZE'] / app.config['MODEL_IMAGE_SIZE']) ** 2
        area_per_predicted_pixel = scale_factor_squared * original_area_per_pixel
        

        unique_id = str(uuid.uuid4())
        pre_filename = f"pre_{unique_id}_{pre_image_file.filename}"
        post_filename = f"post_{unique_id}_{post_image_file.filename}"
        output_filename = f"output_{unique_id}.png"

        pre_image_path = os.path.join(app.config['UPLOAD_FOLDER'], pre_filename)
        post_image_path = os.path.join(app.config['UPLOAD_FOLDER'], post_filename)
        pre_image_file.save(pre_image_path)
        post_image_file.save(post_image_path)

        try:
            pre_image = Image.open(pre_image_path).convert("RGB")
            post_image = Image.open(post_image_path).convert("RGB")
            pre_image_tensor = transform(pre_image).unsqueeze(0).to(device)
            post_image_tensor = transform(post_image).unsqueeze(0).to(device)

            with torch.no_grad():
                _, _, damage_output = model(pre_image_tensor, post_image_tensor)
                preds_cls_tensor = torch.argmax(torch.nn.functional.softmax(damage_output, dim=1), dim=1)
                
            # --- Analysis Calculation ---
            preds_cls_np = preds_cls_tensor.cpu().numpy().flatten() 
            unique_classes, counts = np.unique(preds_cls_np, return_counts=True)
            total_predicted_pixels = preds_cls_np.size
            
            damage_analysis_percent, damage_analysis_area = {}, {}
            total_damaged_area = 0.0

            for i, count in zip(unique_classes, counts):
                class_name = DAMAGE_CLASSES.get(i, "Unknown")
                percentage = (count / total_predicted_pixels) * 100
                area_m2 = count * area_per_predicted_pixel 
                
                damage_analysis_percent[class_name] = percentage
                damage_analysis_area[class_name] = area_m2 

                if i > 0: # Assumes class 0 is 'Background' and non-damage
                    total_damaged_area += area_m2
            
          
            colors = torch.tensor(CLASS_COLORS_RGB, dtype=torch.uint8)
            output_image_tensor = colors[preds_cls_tensor.squeeze(0).cpu().long()].permute(2, 0, 1)
            output_image = transforms.ToPILImage()(output_image_tensor)
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            output_image.save(output_image_path)

            return render_template('index.html',
                                   pre_image_rel_path=url_for('static', filename=f'uploads/{pre_filename}'),
                                   post_image_rel_path=url_for('static', filename=f'uploads/{post_filename}'),
                                   output_image_rel_path=url_for('static', filename=f'uploads/{output_filename}'),
                                   damage_analysis_percent_json=json.dumps(damage_analysis_percent),
                                   damage_analysis_area_json=json.dumps(damage_analysis_area),
                                   damage_analysis_area_dict=damage_analysis_area, 
                                   total_damaged_area=total_damaged_area,
                                   gsd_used=gsd)
        except Exception as e:
            print(f"Error during prediction: {e}")
            if os.path.exists(pre_image_path): os.remove(pre_image_path)
            if os.path.exists(post_image_path): os.remove(post_image_path)
            if 'output_image_path' in locals() and os.path.exists(output_image_path): os.remove(output_image_path)
            return f"An error occurred during processing: {e}", 500

    return render_template('index.html', gsd_used=app.config['DEFAULT_GSD']) 


@app.route('/evaluation')
def evaluation():
    metrics = {
        "Overall Pixel Accuracy": 96.61, 
        "Mean IoU (mIoU)": 49.69,         
        "Macro Average Precision": 65.97, 
        "Macro Average Recall": 58.97,    
        "Macro Average F1-Score": 61.81, 
        "class_metrics": [
            {"name": "Background", "iou": 96.87, "precision": 98.67, "recall": 98.16, "f1": 98.41},  # Example
            {"name": "No Damage", "iou": 57.95, "precision": 67.49, "recall": 80.38, "f1": 73.38},   # Example
            {"name": "Minor Damage", "iou": 7.90, "precision": 32.12, "recall": 9.48, "f1": 9.48},  # Example
            {"name": "Major Damage", "iou": 38.62, "precision": 63.28, "recall": 49.77, "f1": 55.72},  # Example
            {"name": "Destroyed", "iou": 38.27, "precision": 53.01, "recall": 57.93, "f1": 55.36}      # Example
        ]
    }
    return render_template('evaluation.html', metrics=metrics)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)