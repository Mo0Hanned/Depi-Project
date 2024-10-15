from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

app = Flask(__name__)

# Load your model
device = "cpu"  # Ensure this is set to "cpu"
num_classes = 4  # Adjust based on your number of classes

# Define the ResNet18 model
resnet18_model = models.resnet18(weights=None)
resnet18_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet18_model.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.Linear(256, num_classes)
)

# Load the saved model weights
try:
  resnet18_model.load_state_dict(torch.load(r"D:\.folder e\work space\projects\project 3\models\chest-ctscan_model.pth", map_location='cpu', weights_only=True))

except RuntimeError as e:
    print(f"Error loading model: {e}")

resnet18_model.to(device)
resnet18_model.eval()

# Define the data transformation
data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
])

# Define the prediction function
def predict_image(image, model, transform, device):
    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image)
        predicted_class = outputs.argmax(dim=1).item()

    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
     try:   
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if not file:
            return jsonify({'error': 'No file provided'}), 400

        image = Image.open(file.stream).convert("RGB")
        predicted_class = predict_image(image, resnet18_model, data_transform, device)

        # Define class names according to your dataset
        class_names = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]
        return jsonify({'predicted_class': class_names[predicted_class]})
     except Exception as e:
       print("Error during prediction:", str(e))
       return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
