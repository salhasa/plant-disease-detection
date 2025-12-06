from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as models
from ultralytics import YOLO

app = Flask(__name__, static_folder="static", static_url_path="/static")

# -------------------------------------
# HTML ROUTES
# -------------------------------------
@app.route("/")
def home_page():
    # الصفحة الرئيسية (landing page)
    return send_from_directory(".", "index.html")

@app.route("/classifier")
def classifier_page():
    # صفحة رفع الصورة والتصنيف
    return send_from_directory(".", "predict.html")

@app.route("/project-details")
def project_details_page():
    # ✅ صفحة تفاصيل المشروع الجديدة
    return send_from_directory(".", "project-details.html")


# -------------------------------------
# CLASS NAMES (same order as training)
# copied from your app.py / notebook
# -------------------------------------
CLASS_NAMES = [
    "Apple Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Apple Rust",
    "Apple Healthy",
    "Bell Pepper Bacterial Spot",
    "Bell Pepper Healthy",
    "Cherry Healthy",
    "Cherry Powdery Mildew",
    "Corn Maize Cercospora Leaf Spot",
    "Corn Maize Common Rust",
    "Corn Maize Healthy",
    "Corn Maize Northern Leaf Blight",
    "Grape Black Rot",
    "Grape Esca Black Measles",
    "Grape Healthy",
    "Grape Leaf Blight",
    "Peach Bacterial Spot",
    "Peach Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Strawberry Healthy",
    "Strawberry Leaf Scorch",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Septoria Leaf Spot",
    "Tomato Yellow Leaf Curl Virus",
]

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

# -------------------------------------
# LOAD MODELS
# -------------------------------------
# YOLO classification model
YOLO_PATH = "models/yolo11.pt"
model1 = YOLO(YOLO_PATH)

# MobileNetV3-Large model
MNV3_PATH = "models/mobilenetv3_large_best.pth"
model2 = models.mobilenet_v3_large(num_classes=NUM_CLASSES)
state = torch.load(MNV3_PATH, map_location="cpu")
model2.load_state_dict(state, strict=False)
model2.eval()

# -------------------------------------
# TRANSFORMS (MATCH COLAB / TRAINING)
# -------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# -------------------------------------
# HELPERS
# -------------------------------------
def get_health_status(class_name: str) -> str:
    """Return Healthy / Not Healthy based on class name."""
    return "Healthy" if "Healthy" in class_name else "Not Healthy"


def predict_mobilenet(image: Image.Image):
    """Prediction using MobileNetV3 with the SAME pipeline as in Colab."""
    # Fix image orientation based on EXIF (same as ImageOps.exif_transpose in notebook)
    img = ImageOps.exif_transpose(image)

    x = eval_tfms(img).unsqueeze(0)

    with torch.no_grad():
        logits = model2(x)
        probs = F.softmax(logits, dim=1)[0]

    idx = torch.argmax(probs).item()
    class_name = CLASS_NAMES[idx]
    health = get_health_status(class_name)
    return class_name, health


# -------------------------------------
# API ENDPOINT
# -------------------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")

    # YOLO prediction (YOLO has its own internal transforms)
    y_res = model1.predict(img, imgsz=224, verbose=False)[0]
    y_idx = int(y_res.probs.top1)
    y_class_name = CLASS_NAMES[y_idx]
    y_health = get_health_status(y_class_name)

    # MobileNet prediction (using eval_tfms that matches Colab)
    m_class_name, m_health = predict_mobilenet(img)

    return jsonify({
        "model1_class": y_class_name,
        "model1_health": y_health,
        "model2_class": m_class_name,
        "model2_health": m_health,
    })


if __name__ == "__main__":
    app.run(debug=True)
