import streamlit as st
import numpy as np
import random
import cv2
import torch
from PIL import Image, ImageDraw

# ==========================
# Detectron2 (Optional)
# ==========================
DETECTRON_AVAILABLE = True

try:
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.engine import DefaultPredictor
except ImportError:
    DETECTRON_AVAILABLE = False


# Airbnb Amenity Classes
subset = [
    'Toilet',
    'Swimming pool',
    'Bed',
    'Billiard table',
    'Sink',
    'Fountain',
    'Oven',
    'Ceiling fan',
    'Television',
    'Microwave oven',
    'Gas stove',
    'Refrigerator',
    'Kitchen & dining room table',
    'Washing machine',
    'Bathtub',
    'Stairs',
    'Fireplace',
    'Pillow',
    'Mirror',
    'Shower',
    'Couch',
    'Countertop',
    'Coffeemaker',
    'Dishwasher',
    'Sofa bed',
    'Tree house',
    'Towel',
    'Porch',
    'Wine rack',
    'Jacuzzi'
]

subset.sort()

CONFIG_FILE = "retinanet_model_final/retinanet_model_final_config.yaml"
MODEL_FILE = "retinanet_model_final/retinanet_model_final.pth"


# ==========================
# Create Predictor
# ==========================
@st.cache_resource
def create_predictor(model_config, model_weights, threshold):

    if not DETECTRON_AVAILABLE:
        return None, None

    cfg = get_cfg()
    cfg.merge_from_file(model_config)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.SCORE_THRESH_TEST = threshold

    predictor = DefaultPredictor(cfg)

    return cfg, predictor


# ==========================
# Real Inference
# ==========================
def make_inference(image,
                   model_config,
                   model_weights,
                   threshold=0.5,
                   n=5):

    cfg, predictor = create_predictor(
        model_config,
        model_weights,
        threshold
    )

    image = np.asarray(image)

    visualizer = Visualizer(
        img_rgb=image,
        metadata=MetadataCatalog.get(
            cfg.DATASETS.TEST[0]
        ).set(
            thing_classes=subset
        ),
        scale=0.3
    )

    outputs = predictor(image)

    instances = outputs["instances"]

    vis = visualizer.draw_instance_predictions(
        instances[:n].to("cpu")
    )

    return vis.get_image(), instances[:n]


# ==========================
# Fake Demo Prediction
# ==========================
def demo_prediction(image, n_boxes):

    image = image.copy()

    draw = ImageDraw.Draw(image)

    width, height = image.size

    detected = []

    for _ in range(n_boxes):

        x1 = random.randint(20, width // 2)
        y1 = random.randint(20, height // 2)

        x2 = min(width - 20, x1 + random.randint(80, 180))
        y2 = min(height - 20, y1 + random.randint(60, 150))

        amenity = random.choice(subset)

        detected.append(amenity)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4
        )

        draw.text(
            (x1 + 5, y1 + 5),
            amenity,
            fill="red"
        )

    return image, detected


# ==========================
# Main App
# ==========================
def main():

    st.title("🏡 Airbnb Amenity Detection")

    if DETECTRON_AVAILABLE:
        st.success("🟢 ML Model Available")
    else:
        st.warning("🟡 Demo Mode (ML model unavailable)")

    st.write(
        """
This application demonstrates Airbnb's object detection system
for recognizing room amenities using Computer Vision.
"""
    )

    st.write("## Example")

    st.image(
        Image.open("images/example-amenity-detection.png"),
        use_column_width=True
    )

    st.write("## Upload Image")

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_column_width=True
        )

        n_boxes = st.slider(
            "Number of amenities",
            1,
            10,
            5
        )

        if st.button("🔍 Make Prediction"):

            with st.spinner("Analyzing image..."):

                if DETECTRON_AVAILABLE:

                    try:

                        pred_image, preds = make_inference(
                            image=image,
                            model_config=CONFIG_FILE,
                            model_weights=MODEL_FILE,
                            n=n_boxes
                        )

                        st.image(
                            pred_image,
                            caption="Detected Amenities",
                            use_column_width=True
                        )

                        classes = np.array(preds.pred_classes)

                        st.success("Detected Amenities")

                        for i in classes:
                            st.write("✅", subset[i])

                    except Exception as e:

                        st.error(
                            "Model could not be loaded."
                        )

                        st.code(str(e))

                else:

                    pred_image, detected = demo_prediction(
                        image,
                        n_boxes
                    )

                    st.image(
                        pred_image,
                        caption="Demo Detection",
                        use_column_width=True
                    )

                    st.success("Detected Amenities")

                    for amenity in detected:
                        st.write("✅", amenity)

                    st.info(
                        "This is a frontend demonstration. "
                        "Install Detectron2 and provide the trained "
                        "model to enable real predictions."
                    )

    st.write("---")

    st.subheader("About")

    st.write(
        """
This project demonstrates object detection for Airbnb room
amenities using Detectron2, PyTorch and Streamlit.
"""
    )


if __name__ == "__main__":
    main()
