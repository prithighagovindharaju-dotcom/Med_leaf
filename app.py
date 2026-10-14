from config import IS_COLAB
import gradio as gr
from predict import load_model, predict

## Model oru thadava mattum load aagum
model, classes = load_model()

## Image vandha, top 3 leaf name + confidence return pannum
def classify(image):
    if image is None:
        return {}
    results = predict(model, classes, image, topk=3)
    return {name: conf for name, conf in results}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Leaf image "),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Medicinal Leaf Recognition",
    description=", leaf name + confidence .",
)

if __name__ == "__main__":
    demo.launch(share=IS_COLAB)