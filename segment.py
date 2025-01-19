# Installation


# Imports
import supervision as sv
import gradio as gr
import os
import torch
import sys
from matplotlib import pyplot as plt
import numpy as np
from segment_anything import SamPredictor
from segment_anything import sam_model_registry, SamPredictor
import cv2
print(sv.__version__)

"""## Intalling Grounding DINO & Segment Anything Model"""

HOME = os.getcwd()
print("HOME:", HOME)

"""### Grouding DINO Setup"""

# Commented out IPython magic to ensure Python compatibility.
# Grounding DINO
# %cd {HOME}

GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))

GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))

"""#### Load Model

"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}/GroundingDINO

from groundingdino.util.inference import Model
from groundingdino.util.inference import load_model, load_image, predict, annotate

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
# grounding_dino_model = load_model(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH)

"""### Segment Anything Model Setup

"""

SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))

"""#### Load Model"""

SAM_ENCODER_VERSION = "vit_h"


sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

"""# Gradio Interface"""




def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def forward(input_img, text_prompt="shirt", box_threshold=0.35, text_threshold=0.25):
    # image = cv2.imread("/content/test.jpeg")
    image = input_img

    # Prompting Grounding DINO
    detections, labels = grounding_dino_model.predict_with_caption(
      image=image,
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
    )

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)


    print(detections.mask)



    # return detections.mask
    return annotated_image, detections.mask

def test_interface():
    demo = gr.Interface(forward, inputs=[gr.Image(), gr.Textbox(), gr.Slider(value=0.35, minimum=0, maximum=1, step=0.01), gr.Slider(value=0.25, minimum=0, maximum=1, step=0.01)], outputs= ["image"])

    image = cv2.imread("/content/test.jpg")
    # image = input_img
    text_prompt = "a person"
    box_threshold = 0.35
    text_threshold = 0.25
    # Prompting Grounding DINO
    detections, labels = grounding_dino_model.predict_with_caption(
    image=image,
    caption=text_prompt,
    box_threshold=box_threshold,
    text_threshold=text_threshold
    )

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Commented out IPython magic to ensure Python compatibility.

    # %matplotlib inline
    sv.plot_image(annotated_image, (16, 16))

    demo.launch()


def register_hooks():
    # Load SAM model and predictor
    sam_model = sam_model_registry['vit_h'](checkpoint='/content/GroundingDINO/weights/sam_vit_h_4b8939.pth')
    sam_predictor = SamPredictor(sam_model)

    # Define a dictionary to store feature maps
    feature_maps = {}

    # Hook function to capture feature maps
    def hook_fn(module, input, output):
        feature_maps[module] = output

    # Register hooks on specific layers (e.g., last layer of ViT blocks)
    for name, layer in sam.image_encoder.blocks.named_children():
        if int(name) == len(sam.image_encoder.blocks) - 1:  # Example: last block
            layer.register_forward_hook(hook_fn)

    return feature_maps

def overlay_attention_maps(feature_maps):

    feature_map = sum(list(feature_maps.values()))
    # [0][0]  # Get first batch example

    # Resize the feature map to match the input image size
    input_image = image
    h, w, _ = input_image.shape
    feature_map_resized = torch.nn.functional.interpolate(
        feature_map.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    )[0].cpu().detach().numpy()

    # Normalize feature map and apply a colormap
    feature_map_resized = np.mean(feature_map_resized, axis=0)  # Average over channels if needed
    feature_map_normalized = cv2.normalize(feature_map_resized, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(feature_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay heatmap on the original image with some transparency
    overlay = cv2.addWeighted(input_image, 0.6, heatmap, 0.4, 0)

    # Plot original image and overlayed heatmap
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(input_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Feature Map Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()


def launch_interface():

    """# Interface Design"""

    css = """
    h1 {
        text-align: center;
        display:block;
    }
    """

    js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'light') {
            url.searchParams.set('__theme', 'light');
            window.location.href = url.href;
        }
    }
    """

    SERVER_PORT = 7860
    my_theme = gr.Theme.from_hub("shivi/calm_seafoam")

    with gr.Blocks(title="Explainable Visual AI Interface", js=js_func, css=css, theme=my_theme).queue() as demo:

            gr.Markdown("""# Explainable Visual AI Interface with Segment Anything Model & Grounding DINO
            """)

            with gr.Row():
                # UI for Classification task
                with gr.Tab(label="Semantic Segmentation"):

                    with gr.Row():

                        # Image Section
                        with gr.Column():

                            # Image Input
                            with gr.Row():
                                seg_img = gr.Image(label="Input image", show_label=False)

                            with gr.Row():
                                seg_tsk = gr.Textbox(label="Prompt", visible=True)

                            # Text Input
                            # More Controls
                            with gr.Row():
                                seg_slide_box = gr.Slider(value=0.35, minimum=0, maximum=1, step=0.01, label="Segmentation Strength")
                                seg_slide_text = gr.Slider(value=0.25, minimum=0, maximum=1, step=0.01, label="Textual Strength")

                            # Run Segmentation
                            with gr.Row():
                                seg_pred_btn = gr.Button("Run Segmentation")

                        # Segmentation Section
                        with gr.Column():
                            # clf_preds = gr.Label(label="Top-3 Predictions", show_label=False)

                            # Segmented Image
                            with gr.Row():
                                seg_out = gr.Image(label="Segmented Output")

                            # Gallery
                            with gr.Row():
                                seg_gallery = gr.outputs.Gallery(label="Segmentations")

                            # Run Explainable AI
                            with gr.Row():
                                seg_xai = gr.Button("Run Explainable AI")

                        # Explanations
                        with gr.Column():
                            clf_xai_img = gr.Image(type="pil", label="Explanation Image", show_label=False)
                            seg_textual_button = gr.Button("Textual Explanation", variant="primary")
                        with gr.Column():
                            seg_textual = gr.Text(label="Textual Explanation", show_label=False)

            # Run prediction for segmentation
            seg_pred_btn.click(fn=forward,
                            inputs=[seg_img, seg_tsk, seg_slide_box, seg_slide_text],
                            outputs=[seg_out, seg_gallery])

            seg_xai.click(fn=forward,
                            inputs=[seg_img, seg_tsk, seg_slide_box, seg_slide_text],
                            outputs=[seg_out, seg_gallery])

            seg_textual_button.click(fn=forward, inputs=[seg_img, seg_tsk, seg_slide_box, seg_slide_text], outputs=[seg_textual])

            demo.launch()



if __name__ == "__main__":
    launch_interface()