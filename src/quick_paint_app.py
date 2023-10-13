import os

import numpy as np
import gradio as gr
import openvino as ov

from .stylizer import Stylizer
from .segmenter import Segmenter

from ..utils.utils import download_file


class QuickPaintApp:
    def __init__(self, seg_ir_path, device="CPU"):
        self.segmenter = Segmenter(seg_ir_path, device)
        self.device = device
        self.stylizer = None
        self.app = gr.Blocks()

    def init_stylizer(self, style, base_model_dir="model"):
        sty_ir_path = f"{base_model_dir}/{style.lower()}-9.xml"

        # Download and convert
        if not os.path.exists(sty_ir_path):
            # Download
            base_url = "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model"
            model_path = f"{style.lower()}-9.onnx"
            style_url = f"{base_url}/{model_path}"
            download_file(style_url, directory=base_model_dir)

            # Convert
            ov_model = ov.convert_model(f"{base_model_dir}/{model_path}")
            ov.save_model(ov_model, sty_ir_path)

        self.stylizer = Stylizer(sty_ir_path, self.device)

    # Function to apply inpainting based on selected checkboxes
    def stylize_selected_objects(self, checkboxes, style, masks, class_ids,
                                 image_np):
        if self.stylizer is None:
            self.init_stylizer(style)

        selected_labels = [selected for selected in checkboxes if checkboxes]
        selected_ids = [self.segmenter.label_map[label]
                        for label in selected_labels]
        filtered_masks = [
            mask for mask, cls_id in zip(masks, class_ids)
            if cls_id in selected_ids
        ]

        # Apply inpainting
        result = self.stylizer.stylize(image_np, filtered_masks)

        return result

    # Gradio app configuration
    def segment(self, input_image):
        # Convert input_image to numpy array
        image_np = np.array(input_image)

        # Instance segmentation
        seg_out = self.segmenter.segment(image_np)
        boxes, masks, class_ids = (seg_out["boxes"], seg_out["masks"],
                                   seg_out["labels"])

        overlayed_img = self.segmenter.overlay_masks(image_np.copy(), masks)
        overlayed_img = self.segmenter.overlay_labels(
            overlayed_img, boxes[:, :4], class_ids, boxes[:, 4]
        )

        # Create checkboxes for each detected class
        detected_objects = [
            self.segmenter.labels[cls_id] for cls_id in np.unique(class_ids)
        ]
        checkboxes = gr.CheckboxGroup(
            label="Select objects to remove",
            choices=detected_objects,
            visible=True,
            interactive=True,
        )

        # Show styles:
        style = gr.Radio(label="Select a style",
                         choices=['Mosaic', 'Rain-Princess',
                                  'Candy', 'Udnie', 'Pointilism'],
                         visible=True)

        # Show inpaint button
        stylize_button = gr.Button(value="Stylize", visible=True)

        return (overlayed_img, checkboxes, style, stylize_button, boxes, masks,
                class_ids)

    def build(self):
        with self.app:
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image()
                    seg_btn = gr.Button(value="Segment")
                with gr.Column():
                    mask_img = gr.Image(label="Mask")
                    boxes = gr.State()
                    masks = gr.State()
                    class_ids = gr.State()
            with gr.Row():
                with gr.Column():
                    checkboxes = gr.CheckboxGroup(visible=False)
                    stylize_button = gr.Button(visible=False)
                with gr.Column():
                    style = gr.Radio(visible=False)
            with gr.Row():
                stylize_canvas = gr.Image(label="Stylized", visible=True)

            # Run segmentation on input image
            seg_btn.click(
                self.segment,
                inputs=[input_img],
                outputs=[mask_img, checkboxes, style, stylize_button, boxes,
                         masks, class_ids],
                show_progress=True,
            )

            # # Inpaint button
            # inpaint_canvas = gr.Image(label="Inpainted", visible=False)
            stylize_button.click(
                self.stylize_selected_objects,
                inputs=[checkboxes, style, masks, class_ids, input_img],
                outputs=[stylize_canvas],
                show_progress=True,
            )

    def launch(self):
        self.build()
        # Run Gradio app
        self.app.launch()

    def shutdown(self):
        self.app.close()
