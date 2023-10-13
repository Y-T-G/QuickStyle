import cv2
import numpy as np
import openvino as ov


class Segmenter:
    def __init__(self, ir_path, device):
        # Load model
        core = ov.Core()
        model = core.read_model(model=ir_path)
        self.model = compiled_model = core.compile_model(
            model=model, device_name=device
        )

        # Store the input and output nodes
        self.input_layer = compiled_model.input(0)
        self.boxes_layer = compiled_model.output(0)
        self.labels_layer = compiled_model.output(1)
        self.masks_layer = compiled_model.output(2)

        # Shape
        N, C, H, W = self.input_layer.shape
        self.input_shape = (N, C, H, W)

        # COCO labels
        self.labels = open("labels.txt", "r").read().splitlines()
        self.label_map = {
            self.labels[lbl_id]: lbl_id for lbl_id in range(len(self.labels))
        }

        # Color Palleter
        self.palette = {label: np.random.randint(0, 256, 3)
                        for label in range(80)}

    def scale_boxes(self, original_size, resized_size, boxes):
        # original_size: (original_height, original_width) resized_size:
        # (resized_height, resized_width)
        # boxes: List of bounding boxes in the
        # format (x, y, width, height, score)

        # Compute scaling factors for width and height
        width_scale = original_size[1] / resized_size[1]
        height_scale = original_size[0] / resized_size[0]

        # Scale bounding boxes
        scaled_boxes = [
            (
                int(box[0] * width_scale),  # x coordinate
                int(box[1] * height_scale),  # y coordinate
                int(box[2] * width_scale),  # width
                int(box[3] * height_scale),  # height
                box[4],  # score
            )
            for box in boxes
        ]

        return scaled_boxes

    def postprocess(self, result, ori_shape, res_shape):
        output = dict()

        boxes = result[self.boxes_layer]

        # Scale and convert boxes to (x, y, width, height) format
        boxes = self.scale_boxes(ori_shape[:2], res_shape[:2], boxes)
        boxes = np.array(
            [
                (box[0], box[1], box[2] - box[0], box[3] - box[1], box[4])
                for box in boxes
            ]
        )

        # Apply NMS to the filtered boxes
        indices_after_nms = cv2.dnn.NMSBoxes(boxes[:, :4], boxes[:, 4],
                                             0.5, 0.3)

        # Filter
        output["boxes"] = boxes[indices_after_nms]
        output["labels"] = result[self.labels_layer][indices_after_nms]
        masks = result[self.masks_layer][indices_after_nms]

        # Scale masks
        output["masks"] = self.heatmap_to_mask(ori_shape, output["boxes"],
                                               masks)

        return output

    def heatmap_to_mask(self, img_shape, boxes, masks):
        scaled_masks = list()

        for box, mask in zip(boxes, masks):
            # Scale each mask to image
            full_mask = np.zeros(img_shape[:2]).astype(np.float32)

            x, y, w, h, _ = box.astype(np.int16)

            # Resize mask to match bounding box size
            mask = mask.astype(np.float32)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Add mask
            full_mask[y:y+h, x:x+w] = cv2.bitwise_or(
                full_mask[y:y+h, x:x+w], mask
            )

            scaled_masks.append(full_mask)

        return scaled_masks

    def overlay_labels(self, image, boxes, classes, scores):
        labels = (self.labels[class_id] for class_id in classes)

        template = "{}: {:.2f}"

        for box, score, label in zip(boxes, scores, labels):
            text = template.format(label, score)
            # textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
            #                            0.5, 1)[0]
            position = (box[:2]).astype(np.int32)
            cv2.putText(
                image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1
            )
        return image

    def overlay_masks(self, image, masks, ids=None):
        """
        https://github.com/openvinotoolkit/open_model_zoo/demos/common/
        python/visualizers/instance_segmentation.py#L42
        """
        segments_image = image.copy()
        aggregated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        aggregated_colored_mask = np.zeros(image.shape, dtype=np.uint8)
        all_contours = []

        for i, mask in enumerate(masks):
            if mask.dtype == np.float32:
                mask = (mask * 255).astype(np.uint8)
            contours = cv2.findContours(mask, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)[
                -2
            ]
            if contours:
                all_contours.append(contours[0])

            mask_color = self.palette[i if ids is None else ids[i]]
            cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
            cv2.bitwise_or(
                aggregated_colored_mask,
                mask_color,
                dst=aggregated_colored_mask,
                mask=mask,
            )

        # Fill the area occupied by all instances with a colored instances mask
        # image
        cv2.bitwise_and(
            segments_image, (0, 0, 0), dst=segments_image, mask=aggregated_mask
        )
        cv2.bitwise_or(
            segments_image,
            aggregated_colored_mask,
            dst=segments_image,
            mask=aggregated_mask,
        )

        cv2.addWeighted(image, 0.5, segments_image, 0.5, 0, dst=image)
        cv2.drawContours(image, all_contours, -1, (0, 0, 0))
        return image

    def segment(self, image):
        # Resize
        H, W = self.input_shape[2:]
        res_img = cv2.resize(src=image, dsize=(W, H),
                             interpolation=cv2.INTER_AREA)
        # HWC to CHW
        prep_img = res_img.transpose(2, 0, 1)
        # NCHW
        prep_img = prep_img[None, ...]

        # Infer
        result = self.model([ov.Tensor(prep_img.astype(np.float32))])

        # Postprocess
        output = self.postprocess(result, image.shape, res_img.shape)

        return output
