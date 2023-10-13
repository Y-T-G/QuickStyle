import cv2
import numpy as np
import openvino as ov


class Stylizer:
    def __init__(self, ir_path, device):
        # Load model
        core = ov.Core()
        model = core.read_model(model=ir_path)
        self.model = compiled_model = core.compile_model(
            model=model, device_name=device
        )

        # Store the input and output nodes
        self.input_layer = compiled_model.input(0)
        self.output_layer = compiled_model.output(0)

        # Shape NHWC
        N, H, W, C = self.input_layer.shape

        self.input_shape = (N, H, W, C)

    def get_inverse_mask(self, image, masks, clip=0.1):
        # image: Original image
        # boxes: List of bounding boxes in (x, y, width, height) format
        # masks: List of binary masks

        full_mask = np.zeros(image.shape[:2]).astype(np.float32)

        # Overlay masks on bounding boxes and aggregate
        for box, mask in masks:
            # Aggregate mask
            mask = np.where(mask > 0.1, 1.0, 0.0).astype(np.float32)
            full_mask = cv2.bitwise_or(full_mask, mask)

        # Convert to 0-255
        full_mask = (1 - full_mask) * 255

        # Blur the mask with a Gaussian filter to smooth the edges
        blurred_mask = cv2.GaussianBlur(full_mask, (0, 0), 3)

        # Find contours in the blurred mask
        contours, _ = cv2.findContours(blurred_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Find the index of the largest contour
        largest_contour_index = max(range(len(contours)),
                                    key=lambda i: cv2.contourArea(contours[i]))

        # Approximate the largest contour with fewer points to make it smoother
        epsilon = 0.00112 * cv2.arcLength(contours[largest_contour_index],
                                          True)
        smoothed_contour = cv2.approxPolyDP(contours[largest_contour_index],
                                            epsilon, True)

        # Create an empty mask for drawing the smoothed contour
        smoothed_contour_mask = np.zeros_like(mask)
        cv2.drawContours(smoothed_contour_mask, [smoothed_contour], -1, (255),
                         thickness=cv2.FILLED)

        full_mask = smoothed_contour_mask[..., np.newaxis]

        return image, full_mask

    def preprocess_images(self, frame, H, W):
        """
        Preprocess input image to align with network size

        Parameters:
            :param frame:  input frame
            :param H:  height of the frame to style transfer model
            :param W:  width of the frame to style transfer model
            :returns: resized and transposed frame
        """
        image = np.array(frame).astype('float32')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(src=image, dsize=(H, W),
                           interpolation=cv2.INTER_AREA)
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        return image

    def convert_result_to_image(self, frame, stylized_image) -> np.ndarray:
        """
        Postprocess stylized image for visualization

        Parameters:
            :param frame:  input frame
            :param stylized_image:  stylized image with specific style applied
            :returns: resized stylized image for visualization
        """
        h, w = frame.shape[:2]
        stylized_image = stylized_image.squeeze().transpose(1, 2, 0)
        stylized_image = cv2.resize(src=stylized_image, dsize=(w, h),
                                    interpolation=cv2.INTER_CUBIC)
        stylized_image = np.clip(stylized_image, 0, 255).astype(np.uint8)
        stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_BGR2RGB)
        return stylized_image

    def blend_stylized(self, original_image, stylized_image, masks):
        # Get inverse mask
        inverse_mask = self.get_inverse_mask(original_image, masks)

        # Crop the stylized image using the binary mask
        cropped_stylized = cv2.bitwise_and(stylized_image, stylized_image,
                                           mask=inverse_mask)

        # Convert the binary mask to a 3-channel mask for blending
        binary_mask_3channel = cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR)

        # Invert the binary mask for blending with the original image
        inverse_mask = cv2.bitwise_not(binary_mask_3channel)

        # Blend the images
        blended_image = cv2.addWeighted(original_image, 0.7,
                                        cropped_stylized, 0.3, 0)

        # Replace the region of interest in the blended image with the cropped
        # stylized image
        result = cv2.add(blended_image, cv2.bitwise_and(cropped_stylized,
                                                        cropped_stylized,
                                                        mask=inverse_mask))

        return result

    def stylize(self, original_image, masks):
        prepped = self.preprocess_images(original_image)
        stylized_image = self.model([prepped])[self.output_layer]
        blended = self.blend_stylized(original_image, stylized_image, masks)

        return blended
