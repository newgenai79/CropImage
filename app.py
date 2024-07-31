import gradio as gr
import cv2
import numpy as np
import os
from skimage import transform as tf
import dlib

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Define the function for cropping the image
def crop_image(image_path, output_directory, output_filename):
    template = np.load('data_preprocess/M003_template.npy')
    image = cv2.imread(image_path)
    
    # Ensure the image is in RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    if len(rects) != 1:
        return "Error: Could not detect a single face in the image."

    shape = predictor(gray, rects[0])
    shape = shape_to_np(shape)

    pts2 = np.float32(template[:47, :])
    pts1 = np.float32(shape[:47, :])
    tform = tf.SimilarityTransform()
    tform.estimate(pts2, pts1)
    dst = tf.warp(image, tform, output_shape=(256, 256))
    dst = (dst * 255).astype(np.uint8)

    # Convert back to BGR for OpenCV compatibility
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

    output_path = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_path, dst)
    return output_path

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Function to update output directory and filename based on input image
def update_output_fields(input_image_path):
    if not input_image_path:
        return "", ""
    output_directory = os.path.dirname(input_image_path)
    output_filename = os.path.splitext(os.path.basename(input_image_path))[0] + "-cropped.png"
    return output_directory, output_filename

# Gradio interface setup
with gr.Blocks() as iface:
    gr.Markdown("# Prepare Image for Inference")
    
    input_image = gr.Image(type="filepath", label="Input Image (.png only)")
    output_directory = gr.Textbox(label="Output Directory")
    output_filename = gr.Textbox(label="Output Filename")

    crop_button = gr.Button("Crop Image")
    output_result = gr.Textbox(label="Output Path")

    # Connect input changes to dynamic updates
    input_image.change(
        fn=update_output_fields, 
        inputs=input_image, 
        outputs=[output_directory, output_filename]
    )

    # Define button action
    crop_button.click(
        fn=crop_image, 
        inputs=[input_image, output_directory, output_filename], 
        outputs=output_result
    )

iface.launch()
