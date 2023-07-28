import io
import os
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
from keras.saving.object_registration import CustomObjectScope
import base64
from PIL import Image

app = Flask(__name__)

# Load the U-Net model
model_path = 'C:/Users/sjkis/OneDrive/Desktop/AceNet/model.h5'
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
     model = tf.keras.models.load_model(model_path)

def preprocess_input(image):
    # Resize the input image to match the model's input size
    image = cv2.resize(image, (512, 512))
    # Normalize pixel values (if required)
    image = image / 255.0
    # Convert to numpy array and add a batch dimension
    image = np.expand_dims(image, axis=0)
    return image
def postprocess_output(predictions, threshold=0.5):
    # Apply threshold to the predictions to generate binary mask
    binary_mask = (predictions > threshold).astype(np.uint8)

    # Remove the unnecessary singleton dimensions and squeeze the array to (512, 512)
    binary_mask = np.squeeze(binary_mask, axis=(0, 3))

    # Ensure that the binary mask is of data type np.uint8
    binary_mask = binary_mask.astype(np.uint8)

    return binary_mask


app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

@app.route('/')
def index():
    message_from_python = "Python is working with HTML!"
    return render_template('index.html', message=message_from_python)


@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the uploaded image from the request
    file = request.files['file']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    processed_input = preprocess_input(image)
    predictions = model.predict(processed_input)

    postprocessed_output = postprocess_output(predictions)

    try:
        # Convert the binary image data to PIL Image format
        pil_image = Image.fromarray(postprocessed_output * 255)
        # Create an in-memory binary stream to store the image data
        image_stream = io.BytesIO()
        # Save the PIL Image in JPEG format to the in-memory stream
        pil_image.save(image_stream, format='JPEG')
        # Get the base64 encoded string of the image data
        encoded_image_str = base64.b64encode(image_stream.getvalue()).decode('utf-8')
        # Return the encoded image data as a JSON response
        return jsonify({'result': encoded_image_str}), 200
    except Exception as e:
        print("Image Encoding Error:", e)
        # Handle the error appropriately, e.g., return an error response
        return jsonify({'error': 'Image encoding failed'}), 500

if __name__ == '__main__':

     app.run(debug=True)
# from flask import Flask, request, jsonify, render_template
# import cv2
# import numpy as np
# import tensorflow as tf
# from keras.saving.object_registration import CustomObjectScope
#
# app = Flask(__name__)
#
# # Load the U-Net model
# model_path = 'model.h5'
# def iou(y_true, y_pred):
#     def f(y_true, y_pred):
#         intersection = (y_true * y_pred).sum()
#         union = y_true.sum() + y_pred.sum() - intersection
#         x = (intersection + 1e-15) / (union + 1e-15)
#         x = x.astype(np.float32)
#         return x
#     return tf.numpy_function(f, [y_true, y_pred], tf.float32)
#
# smooth = 1e-15
# def dice_coef(y_true, y_pred):
#     y_true = tf.keras.layers.Flatten()(y_true)
#     y_pred = tf.keras.layers.Flatten()(y_pred)
#     intersection = tf.reduce_sum(y_true * y_pred)
#     return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
#
# def dice_loss(y_true, y_pred):
#     return 1.0 - dice_coef(y_true, y_pred)
#
# with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
#      model = tf.keras.models.load_model(model_path)
#
#
# # ... (Rest of the model loading code and functions)
# def preprocess_input(image):
#     # Resize the input image to match the model's input size
#     image = cv2.resize(image, (512, 512))
#     # Normalize pixel values (if required)
#     image = image / 255.0
#     # Convert to numpy array and add a batch dimension
#     image = np.expand_dims(image, axis=0)
#     return image
# def postprocess_output(predictions, threshold=0.5):
#  # Apply threshold to the predictions to generate binary mask
#     binary_mask = (predictions > threshold).astype(np.uint8)
#     # If needed, resize the binary mask to match the original input size
#     # (assuming predictions are obtained from resized inputs)
#     # resized_mask = cv2.resize(binary_mask[0], (original_width, original_height))
# @app.route('/')
# def index():
#     message_from_python = "Python is working with HTML!"
#     return render_template('index.html', message=message_from_python)
#
# @app.route('/process_image', methods=['POST'])
# def process_image():
#     # Get the uploaded image from the request
#     file = request.files['file']
#     image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
#
#     processed_input = preprocess_input(image)
#     predictions = model.predict(processed_input)
#     if predictions is None:
#         return jsonify({'error': 'Model predictions are None'}), 400
#
#     postprocessed_output = postprocess_output(predictions)
#
#     if postprocessed_output is None:
#         return jsonify({'error': 'postprocess_output returned None'}), 500
#
#     # postprocessed_output = postprocess_output(predictions)
#
#     # Encode the binary mask as jpg format
#     _, img_encoded = cv2.imencode('.jpg', postprocessed_output * 255)
#
#     # Convert the result back to a response
#     return jsonify({'result': img_encoded.tobytes()}), 200
#
# if __name__ == '__main__':
#     app.run(debug=True, use_reloader=False)
