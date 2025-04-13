import streamlit as st
import cv2
import numpy as np

# Cache the loading of the YOLO model and its associated resources to avoid reloading on each rerun
@st.cache_data(allow_output_mutation=True)
@st.cache_resource
def load_yolo():
    """Load YOLO model, class labels, and output layers for object detection."""
    
    # Load class labels from 'coco.names'
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load the YOLO network with the configuration and weights files
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # Retrieve layer names and identify the output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    return net, classes, output_layers


def detect_objects(image, net, classes, output_layers):
    """Perform object detection using YOLO on the input image and return the annotated result."""
    
    height, width, _ = image.shape
    
    # Prepare the image for YOLO: convert to blob format and set as input
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Perform a forward pass to obtain detection outputs
    outputs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Iterate over detection outputs to extract bounding boxes, class IDs, and confidence scores
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Calculate bounding box dimensions
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Append detection details
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression (NMS) to remove redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    
    # Annotate the image with bounding boxes and class labels for detected objects
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            conf = confidences[i]
            color = (0, 255, 0)  # Bounding box color (green)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # Draw bounding box
            cv2.putText(image, f"{label}: {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Add label text
            
    return image

def main():
    """Streamlit application for performing object detection using YOLO."""

    # Set app title and description
    st.title("YOLO Object Detection with Streamlit")
    st.write("Upload an image, and the model will detect and label objects within it.")

    # File uploader widget to allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the uploaded image into a NumPy array and decode it with OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Display the uploaded image
        st.image(image, channels="BGR", caption="Uploaded Image")
        
        # Load YOLO model and resources (only once due to caching)
        net, classes, output_layers = load_yolo()
        
        # Perform object detection on the uploaded image
        result_image = detect_objects(image.copy(), net, classes, output_layers)
        
        # Display the result with detected objects
        st.image(result_image, channels="BGR", caption="Detection Result")

if __name__ == "__main__":
    main()
