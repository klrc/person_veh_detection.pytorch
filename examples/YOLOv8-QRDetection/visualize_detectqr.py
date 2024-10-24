import cv2
import numpy as np


# Function to read annotations from the txt file
def read_annotations(txt_file, img_width, img_height):
    with open(txt_file, "r") as file:
        annotations = []
        for line in file:
            values = line.strip().split()
            cls = int(values[0])  # The class label
            points = np.array(values[1:], dtype=float).reshape(4, 2)  # Reshape to 4 points (x1, y1, ..., x4, y4)

            # Denormalize the coordinates by multiplying with image dimensions
            points[:, 0] *= img_width  # x coordinates scaled by width
            points[:, 1] *= img_height  # y coordinates scaled by height

            annotations.append((cls, points))
    return annotations


# Function to draw the polygons on the image
def draw_polygons(image, annotations):
    for annotation in annotations:
        cls, points = annotation
        points = points.astype(int)  # Convert points to integers for drawing
        color = (0, 255, 0)  # Green color for the bounding box
        # Draw the polygon using the points
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
    return image


# Main function to load image, read annotations, and draw
def visualize_bounding_boxes(image_path, txt_path):
    # Load the image
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]  # Get image dimensions

    # Read annotations from the txt file with denormalized coordinates
    annotations = read_annotations(txt_path, img_width, img_height)

    # Draw bounding boxes on the image
    image_with_boxes = draw_polygons(image, annotations)

    # Show the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
# image_path = "/nfs/datasets/DOTAv1/images/val/P0003.jpg"
# txt_path = "/nfs/datasets/DOTAv1/labels/val/P0003.txt"
image_path = "/nfs/datasets/DetectQR-v1i/valid/images/0b03f1591fb8ce6b6bbd55d9530fb700_jpg.rf.c3a49b0edcf575327cfd52bd237403b0.jpg"
txt_path = "/nfs/datasets/DetectQR-v1i/valid/labels/0b03f1591fb8ce6b6bbd55d9530fb700_jpg.rf.c3a49b0edcf575327cfd52bd237403b0.txt"
visualize_bounding_boxes(image_path, txt_path)
