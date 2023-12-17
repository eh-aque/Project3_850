# Import necessary libraries
import cv2
import numpy as np
import torch
from PIL import Image

# Step 1: Object Masking
# ----------------------------------------------
# Load the image
motherboard_img = cv2.imread('motherboard_image.JPEG', cv2.IMREAD_COLOR)

# Convert the image to grayscale
motherboard_img_gray = cv2.cvtColor(motherboard_img, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur
motherboard_img_gray = cv2.GaussianBlur(motherboard_img_gray, (45, 45), 4)

# Apply adaptive thresholding
motherboard_img_gray = cv2.adaptiveThreshold(motherboard_img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 5)

# Detect edges
edges = cv2.Canny(motherboard_img_gray, 1, 1)
edges = cv2.dilate(edges,None, iterations = 7)

cv2.imwrite('motherboard_edge.jpeg', edges)

# Find contours
contours, _ = cv2.findContours(image=edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

# Create an empty image to draw the contours
contour_img = np.zeros_like(motherboard_img)

# Draw the largest contour on the image
cv2.drawContours(image=contour_img, contours=[max(contours, key = cv2.contourArea)], contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED)

cv2.imwrite('motherboard_contour.jpeg', contour_img)

# Bitwise-AND contour image and original image
masked_img = cv2.bitwise_and(contour_img,  motherboard_img)

# Save the result
cv2.imwrite('motherboard_output.jpeg', masked_img)


# ----------------------------------------------



# # Step 2: YOLOv8 Training
# # ----------------------------------------------
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = torch.hub.load('ultralytics/yolov8', 'yolov8_nano')

# # Train the model
# model.train('./Desktop/TMU Y4/S1/AER850 - Intro to Machine Learning/Project 3/data/train', epochs=200, batch_size=16, imgsz=900, name='my_model')

# # Save the model
# torch.save(model.state_dict(), 'Desktop/TMU Y4/S1/AER850 - Intro to Machine Learning/Project 3/my_model.pth')

# # # Evaluate the model
# # for image_path in ['image1.png', 'image2.png', 'image3.png']:
# #     img = Image.open(image_path)
# #     results = model.predict(img)
# #     results.print()  # print results to stdout
# #     results.show()  # display results