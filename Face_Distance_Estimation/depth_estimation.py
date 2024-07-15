import cv2
import torch
import matplotlib.pyplot as plt


# Load the MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cuda')  # Move the model to the GPU
midas.eval()  # Set the model to evaluation mode


# Load the transformation functions for input preprocessing
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the preprocessing transformations and move the input to the GPU
    img_batch = transform(img).to('cuda')

    # Disable gradient computation (inference mode)
    with torch.no_grad():
        # Perform a forward pass through the model to get predictions
        predictions = midas(img_batch)

        # Upsample the predictions to the original image size using bicubic interpolation
        predictions = torch.nn.functional.interpolate(
            predictions.unsqueeze(1),   # Add a channel dimension
            size=img.shape[:2],         # Target size is the original image size
            mode='bicubic',
            align_corners=False
        ).squeeze()  # Remove the channel dimension

        # Convert the predictions to a numpy array for visualization
        output = predictions.cpu().numpy()


    # Display the depth map using matplotlib
    plt.imshow(output)
    # Display the original frame using OpenCV
    cv2.imshow('Image', frame)
    plt.pause(0.00001)  # Pause to allow the plot to update

    
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
plt.show()
