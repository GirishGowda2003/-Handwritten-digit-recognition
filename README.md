# -Handwritten-digit-recognition
A deep learning application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN).

# ✍️ Handwritten Digit Recognition — End-to-End Deep Learning Project

**A beginner-friendly, step-by-step image classification project**

In this notebook we will build a deep learning model that recognizes handwritten digits from **0 to 9** based on handwritten digit images. The model will learn to classify each input image into one of the **10 digit classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9**.

We will go through the **complete deep learning workflow**:

1. Load & explore the raw handwritten digit dataset
2. Basic checks and clean the dataset
3. Analyze the number of images available for each digit class
4. Preprocess the images by resizing and normalizing pixel values
5. Visualize sample images from digits **0–9**
6. Handle missing, duplicate, or corrupted images if present
7. Split the dataset into **training, validation, and testing** sets
8. Build a **CNN-based image classification model**
9. Train the CNN model and monitor training/validation accuracy and loss
10. Apply **Data Augmentation** to improve model generalization
11. Tune hyperparameters such as **learning rate, batch size, number of epochs, dropout, and optimizer**
12. Evaluate the model using **accuracy, precision, recall, F1-score, confusion matrix, and classification report**
13. Test the model by providing new handwritten digit images
14. After running all the models, compare their performance
15. Choose the model with the **best validation/test performance** and select it as the final model
16. Save the final trained model as a `.h5` or `.keras` file
17. Reload the saved model and make a prediction on a new handwritten digit image to prove it works
18. Display the predicted digit and prediction confidence
19. Build a UI using **Streamlit** where users can upload a handwritten digit image and get the predicted digit **(optional)**

> 💡 Every step includes comments explaining **why** we do it, not just how — this notebook is written for someone new to deep learning and image classification.
