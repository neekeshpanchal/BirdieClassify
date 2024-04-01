import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import os
import cv2
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from transformers import AutoTokenizer


# Load the model and preprocessor from HuggingFace
preprocessor = EfficientNetImageProcessor.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")
model = EfficientNetForImageClassification.from_pretrained("dennisjooo/Birds-Classifier-EfficientNetB2")

# Dictionary mapping bird species to colors
bird_colors = {
    "sparrow": "brown",
    "pigeon": "gray",
    "robin": "red",
    # Add more bird species and their corresponding colors here
}

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, label, transform=None):
        self.folder_path = folder_path
        self.label = label
        self.transform = transform
        self.image_paths = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": self.label}

def classify_image(image):
    # Preprocess the input
    inputs = preprocessor(image, return_tensors="pt")

    # Running the inference
    with torch.no_grad():
        logits = model(**inputs).logits

    # Getting the predicted label and confidence score
    predicted_label_idx = logits.argmax(-1).item()
    confidence_score = torch.softmax(logits, dim=-1).max().item()
    predicted_label = model.config.id2label[predicted_label_idx]
    
    return predicted_label, confidence_score

def process_and_classify_image(image_path):
    # Open the image
    img = Image.open(image_path)

    # Classify the image
    predicted_label, confidence_score = classify_image(img)

    return predicted_label, confidence_score, img

def classify_video(video_path):
    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Read video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Classify the frame
        predicted_label, confidence_score = classify_image(img)

        # Display the classification result
        cv2.putText(frame, f"{predicted_label} ({confidence_score*100:.2f}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video Classification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def load_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        if file_path.endswith(('.jpg', '.png', '.jpeg')):
            # Process and classify image
            predicted_label, confidence_score, img = process_and_classify_image(file_path)

            # Update GUI with prediction, confidence score, and color
            color = bird_colors.get(predicted_label.lower(), "black")
            root.configure(bg=color)
            prediction_label.config(text=f"Predicted Bird Species: {predicted_label} ({confidence_score*100:.2f}%)", fg=color)

            # Convert PIL Image to a format suitable for displaying
            img.thumbnail((400, 400))
            img = ImageTk.PhotoImage(img)
            image_viewer.config(image=img)
            image_viewer.image = img

        elif file_path.endswith(('.mp4', '.avi', '.mov')):
            # Classify video
            classify_video(file_path)

def fine_tune_model():
    fine_tune_window = tk.Toplevel(root)
    fine_tune_window.title("Fine-tune Model")

    def select_folder():
        folder_path = filedialog.askdirectory()
        if folder_path:
            label = simpledialog.askstring("Label", "Enter the label for images in the folder:")
            if label:
                try:
                    fine_tune_model_in_folder(folder_path, label)
                except Exception as e:
                    messagebox.showerror("Error", str(e))
            else:
                messagebox.showwarning("Label not provided", "Please provide a label for the images.")

    def fine_tune_model_in_folder(folder_path, label):
        try:
            # Define training arguments
            training_args = TrainingArguments(
                output_dir="./output",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                evaluation_strategy="epoch",
                logging_dir="./logs",
                logging_strategy="epoch",
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

            # Load images and corresponding labels using CustomImageDataset
            dataset = CustomImageDataset(folder_path, label, transform=ToTensor())

            # Fine-tune the model
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=None,
            )
            trainer.train()
            
            messagebox.showinfo("Training Complete", "Model fine-tuning completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))


    select_folder_button = tk.Button(fine_tune_window, text="Select Folder", command=select_folder)
    select_folder_button.pack(pady=10)

# Create Tkinter window
root = tk.Tk()
root.title("Bird Species Classification")

# Create widgets
select_label = tk.Label(root, text="Select an image or video for classification", font=("Helvetica", 14))
select_label.pack(pady=10)

load_button = tk.Button(root, text="Load File", command=load_file, font=("Helvetica", 12))
load_button.pack(pady=5)

prediction_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14))
prediction_label.pack(pady=5)

image_viewer = tk.Label(root)
image_viewer.pack(pady=10)

fine_tune_button = tk.Button(root, text="Fine-tune Model", command=fine_tune_model, font=("Helvetica", 12))
fine_tune_button.pack(pady=5)

# Run Tkinter main loop
root.mainloop()
