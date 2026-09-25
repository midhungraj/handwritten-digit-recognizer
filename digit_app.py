from math import log
import tkinter as tk
from PIL import ImageGrab
import torch
from torchvision import transforms

from model import DigitCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
model = DigitCNN()

model.load_state_dict(torch.load("digit_model.pth", map_location=device))

model.eval()

# Preprocessing
transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

# canvas
WIDTH = 400
HEIGHT = 400

root = tk.Tk()
root.title("Digit Recognizer")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")

canvas.pack()


# Draw
def draw(event):
    x, y = event.x, event.y

    r = 10

    canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")


# Predict
def predict():
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()

    image = ImageGrab.grab(
        bbox=(x, y, x + canvas.winfo_width(), y + canvas.winfo_height())
    )

    print("Image:", image)
    print("Original size:", image.size)

    # Transforming
    image = transform(image)

    print("After transform:", image.shape)

    image = image.unsqueeze(0)

    print("Model input:", image.shape)

    # Pytorch inference
    with torch.no_grad():
        logits = model(image)

        probabilities = torch.softmax(logits, dim=1)

        prediction = probabilities.argmax(dim=1).item()

        confidence = probabilities[0, prediction].item()

    print(f"Prediction: {prediction}")

    print(f"Confidence: {confidence:.2%}")


# Clear canvas
def clear():
    canvas.delete("all")


# Draw while mouse moves
canvas.bind("<B1-Motion>", draw)

# predict when mouse button released
# canvas.bind("<ButtonRelease-1>", predict)

button_frame = tk.Frame(root, bg="#222222")

button_frame.pack(pady=10)

# Predict button
predict_button = tk.Button(
    button_frame, text="Predict", command=predict, bg="#333333", fg="white"
)

predict_button.pack(side="left", pady=5)

# Clear button
clear_button = tk.Button(
    button_frame, text="Clear", command=clear, bg="#333333", fg="white"
)

clear_button.pack(side="left", pady=5)


root.mainloop()
