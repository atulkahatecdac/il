import torch
import urllib.request
from PIL import Image
from torchvision import transforms

print("=====================================================")
print("    HUB 2: PYTORCH (LIVE IMAGE CLASSIFICATION)       ")
print("=====================================================\n")

print("[1] Downloading ResNet-18 Vision Model...")
# Download the pre-trained model directly from the PyTorch Hub
vision_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
vision_model.eval() # Set to inference mode

print("\n[2] Fetching a test image (A Golden Retriever)...")
# Download a sample image of a dog to test the model
image_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
image_filename = "dog.jpg"
urllib.request.urlretrieve(image_url, image_filename)
input_image = Image.open(image_filename)
print("Image loaded successfully.")

print("\n[3] Preprocessing the image (Converting pixels to math)...")
# Neural networks require very strict input formatting
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
# The model expects a "batch" of images, so we create a batch of 1
input_batch = input_tensor.unsqueeze(0) 

print("\n[4] Running Inference (The Model is 'Thinking')...")
with torch.no_grad(): # Tell PyTorch we aren't training, just predicting
    output = vision_model(input_batch)

print("\n[5] Translating Mathematical Output to Human Labels...")
# Download the dictionary of 1,000 object categories so we can read the output
urllib.request.urlretrieve("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Calculate the probabilities for the top 5 predictions
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

print("\n--- FINAL MODEL PREDICTIONS ---")
for i in range(top5_prob.size(0)):
    category_name = categories[top5_catid[i]]
    confidence = top5_prob[i].item() * 100
    print(f"{category_name}: {confidence:.2f}% confidence")
print("=====================================================")