# Import the target CNN and required libraries
from tensorflow.python.keras.applications.vgg16 import (
    decode_predictions,
    preprocess_input,
    VGG16,
)
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

# Step 1: Load a clean image and convert it to numpy array:
image = load_img("acorn1.JPEG", target_size=(224, 224), interpolation="lanczos")
x = img_to_array(image)

y = 306  # Optional! Target category index number. It is only for the targeted attack.

kclassifier = VGG16(weights="imagenet")

# Step 3: Built the attack and generate adversarial image:
attackEA = EA(
    kclassifier, max_iter=10000, confidence=0.55, targeted=True
)  # if targeted is True, then confidence will be taken into account.
advers = attackEA._generate(x, y)  # returns adversarial image as .npy file. y is optional.
np.save("advers.npy", advers)
img = Image.fromarray(advers.astype(np.uint8))
img.save("advers.png", "png")