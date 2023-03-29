# To attack a different model:
# a. import the CNN in main.py
# b. in adaEA.py file update the following XXXs:
# from tensorflow.keras.applications.XXX import preprocess_input
# from tensorflow.python.keras.applications.XXX import decode_predictions

from tensorflow.keras.applications.mobilenet import MobileNet  # Import the target CNN
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from adaEA import *
model = MobileNet(weights='imagenet')

# Settings for the EA...
numberOfElites = 10
pop_size = 40
# Termination criteria of the EA
generation = 10000
accuracy = 0.75  # Threshold accuracy of the adversarial image target label value

# Select the original image and a target category with its index number from:
# https://github.com/aliotopal/EAbasedAttack/blob/master/ImageNet_labels_indx.txt
original_image = 'acorn1.JPEG'
target_category = 'rhinoceros beetle'
target_index_no = 306


# Do not touch this part: SEARCH starts here...
image = load_img(f"{original_image}", target_size=(224, 224))#, interpolation='lanczos'
ancestor = img_to_array(image)
anc = original_image.split('.')[0]
new_EA = EA(ancestor, target_index_no, model, pop_size, generation, numberOfElites, target_category, anc)
new_EA.search(pop_size, generation,"adaptedEA", accuracy, anc)


# The Adversarial image and a detailed report will be saved.

