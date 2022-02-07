# To attack a different model:
# a. import the CNN in main1.py
# b. in adaEA1.py file update the following XXXs:
# from tensorflow.keras.applications.XXX import preprocess_input
# from tensorflow.python.keras.applications.XXX import decode_predictions

from tensorflow.keras.applications.densenet import DenseNet121  # Import the target CNN
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from adaEA1 import *
model = DenseNet121(weights='imagenet')

# Settings for the EA...
numberOfElites = 10
pop_size = 40
run = 1
# Termination criteria of the EA
generation = 10000
accuracy = 0.75  # Threshold accuracy of the adversarial image target label value

# Select the original image and a target category with its index number from:
# https://github.com/aliotopal/EA-based_AdversarialAttack/blob/main/ImageNet_labels_inx.txt
original_image = 'llama2.JPEG'
target_category = 'bannister'
target_index_no = 421


# Do not touch this part: SEARCH starts here...
image = load_img(f"{original_image}", target_size=(224, 224))#, interpolation='lanczos'
ancestor = img_to_array(image)
anc = original_image.split('.')[0]
new_EA = EA(ancestor, target_index_no, model, pop_size, generation, numberOfElites, target_category, anc)
new_EA.search(pop_size, generation, run, "adaptedEA", accuracy, anc)


# The Adversarial image and a detailed report will be saved.

