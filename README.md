# EA-based targeted adversarial attack:
This code generates adversarial images against a selected CNN trained on the ImageNet dataset, using an algorithm described in the TBA paper. To run the algorithm, you'll need to install Python 3.7 (or higher), TensorFlow 2.1, Keras 2.2, and Numpy 1.17.

To convert your image to adversarial image against a specific CNN, open the attack_EA.py file and follow the steps at the end of the file:

Step1: Load a clean image and  and resize to the target CNN's imput size. Convert it to numpy array.
Step2: Load the target CNN trained with ImageNet dataset
Step 3: Create the attack and generate adversarial image:
      - you can select here targeted or untargeted attack. If targeted attack you can decide the confidence of the target label value.


