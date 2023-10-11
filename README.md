# EA-based targeted adversarial attack:
This code generates adversarial images against a selected CNN trained on the ImageNet dataset, using an algorithm described in the TBA paper. To run the algorithm, you'll need to install Python 3.7 (or higher), TensorFlow 2.1, Keras 2.2, and Numpy 1.17.

To convert a clean image to an adversarial image against a specific CNN, open the attack_EA.py file and follow the steps at the end of the file:
Open the attack_EA.py file and go to the end of the code "# SET UP your attack" to set up your attack:

Step 1: Load the target CNN trained with the ImageNet dataset

Step 2: Load a clean image and set up attack parameters.
You can select here a targeted or untargeted attack. If targeted attack you can decide the confidence of the target label value.


