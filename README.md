# EA-based targeted adversarial attack:
This code generates adversarial images against a selected CNN trained on the ImageNet dataset, using an algorithm described in the TBA paper. To run the algorithm, you'll need to install Python 3.7 (or higher), TensorFlow 2.1, Keras 2.2, and Numpy 1.17.

To customize the algorithm, open the main.py file and adjust the following settings: number of population, number of elites, maximum number of generations, minimum target label value, and the target CNN. You can also select the original image and target category in main.py.

Once you've made the necessary changes to the settings, run main.py to generate the adversarial image. The resulting image will be saved in .png format, and a detailed report containing information about the execution time, number of generations, and target label value will be saved in a .txt file.
