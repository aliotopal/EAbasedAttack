from typing import Optional

import numpy as np
import sys
import random

random.seed(0)


# Do NOT touch the class.
class EA:
    def __init__(self, klassifier, max_iter, confidence, targeted):
        self.klassifier = klassifier
        self.max_iter = max_iter
        self.confidence = confidence
        self.targeted = targeted
        self.pop_size = 40
        self.numberOfElites = 10

    @staticmethod
    def _get_class_prob(preds: np.ndarray, class_no: np.array) -> np.ndarray:
        """
        :param preds: an array of predictions of individuals for all the categories: (40, 1000) shaped array
        :param class_no: for the targeted attack target category index number; for the untargeted attack ancestor
        category index number
        :return: an array of the prediction of individuals only for the target/ancestor category: (40,) shaped  array
        """
        return preds[:, class_no]

    @staticmethod
    def _get_fitness(probs: np.ndarray) -> np.ndarray:
        """
         It simply returns the CNN's probability for the images but different objective functions can be used here.
        :param probs: an array of images' probabilities of selected CNN
        :return: returns images' probabilities in an array (40,)
        """
        fitness = probs
        return fitness

    def _selection_untargeted(self, images: np.ndarray, fitness: np.ndarray):
        """
        Population will be divided into elite, middle_class, and didn't make it based on
        images (individuals) fitness values. The images furthest from the ancestor category will be
        closer to be in the elite.
        :param images: the population of images in an array: size (pop_size, 224, 224, 3)
        :param fitness: an array of images' propabilities of selected CNN
        :return: returns a tuple of elite, middle_class images, fitness values of elites, index number of elites
                in the population array, and random_keep images as numpy arrays.
        """
        idx_elite = fitness.argsort()[: self.numberOfElites]
        # print("IDX ELITE:", idx_elite)
        half_pop_size = images.shape[0] / 2
        idx_middle_class = fitness.argsort()[self.numberOfElites : int(half_pop_size)]
        elite = images[idx_elite, :]
        middle_class = images[idx_middle_class, :]

        possible_idx = set(range(0, images.shape[0])) - set(idx_elite)
        idx_keep = random.sample(possible_idx, int(images.shape[0] / 2 - self.numberOfElites))
        random_keep = images[idx_keep]
        return elite, middle_class, random_keep

    def _selection_targeted(self, images: np.ndarray, fitness: np.ndarray):
        """
        Population will be divided into elite, middle_class, and didn't make it based on
        images (individuals) fitness values. The images closest to the target category will be
        closer to be in the elite.
        :param images: the population of images in an array: size (pop_size, 224, 224, 3)
        :param fitness: an array of images' probabilities of selected CNN
        :return: returns elite, middle_class images, fitness values of elites, index number of elites
                in the population array, and random_keep images as numpy arrays.
        """
        idx_elite = fitness.argsort()[-self.numberOfElites :]
        half_pop_size = images.shape[0] / 2
        idx_middle_class = fitness.argsort()[int(half_pop_size) : -self.numberOfElites]
        elite = images[idx_elite, :][::-1]
        middle_class = images[idx_middle_class, :]

        possible_idx = set(range(0, images.shape[0])) - set(idx_elite)
        idx_keep = random.sample(possible_idx, int(images.shape[0] / 2 - self.numberOfElites))
        random_keep = images[idx_keep]
        return elite, middle_class, random_keep

    @staticmethod
    def _get_no_of_pixels(im_size: int) -> int:
        """
        :param im_size: Original inputs' size, represented by an integer value.
        :return: returns an integer that will be used to decide how many pixels will be mutated
        in the image during the current generation.
        """
        u_factor = np.random.uniform(0.0, 1.0)
        n = 60  # normally 60, the smaller n -> more pixels to mutate
        res = (u_factor ** (1.0 / (n + 1))) * im_size
        no_of_pixels = im_size - res
        return no_of_pixels

    @staticmethod
    def _mutation(
        _x: np.ndarray,
        no_of_pixels: int,
        mutation_group: np.ndarray,
        percentage: float,
        boundary_min: int,
        boundary_max: int,
    ) -> np.ndarray:
        """
        :param _x: An array with the original input to be attacked.
        :param no_of_pixels: An integer determines the number of pixels to mutate in the original input for the current
            generation.
        :param mutation_group: An array with the individuals which will be mutated
        :param percentage: A decimal number from 0 to 1 that represents the percentage of individuals in the mutation
            group that will undergo mutation.
        :param boundary_min: keep the pixel within [0, 255]
        :param boundary_max: keep the pixel within [0, 255]
        :return: An array of mutated individuals
        """
        mutated_group = mutation_group.copy()
        # np.random.shuffle(mutated_group)
        no_of_individuals = len(mutated_group)  # 20 individuals
        for individual in range(int(no_of_individuals * percentage)):
            locations_x = np.random.randint(x.shape[0], size=int(no_of_pixels))
            locations_y = np.random.randint(x.shape[1], size=int(no_of_pixels))
            locations_z = np.random.randint(x.shape[2], size=int(no_of_pixels))
            new_values: [int] = random.choices(np.array([-1, 1]), k=int(no_of_pixels))
            mutated_group[individual, locations_x, locations_y, locations_z] = (
                mutated_group[individual, locations_x, locations_y, locations_z] - new_values
            )
            # Epsilon restrictions ####################
            noise = mutated_group[individual] - _x
            noise = np.clip(noise, -epsilon, epsilon)
            mutated_group[individual] = _x + noise
            ###########################################
        mutated_group = np.clip(mutated_group, boundary_min, boundary_max)
        # mutated_group = mutated_group % 200
        return mutated_group


    @staticmethod
    def _get_crossover_parents(crossover_group: np.ndarray) -> list:
        size = crossover_group.shape[0]  # size = 30
        no_of_parents = random.randrange(0, size, 2)  # gives random even number between 0 and size.
        parents_idx = random.sample(range(0, size), no_of_parents)
        return parents_idx  # returns parents indexs who will be used for corssover.


    @staticmethod
    def _crossover(_x: np.ndarray, crossover_group: np.ndarray, parents_idx: list) -> np.ndarray:
        crossedover_group = crossover_group.copy()
        for i in range(0, len(parents_idx), 2):
            parent_index_1 = parents_idx[i]
            parent_index_2 = parents_idx[i + 1]
            crossover_range = int(x.shape[0] * 0.15)  # 15% of the image will be crossovered.
            size_x = np.random.randint(0, crossover_range)
            start_x = np.random.randint(0, _x.shape[0] - size_x)
            size_y = np.random.randint(0, crossover_range)
            start_y = np.random.randint(0, _x.shape[1] - size_y)
            z = np.random.randint(_x.shape[2])
            temp = crossedover_group[
                   parent_index_1,
                   start_x: start_x + size_x,
                   start_y: start_y + size_y,
                   z,
                   ]
            crossedover_group[
            parent_index_1, start_x: start_x + size_x, start_y: start_y + size_y, z
            ] = crossedover_group[parent_index_2, start_x: start_x + size_x, start_y: start_y + size_y, z]
            crossedover_group[parent_index_2, start_x: start_x + size_x, start_y: start_y + size_y, z] = temp
        return crossedover_group

    def _generate(self, x: np.ndarray, y: Optional[int] = None) -> np.ndarray:
        """
        :param x: An array with the original inputs to be attacked.
        :param y: An integer with the true or target labels.
        :return: An array holding the adversarial examples.
        """
        boundary_min = 0
        boundary_max = 255

        img = x.reshape((1, x.shape[0], x.shape[1], x.shape[2])).copy()
        img = preprocess_input(img)
        preds = self.klassifier.predict(img)
        label0 = decode_predictions(preds)
        label1 = label0[0][0]  # Gets the Top1 label and values for reporting.
        ancestor = label1[1]  # label
        anc_indx = np.argmax(preds)
        print("Before the image is:  " + ancestor + " --> " + str(label1[2]) + " ____ index: " + str(anc_indx))
        if self.targeted:
            print("Target class index number is: ", y)
        images = np.array([x] * self.pop_size).astype(int)  # pop_size * ancestor images are created
        count = 0
        while True:
            img = preprocess_input(images)
            preds = self.klassifier.predict(img)  # predictions of 40 images
            dom_indx = np.argmax(preds[int(np.argmax(preds) / 1000)])  # best individual index
            best_image = int(np.argmax(preds) / 1000)
            adv_img = images[int(np.argmax(preds) / 1000)]  # best adversarial image so far.

            # Dominant category report ##################
            label0 = decode_predictions(preds)  # Reports predictions with label and label values
            label1 = label0[best_image][0]  # Gets the Top1 label and values for reporting.
            dom_cat = label1[1]  # label
            dom_cat_prop = label1[2]  # label probability
            ###########################################
            sys.stdout.write(
                "\rgeneration: "
                + str(count)
                + "/"
                + str(self.max_iter)
                + " ______ "
                + dom_cat
                + ": "
                + str(dom_cat_prop)
                + " ____ index: "
                + str(dom_indx)
            )
            # Stopping the algorithm criteria:
            if count == self.max_iter:
                # if algorithm can not create the adversarial image within "generation" stop the algorithm
                print("\nFailed to generate adversarial image within", self.max_iter, " generations")
                break
            if not self.targeted and dom_indx != anc_indx:
                print("\nAdversarial image is generated successfully in", count, "generations")
                break
            if self.targeted and dom_indx == y and dom_cat_prop > self.confidence:
                print("\nAdversarial image is generated successfully in", count, " generations")
                break

            percentage_middle_class = 1
            percentage_keep = 1

            # Select population classes based on fitness and create 'keep' group
            if self.targeted:
                probs = self._get_class_prob(preds, y)
                fitness = self._get_fitness(probs)
                (
                    elite,
                    middle_class,
                    random_keep,
                ) = self._selection_targeted(images, fitness)
            else:
                probs = self._get_class_prob(preds, anc_indx)
                fitness = self._get_fitness(probs)
                (
                    elite,
                    middle_class,
                    random_keep,
                ) = self._selection_untargeted(images, fitness)
            elite2 = elite.copy()
            keep = np.concatenate((elite2, random_keep))

            # Reproduce individuals by mutating Elits and Middle class---------
            # mutate and crossover individuals
            im_size = x.shape[0] * x.shape[1] * x.shape[2]
            no_of_pixels = self._get_no_of_pixels(im_size)
            mutated_middle_class = self._mutation(
                x,
                no_of_pixels,
                middle_class,
                percentage_middle_class,
                boundary_min,
                boundary_max,
            )
            mutated_keep_group1 = self._mutation(x, no_of_pixels, keep, percentage_keep, boundary_min, boundary_max)
            mutated_keep_group2 = self._mutation(
                x,
                no_of_pixels,
                mutated_keep_group1,
                percentage_keep,
                boundary_min,
                boundary_max,
            )

            all_ = np.concatenate((mutated_middle_class, mutated_keep_group2)) # shape: (30, 224, 224, 3)
            parents_idx = self._get_crossover_parents(all_)
            crossover_group = self._crossover(x, all_, parents_idx) # shape: (30, 224, 224, 3)

            # Create new population
            images = np.concatenate((elite, crossover_group))
            count += 1
        return adv_img


# SET UP your attack

# Import the target CNN and required libraries
from tensorflow.python.keras.applications.vgg16 import (
    decode_predictions,
    preprocess_input,
    VGG16,
)
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
epsilon = 8   # maximum allowed pixel change [-epsilon, epsilon]
# Step 1: Load a clean image and convert it to numpy array:
image = load_img("acorn1.JPEG", target_size=(224, 224), interpolation="lanczos")
x = img_to_array(image)

y = 306  # Optional! Target category index number. It is only for the targeted attack.

kclassifier = VGG16(weights="imagenet")

# Step 3: Built the attack and generate adversarial image:
attackEA = EA(
    kclassifier, max_iter=10000, confidence=0.40, targeted=True
)  # if targeted is True, then confidence will be taken into account.
advers = attackEA._generate(x, y)  # returns adversarial image as .npy file. y is optional.
np.save("advers.npy", advers)
img = Image.fromarray(advers.astype(np.uint8))
img.save("advers.png", "png")

