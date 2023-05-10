import math
import numpy as np
from sys import stdout
import random
random.seed(0)

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


# Do NOT touch the class.
class EA:
    def __init__(self, klassifier, max_iter, confidence, targeted):
        self.klassifier = klassifier
        self.max_iter = max_iter
        self.confidence = confidence
        self.targeted = targeted
        self.pop_size = 40
        self.numberOfElites = 10

    def gen_populations(self, x):
        population = []
        for i in range(self.pop_size):
            population.append(x)

    def get_class_prob(self, preds, class_no):
        return preds[:, class_no]

    def l2_norm(self, ind, ancestor):
        l2_norm = np.sum((ind - ancestor) ** 2)
        return math.sqrt(l2_norm)

    def get_distance(self, images, ancestor):
        distance = [None] * len(images)
        for ind in range(len(images)):
            distance[ind] = self.l2_norm(images[ind], ancestor)
        return np.array(distance)


    def get_fitness(self, probs):
        # it returns images' probabilities, but different objective functions can be used here.
        fitness =  probs
        return fitness

    def selection_untargeted(self, images, fitness):  # 40 images
        idx_elite = fitness.argsort()[:self.numberOfElites]
        elite_fitness = fitness[idx_elite]
        half_pop_size = images.shape[0] / 2
        idx_middle_class = fitness.argsort()[self.numberOfElites: int(half_pop_size)]
        elite = images[idx_elite, :]
        middle_class = images[idx_middle_class, :]

        possible_idx = set(range(0, images.shape[0])) - set(idx_elite)
        idx_keep = random.sample(possible_idx, int(images.shape[0]/2-self.numberOfElites))
        random_keep = images[idx_keep]
        return elite, middle_class, elite_fitness, idx_elite, random_keep

    def selection_targeted(self, images, fitness):
        idx_elite = fitness.argsort()[-self.numberOfElites:]
        elite_fitness = fitness[idx_elite]
        half_pop_size = images.shape[0] / 2
        idx_middle_class = fitness.argsort()[int(half_pop_size):-self.numberOfElites]
        elite = images[idx_elite, :][::-1]
        middle_class = images[idx_middle_class, :]

        possible_idx = set(range(0, images.shape[0])) - set(idx_elite)
        idx_keep = random.sample(possible_idx, int(images.shape[0]/2-self.numberOfElites))
        random_keep = images[idx_keep]
        return elite, middle_class, elite_fitness, idx_elite, random_keep

    def get_no_of_pixels(self, im_size):
        u_factor = np.random.uniform(0.0, 1.0)
        n = 60  # normally 60, the smaller n -> more pixels
        res = (u_factor ** (1.0 / (n + 1))) * im_size
        no_of_pixels = im_size - res
        return no_of_pixels

    def mutation(self, x, no_of_pixels, mutation_group, percentage, boundary_min, boundary_max):
        mutated_group = mutation_group.copy()
        random.shuffle(mutated_group)
        no_of_individuals = len(mutated_group)
        for individual in range(int(no_of_individuals * percentage)):
            locations_x = np.random.randint(x.shape[0], size=int(no_of_pixels))
            locations_y = np.random.randint(x.shape[1], size=int(no_of_pixels))
            locations_z = np.random.randint(x.shape[2], size=int(no_of_pixels))
            new_values = random.choices(np.array([-1, 1]), k=int(no_of_pixels))
            mutated_group[individual, locations_x, locations_y, locations_z] = mutated_group[
                                                                                   individual, locations_x, locations_y, locations_z] - new_values
        mutated_group = np.clip(mutated_group, boundary_min, boundary_max)
        return mutated_group

    def get_crossover_parents(self, crossover_group):
        size = crossover_group.shape[0]
        no_of_parents = random.randrange(0, size, 2)
        parents_idx = random.sample(range(0, size), no_of_parents)
        return parents_idx

    # select random no of pixels to interchange
    def crossover(self, x, crossover_group, parents_idx, im_size):
        crossedover_group = crossover_group.copy()
        for i in range(0, len(parents_idx), 2):
            parent_index_1 = parents_idx[i]
            parent_index_2 = parents_idx[i + 1]
            crossover_range = int(x.shape[0] * 0.15)  # 15% of the image will be crossovered.
            size_x = np.random.randint(0, crossover_range)
            start_x = np.random.randint(0, x.shape[0] - size_x)
            size_y = np.random.randint(0, crossover_range)
            start_y = np.random.randint(0, x.shape[1] - size_y)
            z = np.random.randint(x.shape[2])
            temp = crossedover_group[parent_index_1, start_x: start_x + size_x, start_y: start_y + size_y, z]
            crossedover_group[parent_index_1, start_x: start_x + size_x, start_y: start_y + size_y,
            z] = crossedover_group[
                 parent_index_2,
                 start_x: start_x + size_x,
                 start_y: start_y + size_y,
                 z]
            crossedover_group[parent_index_2, start_x: start_x + size_x, start_y: start_y + size_y, z] = temp
        return crossedover_group

    def generate(self, x, y=None):
        boundary_min = 0
        boundary_max = 255

        img = x.reshape((1, x.shape[0], x.shape[1], x.shape[2])).copy()
        img = preprocess_input(img)
        preds = self.klassifier.predict(img)
        label0 = decode_predictions(preds)
        label1 = label0[0][0]  # Gets the Top1 label and values for reporting.
        ancestor = label1[1]  # label
        ancIndx = np.argmax(preds)
        print(f'Before the image is:  {ancestor} --> {label1[2]} ____ index: {ancIndx}')
        if self.targeted:
            print('Target class index number is: ', y)
        images = np.array([x] * self.pop_size).astype(int)  # pop_size * ancestor images are created
        count = 0
        while True:
            img = preprocess_input(images)
            preds = self.klassifier.predict(img)  # predictions of 40 images
            # domIndx = np.mod(np.argmax(preds), 1000)
            domIndx = np.argmax(preds[int(np.argmax(preds) / 1000)])
            # Dominant category report ##################
            label0 = decode_predictions(preds)  # Reports predictions with label and label values
            label1 = label0[0][0]  # Gets the Top1 label and values for reporting.
            domCat = label1[1]   # label
            domCat_prop = label1[2]  # label probability
            ###########################################

            percentage_middle_class = 1
            percentage_keep = 1


        # Select population classes based on fitness and create 'keep' group
            if self.targeted:
                probs = self.get_class_prob(preds, y)
                fitness = self.get_fitness(probs)
                elite, middle_class, elite_fitness, idx_elite, random_keep = self.selection_targeted(images, fitness)
            else:
                probs = self.get_class_prob(preds, ancIndx)
                fitness = self.get_fitness(probs)
                elite, middle_class, elite_fitness, idx_elite, random_keep = self.selection_untargeted(images, fitness)
            elite2 = elite.copy()
            keep = np.concatenate((elite2, random_keep))

        # Reproduce individuals by mutating Elits and Middle class---------

            # mutate and crossover individuals
            im_size = x.shape[0] * x.shape[1] * x.shape[2]
            no_of_pixels = self.get_no_of_pixels(im_size)
            mutated_middle_class = self.mutation(x, no_of_pixels, middle_class, percentage_middle_class, boundary_min,
                                            boundary_max)
            mutated_keep_group1 = self.mutation(x, no_of_pixels, keep, percentage_keep, boundary_min, boundary_max)
            mutated_keep_group2 = self.mutation(x, no_of_pixels, mutated_keep_group1, percentage_keep, boundary_min,
                                           boundary_max)

            all_ = np.concatenate((mutated_middle_class, mutated_keep_group2))
            parents_idx = self.get_crossover_parents(all_)
            crossover_group = self.crossover(x, all_, parents_idx, im_size)

        # Create new population
            images = np.concatenate((elite, crossover_group))

            stdout.write(f'\rgeneration: {count}/{self.max_iter} ______ {domCat}: {domCat_prop} ____ index: {domIndx}')
            count += 1

            if count == self.max_iter:
                # if algorithm can not create the adversarial image within "generation" stop the algorithm
                print(f"Failed to generate adversarial image within {self.max_iter} generations")
                break

            if not self.targeted and domIndx != ancIndx:
                break
            if self.targeted and domIndx == y and domCat_prop >= self.confidence:
                break

        return images[0]


# Step 1: Load a clean image and convert it to numpy array:
original_image = 'acorn1.JPEG'
image = load_img(f"{original_image}", target_size=(224, 224))  # Load the image and resize it into CNN's input size.
x = img_to_array(image)
y = 306  # Optional! It is for targeted attack.


# Step 2: Load a target CNN trained with ImageNet dataset from keras.applications library:
from tensorflow.keras.applications.nasnet import NASNetMobile  # Import the target CNN
from tensorflow.python.keras.applications.nasnet import decode_predictions, preprocess_input
kclassifier = NASNetMobile(weights='imagenet')


# Step 3: Built the attack and generate adversarial image:
attackEA = EA(kclassifier, max_iter=10000, confidence =0.75, targeted = False)   # if targeted is True, then confidence will be taken into account.
advers = attackEA.generate(x, y)   # returns adversarial image as .npy file. y is optional.










