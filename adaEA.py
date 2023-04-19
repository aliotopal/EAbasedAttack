
# DO NOT TOUCH THIS FILE
import math
import random
from tensorflow.python.keras.applications.mobilenet import decode_predictions
from tensorflow.keras.applications.mobilenet import preprocess_input
random.seed(0)
import time
from PIL import Image
import numpy as np
from tqdm import tqdm
from sys import stdout
np.set_printoptions(threshold=np.inf)


class EA:
    def __init__(self, ancestor, target, model, pop_size, generation, numberOfElites, targetx, ancestorx):
        self.ancestor = ancestor
        self.model = model
        self.pop_size = pop_size
        self.target = target
        self.generation = generation
        self.numberOfElites = numberOfElites
        self.targetx = targetx
        self.ancestorx = ancestorx

    def gen_populations(self, x):
        # Generate populations by using indiv.npy image as source in our case
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

    def get_ssim(self, images, ancestor):
        structural_similarities = [self.ssim(images[i], ancestor, multichannel=True) for i in range(len(images))]
        return np.array(structural_similarities)

    def get_fitness(self, probs):
        fitness =  probs  
        return fitness

    def selectionImgNet(self, images, fitness):
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
        res = (u_factor ** (1.0 / (n + 1))) * im_size  # (n+1)√(u_factor) * im_size
        no_of_pixels = im_size - res
        return no_of_pixels

    def mutationImgNet(self, no_of_pixels, mutation_group, percentage, boundary_min, boundary_max):
        mutated_group = mutation_group.copy()
        random.shuffle(mutated_group)
        no_of_individuals = len(mutated_group)
        for individual in range(int(no_of_individuals * percentage)):
            locations_x = np.random.randint(self.ancestor.shape[0], size=int(no_of_pixels))
            locations_y = np.random.randint(self.ancestor.shape[1], size=int(no_of_pixels))
            locations_z = np.random.randint(self.ancestor.shape[2], size=int(no_of_pixels))
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
    def crossoverImgNet(self, crossover_group, parents_idx, im_size):
        crossedover_group = crossover_group.copy()
        for i in range(0, len(parents_idx), 2):
            parent_index_1 = parents_idx[i]
            parent_index_2 = parents_idx[i + 1]
            size_x = np.random.randint(0, 30)
            start_x = np.random.randint(0, self.ancestor.shape[0] - size_x)
            size_y = np.random.randint(0, 30)
            start_y = np.random.randint(0, self.ancestor.shape[1] - size_y)
            z = np.random.randint(self.ancestor.shape[2])
            temp = crossedover_group[parent_index_1, start_x: start_x + size_x, start_y: start_y + size_y, z]
            crossedover_group[parent_index_1, start_x: start_x + size_x, start_y: start_y + size_y,
            z] = crossedover_group[
                 parent_index_2,
                 start_x: start_x + size_x,
                 start_y: start_y + size_y,
                 z]
            crossedover_group[parent_index_2, start_x: start_x + size_x, start_y: start_y + size_y, z] = temp
        return crossedover_group

    def search(self, x, generation, title, accuracy, ancestorx):
        pop_size = x
        self.generation = generation
        duration = []
        all_bests = []
        counts = []
        predictions = []
        all_best_prob = []
        boundary_min = 0
        boundary_max = 255

        anc = self.ancestor
        images = np.array([anc] * pop_size).astype(int)  # pop_size x ancestor images are created
        count = 0
        bests = []
        accur = 0.0
        begin = time.time()
        # pbar = tqdm(total=generation)
        while accur <= accuracy:
            if count == generation:
                # if algorithm can not create the adversarial image within "generation" stop the algorithm and report the results.
                 filename3 = "Failed/%s-%s.npy" % ( self.ancestorx, self.targetx)
                 np.save(filename3, images[0])
                 img = Image.fromarray(images[0].astype(np.uint8))
                 filename2 = "Failed/%s-%s.png" % ( self.ancestorx, self.targetx)
                 img.save(filename2, 'png')

                 filename4 = "Failed/%s-%s.txt" % (self.ancestorx, self.targetx)
                 file2 = open(filename4, 'w')
                 file2.write("iterations:  " + str(count) + "\n")
                 file2.write("best_prob:  " + str(best_prob) + "\n")
                 endtime = time.time()
                 file2.write("time:  " + str(endtime-begin) + "\n")
                 file2.close()
                 break


            img = preprocess_input(images)
            preds = self.model.predict(img)  # we can find the predictions here
            probs = self.get_class_prob(preds, self.target)
            best_prob = max(probs)
            indx_best_prob = list(probs).index(best_prob)
            best_preds = preds[indx_best_prob]
            accur = best_prob


            percentage_middle_class = 1
            percentage_keep = 1

            fitness = self.get_fitness(probs)

            # select population classes based on fitness and create 'keep' group
            elite, middle_class, elite_fitness, idx_elite, random_keep = self.selectionImgNet(images, fitness)
            elite2 = elite.copy()
            keep = np.concatenate((elite2, random_keep))

            # ---STEP 3: Reproduce individuals by mutating Elits and Middle class---------

            # mutate and crossover individuals
            im_size = self.ancestor.shape[0] * self.ancestor.shape[1] * self.ancestor.shape[2]
            no_of_pixels = self.get_no_of_pixels(im_size)
            mutated_middle_class = self.mutationImgNet(no_of_pixels, middle_class, percentage_middle_class, boundary_min,
                                            boundary_max)
            mutated_keep_group1 = self.mutationImgNet(no_of_pixels, keep, percentage_keep, boundary_min, boundary_max)
            mutated_keep_group2 = self.mutationImgNet(no_of_pixels, mutated_keep_group1, percentage_keep, boundary_min,
                                           boundary_max)

            all_ = np.concatenate((mutated_middle_class, mutated_keep_group2))
            parents_idx = self.get_crossover_parents(all_)
            crossover_group = self.crossoverImgNet(all_, parents_idx, im_size)

            # create new population
            images = np.concatenate((elite, crossover_group))
            stdout.write(f'\rgeneration: {count}/{generation} {self.targetx}: {max(probs)}')
            # pbar.update(1)
            bests.append(math.log(best_prob))
            count += 1

        # pbar.close()
# SAVING adversarial images of each run  **********************************          
        img = Image.fromarray(images[0].astype(np.uint8))
        # filename = "%s-%s-%s_advers.npy" % ( self.ancestorx, self.targetx, count)
        filename2 = "%s_Adversarial.png" % ( self.ancestorx)
        img.save(filename2, 'png')
        # np.save(filename, images[0])
        all_bests.append(bests)
        all_best_prob.append(best_prob)
        predictions.append(best_preds)
        end = time.time()
        duration.append((end - begin))  # add all run's times
        counts.append(count)

        filename3 = "%s_to_%s_report.txt" % (self.ancestorx, self.targetx)
        file2 = open(filename3, 'w')
        count = np.array(count)
        duration = np.array(duration)
        all_best_prob = np.array(all_best_prob)
        file2.write('Number of generations: ' + str(count) + "\n")
        file2.write("It took: %.2f secs.\n" %(duration[0]))
        file2.write("---------------------------------------------------\n")
        img = anc.reshape(1, 224, 224, 3)
        img = preprocess_input(img)
        pred = self.model.predict(img)  # we can find the predictions here
        label = decode_predictions(pred)
        label1 = label[0][0]
        print("\nBefore the image was: " + str(label1[2]) + " " + str(label1[1]))
        print("Now the image is: "  +  str(all_best_prob[0]) + " " + self.targetx)

        file2.write("Before the image was: " + str(label1[2]) + " " + str(label1[1]) +"\n")
        file2.write("Now the image is: "  +  str(all_best_prob[0]) + " " + self.targetx + "\n")
        file2.write("---------------------------------------------------\n")
        file2.close()

# *************************************************************************
        return all_best_prob, all_bests, counts, duration, images, title, predictions  # if accuracy fix return counts not count






