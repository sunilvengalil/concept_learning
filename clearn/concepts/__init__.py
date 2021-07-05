from abc import abstractmethod
import numpy as np

from clearn.analysis import ImageConcept

OPERATORS = ["IDENTITY", "VERTICAL_CONCATENATE"]


class Operator:
    def __init__(self, num_samples_required, operator):
        self.index = 0
        self.num_samples_required = num_samples_required
        self.operator = operator
        if OPERATORS[operator] == "VERTICAL_CONCATENATE":
            self.concept_list = [12, 13, 14, 16, 18, 21, 26, 29, 30, 31, 32, 35, 36]
            # self.concept_list = [12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        elif OPERATORS[operator] == "IDENTITY":
            self.concept_list = [10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        self.concepts_to_use_1 = np.random.choice(self.concept_list, num_samples_required)
        self.concepts_to_use_2 = np.random.choice(self.concept_list, num_samples_required)

    def reset_index(self):
        self.index = 0

    def get_next_images(self):
        c1, c2 = self.concepts_to_use_1[self.index], self.concepts_to_use_2[self.index]
        self.index += 1
        if self.index == self.concepts_to_use_1.shape[0]:
            self.index = 0
        return c1, c2

    @staticmethod
    def add_padding_to_create_full_image(combined):
        masked_image = np.zeros([
            1,
            28,
            28,
            1
        ])
        from_index_h = (28 - combined.shape[0]) // 2
        to_index_h = from_index_h + combined.shape[0]
        from_index_v = (28 - combined.shape[1]) // 2
        to_index_v = from_index_v + combined.shape[1]
        masked_image[0, from_index_h:to_index_h, from_index_v:to_index_v, 0] = combined
        return masked_image

    @abstractmethod
    def apply_operation(self, concept_1_image, concept_2_image):
        pass


class IdentityOperator(Operator):
    def __init__(self, num_samples_required):
        super().__init__(num_samples_required, operator=0)

    def apply_operation(self, concept_1_image, concept_2_image=None):
        masked_image = Operator.add_padding_to_create_full_image(concept_1_image)
        return masked_image


class ConcatenateVerticalOperator(Operator):
    def __init__(self, num_samples_required):
        super().__init__(num_samples_required,
                         1)

    def apply_operation(self, concept_1_image, concept_2_image):
        if concept_1_image.shape[1] < concept_2_image.shape[1]:
            num_zero_padding = concept_2_image.shape[1] - concept_1_image.shape[1]
            zero_padding_image = np.zeros((concept_1_image.shape[0], num_zero_padding))
            concept_1_image = np.hstack([zero_padding_image, concept_1_image])
        elif concept_2_image.shape[1] < concept_1_image.shape[1]:
            num_zero_padding = concept_1_image.shape[1] - concept_2_image.shape[1]
            zero_padding_image = np.zeros((concept_2_image.shape[0], num_zero_padding))
            concept_2_image = np.hstack([zero_padding_image, concept_2_image])
        combined = np.vstack([concept_1_image, concept_2_image])
        masked_image = Operator.add_padding_to_create_full_image(combined)
        return masked_image


def apply_operator(operators_to_use,
                   key_image_concept_map,
                   key_to_label_map,
                   label_start):

    derived_images = np.zeros((operators_to_use.shape[0], 28, 28, 1))
    derived_labels = np.zeros((operators_to_use.shape[0]), np.int8)
    print(key_to_label_map)
    print("Total number of images to generate", operators_to_use.shape[0])
    images_for_operator = [IdentityOperator(operators_to_use.shape[0]),
                           ConcatenateVerticalOperator(operators_to_use.shape[0])]
    concepts_1_to_use = np.zeros(operators_to_use.shape[0])
    concepts_2_to_use = np.zeros(operators_to_use.shape[0])

    for image_index, operator_to_use in enumerate(operators_to_use):
        concept_to_use_1, concept_to_use_2 = images_for_operator[operator_to_use].get_next_images()
        concepts_1_to_use[image_index] = concept_to_use_1
        concepts_2_to_use[image_index] = concept_to_use_2

        image_concept_1: ImageConcept = key_image_concept_map[key_to_label_map[concept_to_use_1]]
        image_concept_2: ImageConcept = key_image_concept_map[key_to_label_map[concept_to_use_2]]
        cropped_1, _, _ = image_concept_1.get_cropped_and_stripped()
        cropped_2, _, _ = image_concept_2.get_cropped_and_stripped()
        derived_images[image_index] = images_for_operator[operator_to_use].apply_operation(cropped_1, cropped_2)
        derived_labels[image_index] = label_start + operator_to_use
        if image_index % 1000 == 0:
            print(f"Generated {image_index} out of {operators_to_use.shape[0]} images")
    return concepts_1_to_use, concepts_2_to_use, derived_images, derived_labels
