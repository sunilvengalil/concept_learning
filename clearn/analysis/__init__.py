# Column names in annotated csv file
import enum
from typing import List
import numpy as np
import cv2
from copy import deepcopy


CSV_COL_NAME_EPOCH = "epoch"
CSV_COL_NAME_STEP = "step"
CSV_COL_NAME_IMAGE_ID = "_idx"
CSV_COL_NAME_ROW_ID_WITHIN_IMAGE = "num_rows_annotated"
ANNOTATION_FOLDER_NAME_PREFIX = "manual_annotation_"
COMBINED_AND_CORRECTED_COLUMN_NAME = "text"


class ManualAnnotation:

    class Label(enum.IntEnum):
        ZERO = 0
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5
        SIX = 6
        SEVEN = 7
        EIGHT = 8
        NINE = 9
        TOP_3 = 12

        VERTICAL_LINE_HALF = 19
        HORIZONTAL_LINE_TOP = 18
        SMALL_CIRCLE = 17
        TOP_FOUR_U = 35
        TOP_5 = 14
        LEFT_OPEN_CIRCLE = 15
        RIGHT_SLANTED_VERTICAL_LINE = 16


    # All intervals are half open which is closed on lower bound and open on upper bound
    confidence_intervals = {"low_confidences_clusters": (0, 0.35),
                            "average_clusters": (0.35, 0.65),
                            "good_clusters": (0.65, 1)
                            }

    def __init__(self, label, confidence):
        self.label = label
        self.confidence = confidence

    def get_label(self):
        # TODO implement this for tuples with multiple element
        if self.confidence is tuple:
            for confidence_label, confidence_interval in ManualAnnotation.confidence_intervals.items():
                if confidence_interval[0] <= self.confidence[0] < confidence_interval[1]:
                    confidence_1 = confidence_label

            for confidence_label, confidence_interval in ManualAnnotation.confidence_intervals.items():
                if confidence_interval[0] <= self.confidence[1] < confidence_interval[1]:
                    confidence_2 = confidence_label

            return confidence_1, confidence_2
        for confidence_label, confidence_interval in ManualAnnotation.confidence_intervals.items():
            if confidence_interval[0] <= self.confidence <= confidence_interval[1]:
                return confidence_label


class Cluster:
    def __init__(self,
                 cluster_id,
                 name,
                 cluster_details,
                 level=1,
                 manual_annotation=None):
        self.id = cluster_id
        self.name = name
        self.details = cluster_details
        self.level = level
        self.manual_annotation = manual_annotation
        self.next_level_clusters = ClusterGroup("Clusters_level_{}".format(level))

    def set_next_level_clusters(self, cluster_group_dict):
        # TODO ***IMPORTANT***  Create  Cluster Group object from dict.
        self.next_level_clusters = cluster_group_dict

    def next_lever_cluster_count(self):
        num_clusters = 0
        for k, v in self.next_level_clusters.items():
            num_clusters += len(v.cluster_list)
        return num_clusters


class ClusterGroup:
    def __init__(self,
                 name,
                 cluster_list=None,
                 manual_annotation=None):
        self.name = name,
        self.iter_index = 0
        self.cluster_list = cluster_list
        self.manual_annotation = manual_annotation

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_index < len(self.cluster_list):
            self.iter_index += 1
            return self.cluster_list[self.iter_index - 1]
        else:
            self.iter_index = 0
            raise StopIteration()

    def is_singleton(self):
        if len(self.cluster_list) == 1:
            return True
        else:
            return False

    def add_cluster(self, cluster):
        if self.cluster_list is None or len(self.cluster_list) == 0:
            self.cluster_list = [cluster]
        else:
            self.cluster_list.append(cluster)

    def get_cluster(self, cluster_num):
        for cluster in self.cluster_list:
            if cluster.id == cluster_num:
                return cluster


class ImageConcept:
    def __init__(self,
                 digit_image: np.ndarray,
                 h_extend: List,
                 v_extend: List,
                 digit: int,
                 num_clusters: int,
                 cluster_name: str,
                 sample_index: int,
                 epochs_completed=0,
                 name=None,
                 split_id=1,
                 mode_id=1,
                 should_use_original_cordinate=True,
                 top_largest_cc=-1,
                 bottom_largest_cc=-1,
                 left_largest_cc=-1,
                 right_largest_cc=-1
                 ):
        self.split_id = split_id
        self.mode_id = mode_id
        self.digit_image = digit_image
        if len(h_extend) == 0:
            self.h_extend = [0, np.squeeze(digit_image).shape[1]]
        else:
            self.h_extend = h_extend

        if len(v_extend) == 0:
            self.v_extend = [0, np.squeeze(digit_image).shape[0]]
        else:
            self.v_extend = v_extend

        self.orig_v_extend = deepcopy(self.v_extend)
        self.orig_h_extend = deepcopy(self.h_extend)

        self.should_use_original_cordinate = should_use_original_cordinate

        self.digit:int = digit

        self.digits:List[int] = [digit]

        self.num_clusters = num_clusters
        self.cluster_name = cluster_name
        self.sample_index = sample_index
        self.epochs_completed = epochs_completed
        if name is None:
            self.name = self.get_key()
        else:
            key = self.get_key()
            self.name = f"{name}_{key}"

        self.top_largest_cc = top_largest_cc
        self.left_largest_cc = left_largest_cc
        self.bottom_largest_cc = bottom_largest_cc
        self.right_largest_cc = right_largest_cc

    def get_full_image(self):
        squeezed = np.squeeze(self.digit_image)
        mask = np.zeros_like(squeezed)
        # print(mask.shape, self.v_extend, self.h_extend)
        mask[self.v_extend[0]:self.v_extend[1], self.h_extend[0]:self.h_extend[1]] = self.get_cropped_image()
        return mask

    def find_largest_cc(self):
        cropped = self.get_full_image()
        thresh = cv2.threshold((cropped * 256).astype(np.uint8), 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        label_of_largest_connected_component = np.argmax(stats[1:, 4]) + 1
        centroid_of_largest_connected_component = centroids[label_of_largest_connected_component]
        self.top_largest_cc = stats[label_of_largest_connected_component, cv2.CC_STAT_TOP]
        self.left_largest_cc = stats[label_of_largest_connected_component, cv2.CC_STAT_LEFT]
        self.bottom_largest_cc = self.top_largest_cc + stats[label_of_largest_connected_component, cv2.CC_STAT_HEIGHT]
        self.right_largest_cc = self.left_largest_cc + stats[label_of_largest_connected_component, cv2.CC_STAT_WIDTH]
        self.centroid_largest_cc = centroid_of_largest_connected_component
        # print(centroid_of_largest_connected_component)
        cropped[labels != label_of_largest_connected_component] = 0
        return cropped

    def use_largest_cc(self):
        self.should_use_original_cordinate = False
        self.v_extend = [self.top_largest_cc, self.bottom_largest_cc]
        self.h_extend = [self.left_largest_cc, self.right_largest_cc]

    def use_original_cc(self):
        self.should_use_original_cordinate = True
        self.v_extend = deepcopy(self.orig_v_extend)
        self.h_extend = deepcopy(self.orig_h_extend)

    def get_image_largest_cc(self):
        squeezed = np.squeeze(self.digit_image)
        mask = np.zeros_like(squeezed)
        # print(mask.shape, self.v_extend, self.h_extend)
        cropped = squeezed[self.top_largest_cc:self.bottom_largest_cc, self.left_largest_cc:self.right_largest_cc]
        mask[self.top_largest_cc:self.bottom_largest_cc, self.left_largest_cc:self.right_largest_cc] = cropped
        return mask

    def get_cropped_image(self):
        v_extend = self.v_extend
        h_extend = self.h_extend
        # if len(v_extend) == 0:
        #     v_extend = [0, 28]
        # if len(h_extend) == 0:
        #     h_extend = [0, 28]
        cropped = np.asarray(self.digit_image)
        # print(h_extend, v_extend, cropped.shape)
        return cropped[0, v_extend[0]:v_extend[1], h_extend[0]:h_extend[1], 0]

    def get_cropped_and_stripped(self):
        cropped = np.squeeze(self.get_cropped_image())
        h_im, h_extend = ImageConcept.tight_bound_h(cropped)
        h_extend[0] += self.h_extend[0]
        h_extend[1] += self.h_extend[0]
        cropped_and_stripped, v_extend = ImageConcept.tight_bound_v(h_im)
        v_extend[0] += self.v_extend[0]
        v_extend[1] += self.v_extend[0]

        return cropped_and_stripped, h_extend, v_extend

    @staticmethod
    def tight_bound_h(cropped):
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            raise Exception(f"One or both dimension of cropped shape is {cropped.shape}")
        width = cropped.shape[1]
        row = 0
        max_value = np.max(cropped)
        single_col = cropped[:, row]
        non_zero_pixels_in_col = np.sum(single_col[single_col > 0.7 * max_value])
        if non_zero_pixels_in_col == 0:
            while non_zero_pixels_in_col == 0 and row < width:
                # non_zero_pixels_in_col = np.sum(cropped[:, row])
                single_col = cropped[:, row]
                non_zero_pixels_in_col = np.sum(single_col[single_col > 0.7 * max_value])
                row += 1
            from_row = row - 1
        else:
            from_row = row

        row = width - 1
        single_col = cropped[:, row]
        non_zero_pixels_in_col = np.sum(single_col[single_col > 0.7 * max_value])
        if non_zero_pixels_in_col == 0:
            while non_zero_pixels_in_col == 0 and row > from_row:
                single_col = cropped[:, row]
                # non_zero_pixels_in_col = np.sum(cropped[:, row])
                non_zero_pixels_in_col = np.sum(single_col[single_col > 0.7 * max_value])
                row -= 1
            to_row = row + 1
        else:
            to_row = row
        h_extend_and_stripped = [max(0, from_row - 1), min(to_row + 1, cropped.shape[1])]
        return cropped[:, h_extend_and_stripped[0]:h_extend_and_stripped[1]], h_extend_and_stripped

    @staticmethod
    def tight_bound_v(cropped):
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            raise Exception(f"One or both dimension of cropped shape is {cropped.shape}")
        height = cropped.shape[0]
        col = 0
        # non_zero_pixels_in_row = np.sum(cropped[col, :])
        max_value = np.max(cropped)
        single_row = cropped[col, :]
        non_zero_pixels_in_row = np.sum(single_row[single_row > 0.7 * max_value])
        if non_zero_pixels_in_row == 0:
            while non_zero_pixels_in_row == 0 and col < height:
                # non_zero_pixels_in_row = np.sum(cropped[col, :])
                single_row = cropped[col, :]
                non_zero_pixels_in_row = np.sum(single_row[single_row > 0.7 * max_value])
                col += 1
            from_col = col - 1
        else:
            from_col = col
        col = height - 1
        # non_zero_pixels_in_row = np.sum(cropped[col, :])
        single_row = cropped[col, :]
        non_zero_pixels_in_row = np.sum(single_row[single_row > 0.7 * max_value])

        if non_zero_pixels_in_row == 0:
            while non_zero_pixels_in_row == 0 and col > from_col:
                # non_zero_pixels_in_row = np.sum(cropped[col, :])
                single_row = cropped[col, :]

                non_zero_pixels_in_row = np.sum(single_row[single_row > 0.7 * max_value])
                col -= 1
            to_col = col + 1
        else:
            to_col = col
        v_extend_stripped = [max(from_col - 1, 0), min(to_col + 1, cropped.shape[0])]
        return cropped[v_extend_stripped[0]:v_extend_stripped[1], :], v_extend_stripped

    def get_key(self):
        return f"{self.digit}_{self.h_extend[0]}_{self.h_extend[1]}_{self.v_extend[0]}_{self.v_extend[1]}"

    def todict(self):
        concept_dict = dict()
        if isinstance(self.digit_image, np.ndarray):
            concept_dict["digit_image"] = self.digit_image.tolist()
        else:
            concept_dict["digit_image"] = self.digit_image

        if isinstance(self.h_extend, np.ndarray):
            concept_dict["h_extend"] = self.h_extend.tolist()
        else:
            concept_dict["h_extend"] = self.h_extend

        concept_dict["h_extend"] = [int(self.h_extend[0]), int(self.h_extend[1])]

        concept_dict["v_extend"] = [int(self.v_extend[0]), int(self.v_extend[1])]

        concept_dict["digit"] = self.digit
        concept_dict["num_clusters"] = self.num_clusters
        concept_dict["cluster_name"] = self.cluster_name
        concept_dict["sample_index"] = self.sample_index
        concept_dict["epochs_completed"] = self.epochs_completed
        concept_dict["split_id"] = self.split_id
        concept_dict["mode_id"] = self.mode_id
        concept_dict["name"] = self.name
        concept_dict["should_use_original_cordinate"] = self.should_use_original_cordinate
        concept_dict["orig_v_extend"] = [int(self.orig_v_extend[0]), int(self.orig_v_extend[1])]
        concept_dict["orig_h_extend"] = [int(self.orig_h_extend[0]), int(self.orig_h_extend[1])]
        concept_dict["v_extend_largest_cc"] = [int(self.top_largest_cc), int(self.bottom_largest_cc)]
        concept_dict["h_extend_largest_cc"] = [int(self.left_largest_cc), int(self.right_largest_cc)]
        concept_dict["digits"] = self.digits

        return concept_dict

    @classmethod
    def fromdict(cls, image_concept_dict):
        instance = cls(digit_image=np.asarray(image_concept_dict["digit_image"]),
                       h_extend=image_concept_dict["h_extend"],
                       v_extend=image_concept_dict["v_extend"],
                       digit=image_concept_dict["digit"],
                       num_clusters=image_concept_dict["num_clusters"],
                       cluster_name=image_concept_dict["cluster_name"],
                       sample_index=image_concept_dict["sample_index"]
                        )
        if "epochs_completed" in image_concept_dict:
            instance.epochs_completed = image_concept_dict["epochs_completed"],

        if "name" in image_concept_dict:
            instance.name = image_concept_dict["name"],
        if "should_use_original_cordinate" in image_concept_dict:
            instance.should_use_original_cordinate = image_concept_dict["should_use_original_cordinate"]
        if "split_id" in image_concept_dict:
            instance.split_id = image_concept_dict["split_id"]
        if "mode_id" in image_concept_dict:
            instance.mode_id = image_concept_dict["mode_id"]
        if "top_largest_cc" in image_concept_dict:
            instance.top_largest_cc = image_concept_dict["top_largest_cc"]
        if "bottom_largest_cc" in image_concept_dict:
            instance.bottom_largest_cc = image_concept_dict["bottom_largest_cc"]
        if "left_largest_cc" in image_concept_dict:
            instance.left_largest_cc = image_concept_dict["left_largest_cc"]
        if "right_largest_cc" in image_concept_dict:
            instance.right_largest_cc = image_concept_dict["right_largest_cc"]
        if "digits" in image_concept_dict:
            instance.digits = image_concept_dict["digits"]
        else:
            instance.digits = [instance.digit]
        return instance

    def tolist(self, image_concept_dict):
        return [image_concept_dict["digit_image"],
                image_concept_dict["h_extend"],
                image_concept_dict["v_extend"],
                image_concept_dict["digit"],
                image_concept_dict["num_clusters"],
                image_concept_dict["cluster_name"],
                image_concept_dict["sample_index"]
                ]