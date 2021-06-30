# Column names in annotated csv file
from typing import List
import numpy as np

CSV_COL_NAME_EPOCH = "epoch"
CSV_COL_NAME_STEP = "step"
CSV_COL_NAME_IMAGE_ID = "_idx"
CSV_COL_NAME_ROW_ID_WITHIN_IMAGE = "num_rows_annotated"
ANNOTATION_FOLDER_NAME_PREFIX = "manual_annotation_"
COMBINED_AND_CORRECTED_COLUMN_NAME = "text"


class ManualAnnotation:
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
                 digit_image:np.ndarray,
                 h_extend:List,
                 v_extend:List,
                 digit:int,
                 num_clusters:int,
                 cluster_name:str,
                 sample_index:int,
                 epochs_completed = 0):
        self.digit_image = digit_image
        self.h_extend = h_extend
        self.v_extend = v_extend
        self.digit = digit
        self.num_clusters = num_clusters
        self.cluster_name = cluster_name
        self.sample_index = sample_index
        self.epochs_completed = epochs_completed

    def get_cropped_image(self):
        v_extend = self.v_extend
        h_extend = self.h_extend
        if len(v_extend) == 0:
            v_extend = [0, 28]
        if len(h_extend) == 0:
            h_extend = [0, 28]
        return self.digit_image[v_extend[0]:v_extend[1], h_extend[0]:h_extend[1]]

    def get_cropped_and_stripped(self):
        cropped = self.get_cropped_image()
        return ImageConcept.tight_bould_v(ImageConcept.tight_bound_h(cropped))

    @staticmethod
    def tight_bound_h(cropped):
        width = cropped.shape[1]
        height = cropped.shape[0]
        row = 0
        non_zero_pixels_in_col = np.sum(cropped[:, row])
        print(row, non_zero_pixels_in_col)

        while non_zero_pixels_in_col == 0 and row <= width:
            non_zero_pixels_in_col = np.sum(cropped[:, row])
            print(row, non_zero_pixels_in_col)
            row += 1
        from_row = row - 1

        row = height - 1
        non_zero_pixels_in_col = np.sum(cropped[:, row])
        print(row, non_zero_pixels_in_col)

        while non_zero_pixels_in_col == 0 and row > from_row:
            non_zero_pixels_in_col = np.sum(cropped[:, row])
            print(row, non_zero_pixels_in_col)
            row -= 1
        to_row = row + 1
        return cropped[from_row:to_row, :]

    @staticmethod
    def tight_bould_v(cropped):
        width = cropped.shape[1]
        height = cropped.shape[0]
        col = 0
        non_zero_pixels_in_row = np.sum(cropped[col, :])
        print(col, non_zero_pixels_in_row)

        while non_zero_pixels_in_row == 0 and col <= height:
            non_zero_pixels_in_row = np.sum(cropped[col, :])
            print(col, non_zero_pixels_in_row)
            col += 1
        from_col = col - 1

        col = width - 1
        non_zero_pixels_in_row = np.sum(cropped[col, :])
        print(col, non_zero_pixels_in_row)

        while non_zero_pixels_in_row == 0 and col > from_col:
            non_zero_pixels_in_row = np.sum(cropped[col, :])
            print(col, non_zero_pixels_in_row)
            col -= 1
        to_col = col + 1

        return cropped[:, from_col:to_col]

    def todict(self):
        concept_dict = dict()
        concept_dict["digit_image"] = self.digit_image.tolist()
        concept_dict["h_extend"] = self.h_extend
        concept_dict["v_extend"] = self.v_extend
        concept_dict["digit"] = self.digit
        concept_dict["num_clusters"] = self.num_clusters
        concept_dict["cluster_name"] = self.cluster_name
        concept_dict["sample_index"] = self.sample_index
        return concept_dict

    @classmethod
    def fromdict(cls, image_concept_dict):
        instance = cls(digit_image=image_concept_dict["digit_image"],
                       h_extend=image_concept_dict["h_extend"],
                       v_extend=image_concept_dict["v_extend"],
                       digit=image_concept_dict["digit"],
                       num_clusters=image_concept_dict["num_clusters"],
                       cluster_name=image_concept_dict["cluster_name"],
                       sample_index=image_concept_dict["sample_index"] )
        return instance

    # classmethod
    def tolist(self, image_concept_dict):
        return [ image_concept_dict["digit_image"],
                       image_concept_dict["h_extend"],
                       image_concept_dict["v_extend"],
                       image_concept_dict["digit"],
                       image_concept_dict["num_clusters"],
                       image_concept_dict["cluster_name"],
                       image_concept_dict["sample_index"]
        ]

