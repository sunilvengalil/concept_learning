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

            return (confidence_1, confidence_2)
        for confidence_label, confidence_interval in ManualAnnotation.confidence_intervals.items():
            if confidence_interval[0] <= self.confidence < confidence_interval[1]:
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

    def get_cluster(self,cluster_num):
        for cluster in self.cluster_list:
            if cluster.id == cluster_num:
                return cluster
