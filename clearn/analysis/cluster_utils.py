from typing import *
from sklearn.cluster import KMeans
import tensorflow as tf
import numpy as np
from numpy.linalg import norm

from pandas import DataFrame
from clearn.config import ExperimentConfig
from clearn.dao.idao import IDao
from clearn.analysis.encode_decode import decode
import math
from matplotlib import pyplot as plt
import matplotlib
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis
import scipy as sp


from clearn.analysis import Cluster
from clearn.analysis import ClusterGroup
from clearn.analysis import ManualAnnotation
from clearn.models.classify.classifier import ClassifierModel
from clearn.experiments.experiment import get_model
from clearn.utils.utils import get_pmf_y_given_z, get_latent_vector_column


def trace_dim(f, num_trace_steps, dim, feature_dim, z_min, z_max):
    """
    Returns a tensor of dimension (`num_trace_steps`, `feature_dim`) with each row containing feature vector `f`
    with values in dimension `dim` changed from `z_min` to `z_max`
    @param f feature vector
    @param num_trace_steps number of steps to trace
    @param dim dimension which should be modified during tracing
    @param feature_dim dimension of feature vector
    @param z_min starting value of tracing
    @param z_max end value of tracing
    """
    z = np.zeros([num_trace_steps, feature_dim])
    for i in range(num_trace_steps):
        z[i] = f

    step = 1 / num_trace_steps
    for i in range(num_trace_steps):
        alpha_i = step * i
        z[i, dim] = alpha_i * z_min + (1 - alpha_i) * z_max
    return z


def plot_features(exp_config, features, digits, dimensions_to_be_plotted, new_fig=True):
    """
    Plot the mean latent vector corresponding to symbols `digits`
    @param exp_config Instance of ExperimentConfig
    @param digits list of symbols for which the feature should be plotted
    @param dimensions_to_be_plotted indices of the feature dimensions that needs to be plotted
    @param new_fig If true will plot in a new figure
    """

    # Load Model
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = ClassifierModel(exp_config,
                                sess,
                                epoch=1,
                                num_units_in_layer=exp_config.num_units,
                                )
        print(model.get_trainable_vars())
        num_steps_completed = model.counter
        print("Number of steps completed={}".format(num_steps_completed))
        num_batches = exp_config.num_train_samples / exp_config.BATCH_SIZE
        epochs_completed = num_steps_completed // num_batches
        print("Number of epochs completed {}".format(epochs_completed))
        means = np.asarray([np.mean(features[digit], axis=0) for digit in digits])
        if new_fig:
            plt.figure(figsize=(15, 10))

        # TODO remove subplot
        # ax = fig.add_subplot(2, 2, i % 4 + 1)
        for d, mean in zip(digits, means):
            # TODO plot only sensitive dimensions
            plt.plot(mean[dimensions_to_be_plotted[d]])
            # plt.xticks(list(range(len(sensitive_dimensions))), sensitive_dimensions[d])
            # ax = plt.gca()
            # iax = inset_axes(ax, width="50%", height=1, loc=1)
            # plt.axes([0.65, 0.65, 0.2, 0.2], facecolor='y')
        reconstructed_image_for_means = decode(model, means.reshape([len(digits), 10]), exp_config.BATCH_SIZE)
        # plt.imshow(np.squeeze(reconstructed_image),cmap="gray")
    tf.reset_default_graph()
    return reconstructed_image_for_means


def cluster_next_level(exp_config: ExperimentConfig,
                       df: DataFrame,
                       cluster_column_name_2,
                       cluster_labels,
                       z_col_names,
                       model_type,
                       epochs_completed,
                       dao: IDao,
                       cluster_group_dict: Dict[str, ClusterGroup],
                       processed_clusters=[],
                       cluster_type="unknown_cluster"
                       ):
    if cluster_type in cluster_group_dict.keys():
        for cluster in cluster_group_dict[cluster_type]:
            print(cluster.id)
            if cluster.id in processed_clusters:
                continue
            _indices = np.where(cluster_labels == cluster.id)
            _df = df.iloc[_indices]

            _latent_vectors = _df[z_col_names].values
            tf.reset_default_graph()
            _decoded_images, _cluster_centers, _cluster_labels = cluster_and_decode_latent_vectors(
                model_type,
                10,
                _latent_vectors,
                exp_config,
                dao
            )
            df[cluster_column_name_2].iloc[_indices] = _cluster_labels
            image_filename = exp_config.ANALYSIS_PATH + f"cluster_centers__level_2_epoch_{epochs_completed}_cluster_id_{cluster.id}.png"

            display_images(_decoded_images, image_filename, _cluster_centers)

            return cluster, _cluster_centers, _cluster_labels
    return None, None, None


def cluster_next_level_gmm(exp_config: ExperimentConfig,
                           df: DataFrame,
                           cluster_column_name_2,
                           cluster_labels,
                           model_type,
                           epochs_completed,
                           dao: IDao,
                           cluster_group_dict: Dict[str, ClusterGroup],
                           cluster_type="unknown_cluster"
                           ):
    _, _, z_col_names, _ = get_latent_vector_column(exp_config.Z_DIM)
    level2_manual_annotations = dict()
    if cluster_type in cluster_group_dict.keys():
        df[cluster_column_name_2] = -1
        tf.reset_default_graph()
        for cluster in cluster_group_dict[cluster_type]:
            print(cluster.id)
            _indices = np.where(cluster_labels == cluster.id)
            _df = df.iloc[_indices]

            _latent_vectors = _df[z_col_names].values
            tf.reset_default_graph()
            _decoded_images, _cluster_centers, _cluster_labels, posterior_proba_level_2 = cluster_and_decode_latent_vectors_gmm(
                model_type,
                10,
                _latent_vectors,
                exp_config,
                dao
            )
            df[cluster_column_name_2].iloc[_indices] = _cluster_labels
            print(posterior_proba_level_2.shape, _cluster_labels.shape)
            for i in range(10):
                __indices = np.where((cluster_labels == cluster.id) &
                                     (df[cluster_column_name_2].values == i))[0]
                df[f"confidence_level_2_{cluster.id}_{i}"] = 0
                df[f"confidence_level_2_{cluster.id}_{i}"].iloc[_indices] = posterior_proba_level_2[:, i]

            image_filename = exp_config.ANALYSIS_PATH + f"cluster_centers__level_2_epoch_{epochs_completed}_cluster_id_{cluster.id}.png"

            display_images(_decoded_images, image_filename, _cluster_centers)
            # class_labels = widgets.Text(place_holder="-1,-1,-1,-1,-1,-1,-1,-1,-1,-1",
            #     description="Labels")
            # display(class_labels)
            # labels_input = input("Enter the labels in above text box")
            # print(validate_tokenize(labels_input, 10))

            # confidence_input = input("Enter the confidence in above text box")
            # print(validate_tokenize(confidence_input, 10))

            level_2_cluster_dict = dict()
            level_2_cluster_dict["cluster_centers"] = _cluster_centers.tolist()
            level_2_cluster_dict["cluster_labels"] = _cluster_labels.tolist()
            level_2_cluster_dict["posterier_prob"] = posterior_proba_level_2.tolist()
            # level_2_cluster_dict["manual_labels"] = validate_tokenize_int(labels_input, 10)
            # level_2_cluster_dict["manual_confidences"] = validate_tokenize_float(confidence_input, 10)
            level2_manual_annotations[cluster.id] = level_2_cluster_dict

    return level2_manual_annotations


def plot_number_of_samples_vs_label(exp_config: ExperimentConfig,
                                    cluster_group_name: str,
                                    cluster_group: ClusterGroup,
                                    title_string: str,
                                    manual_labels: List[int],
                                    manual_confidence: List[float]
                                    ):
    for cluster in cluster_group:
        cluster_num = cluster.id
        cluster_details = cluster.details
        _df = cluster_details["cluster_data_frame"]
        fig = plt.figure(figsize=(20, 8))
        # fig.tight_layout()
        plt.title(title_string.format(cluster_group_name, cluster_num), fontsize=22)
        plt.xlabel("Label")
        plt.ylabel("Number of samples")
        number_of_samples_for_label = get_pmf_y_given_z(_df, "label", exp_config.Z_DIM, normalized=False)
        plt.bar(x=number_of_samples_for_label.index.values + 0.5,
                height=number_of_samples_for_label,
                width=0.8, align="center")
        plt.text(0.22, 0.84, f"Manual Label : {manual_labels[cluster_num]},{manual_confidence[cluster_num]}",
                 bbox=dict(facecolor='red', alpha=0.5),
                 transform=fig.transFigure)
        plt.xticks(number_of_samples_for_label.index.values + 0.5, number_of_samples_for_label.index.values)
        plt.ylim(0, max(number_of_samples_for_label) * 1.1)
        plt.grid(which="major", axis="x")


def plot_distance_distribution(df: DataFrame,
                               clusters,
                               cluster_column_name: str,
                               manual_labels: List[int]):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}
    legend_string = "Cluster Number={} Label={}"
    matplotlib.rc('font', **font)
    plt.figure(figsize=(20, 8))
    for cluster_num in clusters:
        _df = get_samples_for_cluster(df, cluster_num, cluster_column_name)
        col_name = f"distance_{cluster_num}"
        v, b = np.histogram(_df[col_name].values, bins=20, normed=False)
        v = v / np.sum(v)
        plt.plot(b[:-1], v, label=legend_string.format(cluster_num, manual_labels[cluster_num]))
        plt.xlabel("Distance from cluster center")
        plt.ylabel("Percentage of samples")
        plt.title("Distribution of distance from cluster center")
    plt.legend()


def decode_latent_vectors(model_type: str,
                          cluster_centers: np.ndarray,
                          exp_config: ExperimentConfig,
                          dao: IDao):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = get_model(dao, exp_config, model_type, num_epochs=0, sess=sess)
        z = np.zeros([cluster_centers.shape[0], exp_config.Z_DIM])
        for i in range(cluster_centers.shape[0]):
            z[i, :] = cluster_centers[i]
        decoded_images = decode(model, z, exp_config.BATCH_SIZE)
        return decoded_images


def cluster_and_decode_latent_vectors(model_type: str,
                                      num_clusters: int,
                                      latent_vectors: np.ndarray,
                                      exp_config: ExperimentConfig,
                                      dao: IDao):
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    cluster_centers = kmeans.cluster_centers_
    decoded_images = decode_latent_vectors(model_type, cluster_centers, exp_config, dao)

    return decoded_images, cluster_centers, cluster_labels


def cluster_and_decode_latent_vectors_gmm(model_type: str,
                                          num_clusters: int,
                                          latent_vectors: np.ndarray,
                                          exp_config: ExperimentConfig,
                                          dao: IDao):
    gm = GaussianMixture(n_components=num_clusters, random_state=0)
    cluster_labels = gm.fit_predict(latent_vectors)
    cluster_centers = gm.means_
    decoded_images = decode_latent_vectors(model_type, cluster_centers, exp_config, dao)
    posterior = gm.predict_proba(latent_vectors)

    return decoded_images, cluster_centers, cluster_labels, posterior


def display_images(decoded_images,
                   image_filename,
                   title
                   ):
    colormap = "Greys"
    fig = plt.figure()
    fig.tight_layout()
    num_cols = 4
    num_images = decoded_images.shape[0]
    print(num_images)
    num_rows = math.ceil(num_images / num_cols)
    fig.suptitle(title)
    for i in range(num_images):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(np.squeeze(decoded_images[i]), cmap=colormap)
    plt.savefig(image_filename,
                bbox="tight",
                pad_inches=0)
    plt.show()


# Given a cluster_num, return the cluster object and ClusterGroup which it belongs to
def get_cluster(cluster_num,
                cluster_group_dict):
    for cluster_group_name, cluster_group in cluster_group_dict.items():
        cluster = cluster_group.get_cluster(cluster_num)
        if cluster is not None:
            return cluster_group, cluster


def assign_manual_label_and_confidence(df,
                                       manual_annotation_dict,
                                       dist_to_conf,
                                       cluster_group_dict,
                                       cluster_column_name_2,
                                       assign_only_correct=False,
                                       estimate_type="distance_function"
                                       ):
    def assign_label(_df, _manual_label):
        _indices = np.where((np.asarray(cluster_labels) == cluster.id)
                            & (_df[cluster_column_name_2].values == _cluster.id))[0]
        _df["manual_annotation"].iloc[_indices] = _manual_label
        if estimate_type == "distance_function":
            _distance_df = df[f"distance_level_2_{cluster.id}_{_cluster.id}"]
            dst = _distance_df.iloc[_indices]
            _df["manual_annotation_confidence"].iloc[_indices] = _cluster.manual_annotation.confidence * dist_to_conf(dst)
            _df["distance_to_confidence"].iloc[_indices] = dist_to_conf(dst)
        elif estimate_type == "map":
            _confidence_df = df[f"confidence_level_2_{cluster.id}_{_cluster.id}"]
            _df["manual_annotation_confidence"].iloc[_indices] = _cluster.manual_annotation.confidence * _confidence_df.iloc[_indices]
        if assign_only_correct:
            wrong_indices = (_df["manual_annotation"] == _manual_label) & (_df["label"] != _manual_label)
            _df["manual_annotation_confidence"].loc[wrong_indices] = 0
    num_individual_samples_annotated = 0
    df["manual_annotation"] = np.ones(df.shape[0]) * -1
    df["manual_annotation_confidence"] = np.zeros(df.shape[0])
    df["distance_to_confidence"] = np.zeros(df.shape[0])
    # manually given label for each cluster center -1 for unknown (cluster which has no semantic meaning)
    manual_labels = manual_annotation_dict["manual_labels"]
    cluster_labels = np.asarray(manual_annotation_dict["cluster_labels"])

    num_clusters = len(manual_labels)
    for annotate_cluster in range(num_clusters):
        manual_label = manual_labels[annotate_cluster]
        _manual_confidence = manual_annotation_dict["manual_confidence"][annotate_cluster]
        if isinstance(manual_label, tuple) or isinstance(manual_label, list):
            _, cluster = get_cluster(annotate_cluster, cluster_group_dict)
            for _cluster in cluster.next_level_clusters["good_clusters"]:
                _distance_df = df[f"distance_level_2_{cluster.id}_{_cluster.id}"]
                _confidence_df = df[f"confidence_level_2_{cluster.id}_{_cluster.id}"]
                _manual_label = _cluster.manual_annotation.label

                if isinstance(_manual_label, tuple) or isinstance(_manual_label, list):
                    # TODO add this code
                    pass
                elif _manual_label != -1:
                    assign_label(df, _manual_label)
        elif manual_label != -1:
            print("Manual Label", manual_label)
            indices = np.where(cluster_labels == annotate_cluster)
            df["manual_annotation"].iloc[indices] = manual_label
            _, cluster = get_cluster(annotate_cluster, cluster_group_dict)
            print(df[df["manual_annotation"] == manual_label].shape, cluster.details["cluster_data_frame"].shape)
            num_correct = df[(manual_label == df["manual_annotation"]) & (df["label"] == manual_label)].shape[0]
            print("Num correct={}".format(num_correct))

            percentage_correct = 100 * num_correct / df[df["manual_annotation"] == manual_label].shape[0]
            print(f"Cluster {annotate_cluster} Manual Label {manual_label} Percentage correct {percentage_correct}")

            if estimate_type == "distance_function":
                distance_df = df["distance_{}".format(annotate_cluster)]
                dist = distance_df.iloc[indices]
                df["manual_annotation_confidence"].iloc[indices] = _manual_confidence * dist_to_conf(dist)
                df["distance_to_confidence"].iloc[indices] = dist_to_conf(dist)
            elif estimate_type == "map":
                confidence_df = df["confidence_{}".format(annotate_cluster)]
                df["manual_annotation_confidence"].iloc[indices] = _manual_confidence * confidence_df.iloc[indices]
            if assign_only_correct:
                wrong_indices = (df["manual_annotation"] == manual_label) & (df["label"] != manual_label)
                print(len(wrong_indices), wrong_indices.shape)
                df["manual_annotation_confidence"].loc[wrong_indices] = 0
        else:
            print("unknown")
            # TODO second level clustering is not used now so commenting the code
            # unknown, check if second level clustering is done or not
            _, cluster = get_cluster(annotate_cluster, cluster_group_dict)
            print(type(cluster.next_level_clusters))
            print(list(cluster.next_level_clusters.keys()))
            print(cluster.id)

            for cluster_group_name, cluster_group in cluster.next_level_clusters.items():
                for _cluster in cluster_group:
                    print(f"Second level cluster id {_cluster.id}")
                    _manual_label = _cluster.manual_annotation.label
                    print(f"********{_manual_label}*******")
                    if isinstance(_manual_label, tuple) or isinstance(_manual_label, list):
                        # TODO add this code
                        pass
                    elif _manual_label != -1:
                        print("Manual_label", _manual_label)
                        assign_label(df, _manual_label)
                    else:
                        # Manual label is -1
                        # Label all the ~600 samples in the second level cluster
                        indices = np.where((np.asarray(cluster_labels) == cluster.id)
                                           & (df[cluster_column_name_2].values == _cluster.id))[0]
                        print(f"Annotating individual samples {indices.shape}")
                        num_individual_samples_annotated += indices.shape[0]
                        df["manual_annotation"].iloc[indices] = df["label"][indices].values
                        df["manual_annotation_confidence"].iloc[indices] = 1
                        if estimate_type =="distance_function":
                            _dist = _distance_df.iloc[indices]
                            df["distance_to_confidence"].iloc[indices] = dist_to_conf(_dist)
        print("********************************")

    return num_individual_samples_annotated


def get_samples_for_cluster(df, cluster_num, cluster_column_name):
    _df = df[df[cluster_column_name] == cluster_num]
    return _df


def get_cluster_groups(manual_labels,
                       manual_confidence,
                       cluster_column_name,
                       cluster_centers,
                       cluster_labels,
                       df,
                       parent_indices=None):
    """
    Create groups of clusters based on manual label given to each cluster. Different categories of cluster group are

    1. impure
    Contains impure clusters. impure cluster is a cluster where the cluster center have similarity with multiple labels,
    each with different confidence
    2. unknown
    The user who annotated the label doesn't know the label, or it is an invalid data for the domain under consideration
    3. similar_labels
    There are multiple clusters in the cluster group all of which has similar label

    4. good_clusters
    There are multiple clusters all with different labels each with a different confidence level greater than  threshold
    4. average_clusters
    There are multiple clusters all with different labels each with a different confidence level around 0.5
    5. low_confidence_clusters
    There are multiple clusters all with different labels each with a different confidence level close to zero
    """
    # annotation_string = "pure_cluster:Cluster number: {}\nCluster center label:{}\n Confidence: {}"

    cluster_groups_dict = dict()
    for cluster_num, cluster_center_label in enumerate(manual_labels):
        if parent_indices is None:
            _df = get_samples_for_cluster(df, cluster_num, cluster_column_name)
            indices = np.where(df[cluster_column_name].values == cluster_num)
        else:
            _df = get_samples_for_cluster(df.iloc[parent_indices], cluster_num, cluster_column_name)
            indices = np.where(df[cluster_column_name].iloc[parent_indices].values == cluster_num)

        cluster_details = {
            "cluster_centers": cluster_centers[cluster_num],
            "cluster_labels": cluster_labels,
            "indices": indices,  # Indices of this cluster elements in parent DataFrame
            "cluster_data_frame": _df,
            "whole_data_frame": df
        }
        if isinstance(cluster_center_label, tuple) or isinstance(cluster_center_label, list):
            # impure cluster
            # create an impure clusterGroup
            manual_annotation = ManualAnnotation(cluster_center_label,
                                                 manual_confidence[cluster_num])
            cluster = Cluster(cluster_id=cluster_num,
                              name=f"cluster_{cluster_num}",
                              cluster_details=cluster_details,
                              level=1,
                              manual_annotation=manual_annotation)
            if "impure_cluster" in cluster_groups_dict.keys():
                cluster_groups_dict["impure_cluster"].add_cluster(cluster)
            else:
                cluster_groups_dict["impure_cluster"] = ClusterGroup("impure_cluster", [cluster])
        elif cluster_center_label == -1:
            # unknown cluster
            manual_annotation = ManualAnnotation(cluster_center_label, manual_confidence[cluster_num])
            cluster = Cluster(cluster_id=cluster_num,
                              name=f"cluster_{cluster_num}",
                              cluster_details=cluster_details,
                              level=1,
                              manual_annotation=manual_annotation
                              )
            if "unknown_cluster" in cluster_groups_dict.keys():
                cluster_groups_dict["unknown_cluster"].add_cluster(cluster)
            else:
                cluster_groups_dict["unknown_cluster"] = ClusterGroup("unknown_cluster", [cluster])
            # unknown cluster
        else:
            # good/average/low confidence
            manual_annotation = ManualAnnotation(cluster_center_label, manual_confidence[cluster_num])
            cluster = Cluster(cluster_id=cluster_num,
                              name="cluster_".format(cluster_num),
                              cluster_details=cluster_details,
                              level=1,
                              manual_annotation=manual_annotation)
            cluster_group_label = manual_annotation.get_label()
            if cluster_group_label in cluster_groups_dict.keys():
                cluster_groups_dict[cluster_group_label].add_cluster(cluster)
            else:
                cluster_groups_dict[cluster_group_label] = ClusterGroup(cluster_group_label,
                                                                        [cluster])
    return cluster_groups_dict


def distance_eu(row,cluster_center,z_col_names):
    return norm(row[z_col_names].values - cluster_center)


def distance(row, inv_cov, cluster_center,z_col_names):
    return mahalanobis(row[z_col_names].values, cluster_center, inv_cov)


def compute_distance(df, num_clusters, cluster_labels, z_col_names, cluster_centers):
    for i in range(num_clusters):
        df["distance_{}".format(i)] = 100000
    for cluster_num in range(num_clusters):
        indices = np.where( np.asarray(cluster_labels) == cluster_num)[0]
        lv = df[z_col_names].values[indices, :]
        print(lv.shape)
        cov = np.cov(lv.T)
        inv_cov = sp.linalg.inv(cov)
        df["distance_{}".format(cluster_num)].iloc[indices] = df.iloc[indices].apply(lambda x:distance(x,
                                                                                                       inv_cov,
                                                                                                       cluster_centers[cluster_num],
                                                                                                       z_col_names),
                                                                                     axis=1)


def compute_distance_level_2(df, num_level_2_clusters, cluster_labels, z_col_names, cluster, cluster_column_name_2):
    for i in range(num_level_2_clusters):
        df[f"distance_level_2_{cluster.id}_{i}"] = 100000
    for cluster_group_label, level_2_cluster_group in cluster.next_level_clusters.items():
        if cluster_group_label == "unknown_cluster":
            print("Skipping distance computation for unknown cluster")
            continue
        for level_2_cluster in level_2_cluster_group:
            indices = np.where((np.asarray(cluster_labels) == cluster.id) &
                               (df[cluster_column_name_2].values == level_2_cluster.id) )[0]
            cluster_centers_level_2 = level_2_cluster.details["cluster_centers"]
            lv= df[z_col_names].values[indices, :]
            cov = np.cov(lv.T)
            inv_cov = sp.linalg.inv(cov)

            print(level_2_cluster.id, indices.shape)
            df[f"distance_level_2_{cluster.id}_{level_2_cluster.id}"].iloc[indices] = df.iloc[indices].apply(lambda x:distance(x,
                                                                                                                               inv_cov,
                                                                                                                               cluster_centers_level_2,
                                                                                                                               z_col_names),
                                                                                                             axis=1)


def get_manual_annotation_col_name(epochs_completed):
    epoch_col_confidence = f"manual_confidence_{int(epochs_completed)}"
    epoch_col_label = f"manual_label_{int(epochs_completed)}"
    return epoch_col_confidence, epoch_col_label

def get_overall_confidence(row, epochs_competed:List[int]):
    if len(epochs_competed) == 1:
        epoch_col_confidence, epoch_col_label = get_manual_annotation_col_name(epochs_competed[0])
        return row[epoch_col_confidence]
    else:
        confidences = [row[get_manual_annotation_col_name(epoch)[0]] for epoch in epochs_competed]
        if max(confidences) == 1:
            return 1
        else:
            return confidences[-1]


def get_overall_label(row, epochs_competed:List[int]):
    if len(epochs_competed) == 1:
        epoch_col_confidence, epoch_col_label = get_manual_annotation_col_name(epochs_competed[0])
        return row[epoch_col_label]
    else:
        confidences = [row[get_manual_annotation_col_name(epoch)[0]] for epoch in epochs_competed]
        labels = [row[get_manual_annotation_col_name(epoch)[1]] for epoch in epochs_competed]

        # argmax of confidences
        max_index = max(enumerate(confidences), key=lambda x: x[1])[0]
        if confidences[max_index] == 1:
            return labels[max_index]
        else:
            return labels[-1]
