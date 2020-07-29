from sklearn.cluster import KMeans
import tensorflow as tf
import numpy as np
from clearn.models.generative_models.vae import VAE
from clearn.analysis.encode_decode import decode
import math
from matplotlib import pyplot as plt
from clearn.analysis import Cluster
from clearn.analysis import ClusterGroup
from clearn.analysis import ManualAnnotation


def decode_latent_vectors(cluster_centers, exp_config):
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = VAE(sess,
                    epoch=1,
                    batch_size=exp_config.BATCH_SIZE,
                    z_dim=exp_config.Z_DIM,
                    dataset_name=exp_config.dataset_name,
                    beta=exp_config.beta,
                    num_units_in_layer=exp_config.num_units,
                    log_dir=exp_config.LOG_PATH,
                    checkpoint_dir=exp_config.TRAINED_MODELS_PATH,
                    result_dir=exp_config.PREDICTION_RESULTS_PATH
                    )
        z = np.zeros([cluster_centers.shape[0], exp_config.Z_DIM])

        for i in range(cluster_centers.shape[0]):
            z[i, :] = cluster_centers[i]
        decoded_images = decode(model, z, exp_config.BATCH_SIZE)
        return decoded_images


def cluster_and_decode_latent_vectors(num_clusters, latent_vectors, exp_config):
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    cluster_centers = kmeans.cluster_centers_
    decoded_images = decode_latent_vectors(cluster_centers, exp_config)

    return decoded_images, cluster_centers, cluster_labels


def display_cluster_center_images(decoded_images,
                                  image_filename,
                                  cluster_centers
                                  ):
    colormap = "Greys"
    fig = plt.figure()
    fig.tight_layout()
    num_cols = 4
    num_clusters = cluster_centers.shape[0]
    num_rows = math.ceil(num_clusters / num_cols)
    # fig.suptitle(
    #     f"Decoded Cluster Centers. \nClustered the latent vectors of training set N_3={exp_config.num_units[2]}"
    #     f" z_dim={exp_config.Z_DIM} run_id={run_id + 1}")
    for i in range(cluster_centers.shape[0]):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(np.squeeze(decoded_images[i]), cmap=colormap)
    plt.savefig(image_filename,
                bbox="tight",
                pad_inches=0)


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
                                       cluster_column_name_2
                                       ):
    df["manual_annotation"] = np.ones(df.shape[0]) * -1
    df["manual_annotation_confidence"] = np.zeros(df.shape[0])
    df["distance_to_confidence"] = np.zeros(df.shape[0])
    manual_labels = manual_annotation_dict["manual_labels"]
    cluster_labels = np.asarray(manual_annotation_dict["cluster_labels"])

    num_clusters = len(manual_labels)
    for annotate_cluster in range(num_clusters):
        distance_df = df["distance_{}".format(annotate_cluster)]
        manual_label = manual_labels[annotate_cluster]
        _manual_confidence = manual_annotation_dict["manual_confidence"][annotate_cluster]
        if isinstance(manual_label, tuple) or isinstance(manual_label, list):
            _, cluster = get_cluster(annotate_cluster, cluster_group_dict)
            for _cluster in cluster.next_level_clusters["good_clusters"]:
                _distance_df = df[f"distance_level_2_{cluster.id}_{_cluster.id}"]
                _manual_label = _cluster.manual_annotation.label
                if isinstance(_manual_label, tuple) or isinstance(_manual_label, list):
                    # TODO add this code
                    pass
                elif _manual_label != -1:
                    indices = np.where((np.asarray(cluster_labels) == cluster.id)
                                       & (df[cluster_column_name_2].values == _cluster.id))[0]
                    df["manual_annotation"].iloc[indices] = _manual_label
                    _dist = _distance_df.iloc[indices]
                    df["manual_annotation_confidence"].iloc[indices] = _cluster.manual_annotation.confidence * dist_to_conf(_dist)
                    df["distance_to_confidence"].iloc[indices] = dist_to_conf(_dist)
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
            dist = distance_df.iloc[indices]
            df["manual_annotation_confidence"].iloc[indices] = _manual_confidence * dist_to_conf(dist)
            df["distance_to_confidence"].iloc[indices] = dist_to_conf(dist)
        else:
            print("unknown")
            # unknown, check if second level clustering is done or not
            _, cluster = get_cluster(annotate_cluster, cluster_group_dict)
            print(type(cluster.next_level_clusters))
            print(list(cluster.next_level_clusters.keys()))

            for cluster_group_name, cluster_group in cluster.next_level_clusters.items():
                for _cluster in cluster_group:
                    _distance_df = df[f"distance_level_2_{cluster.id}_{_cluster.id}"]
                    _manual_label = _cluster.manual_annotation.label
                    if isinstance(_manual_label, tuple) or isinstance(_manual_label, list):
                        # TODO add this code
                        pass
                    elif _manual_label != -1:
                        print("Manual_label", manual_label)
                        indices = np.where((np.asarray(cluster_labels) == cluster.id)
                                           & (df[cluster_column_name_2].values == _cluster.id))[0]
                        df["manual_annotation"].iloc[indices] = _manual_label
                        _dist = _distance_df.iloc[indices]
                        df["manual_annotation_confidence"].iloc[
                            indices] = _cluster.manual_annotation.confidence * dist_to_conf(_dist)
                        df["distance_to_confidence"].iloc[indices] = dist_to_conf(_dist)
        print("********************************")


def get_samples_for_cluster(df, cluster_num, cluster_column_name):
    _df = df[df[cluster_column_name] == cluster_num]
    return _df


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
There are multiple clusters all with different labels each with a different confidence level greater than a threshold
4. average_clusters
There are multiple clusters all with different labels each with a different confidence level around 0.5
5. low_confidence_clusters
There are multiple clusters all with different labels each with a different confidence level close to zero
"""
# TODO check how to do this with parameters to constructer
# annotation_string = "pure_cluster:Cluster number: {}\nCluster center label:{}\n Confidence: {}"


def get_cluster_groups(manual_labels,
                       manual_confidence,
                       cluster_column_name,
                       cluster_centers,
                       cluster_labels,
                       df,
                       parent_indices=None):

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
