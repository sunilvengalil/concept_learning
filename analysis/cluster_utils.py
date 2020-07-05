from sklearn.cluster import KMeans
import tensorflow as tf
import numpy as np
from generative_models.vae import VAE
from analysis.encode_decode import decode
import math
from matplotlib import pyplot as plt


def cluster_and_decode_latent_vectors(num_clusters, latent_vectors, exp_config):
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    cluster_centers = kmeans.cluster_centers_
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
        z = np.zeros([len(cluster_centers) * num_clusters, exp_config.Z_DIM])

        for i in range(cluster_centers.shape[0]):
            z[i, :] = cluster_centers[i]

        decoded_images = decode(model, z, exp_config.BATCH_SIZE)
        return decoded_images, cluster_centers, cluster_labels


def display_cluster_center_images(decoded_images,
                                  image_filename,
                                  cluster_centers,
                                  exp_config,
                                  run_id):
    colormap = "Greys"
    fig = plt.figure()
    fig.tight_layout()
    num_cols = 4
    num_clusters = cluster_centers.shape[0]
    num_rows = math.ceil(num_clusters / num_cols)
    fig.suptitle(
        f"Decoded Cluster Centers. \nClustered the latent vectors of training set N_3={exp_config.num_units[2]}"
        f" z_dim={exp_config.Z_DIM} run_id={run_id + 1}")
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
                                       convert_distance_to_confidence,
                                       cluster_group_dict):
    df["manual_annotation"] = np.ones(df.shape[0]) * -1
    df["manual_annotation_confidence"] = np.zeros(df.shape[0])
    df["distance_to_confidence"] = np.zeros(df.shape[0])
    manual_labels = manual_annotation_dict["manual_labels"]
    cluster_labels = np.asarray(manual_annotation_dict["cluster_labels"])

    num_clusters = len(manual_labels)
    for i in range(num_clusters):
        annotate_cluster = i
        distance_df = df["distance_{}".format(annotate_cluster)]
        manual_label = manual_labels[annotate_cluster]
        _manual_confidence = manual_annotation_dict["manual_confidence"][annotate_cluster]
        if isinstance(manual_label, tuple) or isinstance(manual_label, list):
            # TODO write code to handle this
            pass
        if manual_label != -1:
            print("Manual Label", manual_label)
            indices = np.where(cluster_labels == annotate_cluster)
            df["manual_annotation"].iloc[indices] = manual_label
            _, cluster = get_cluster(annotate_cluster, cluster_group_dict)
            print(df[df["manual_annotation"] == manual_label].shape, cluster.details["cluster_data_frame"].shape)
            num_correct = df[(manual_label == df["manual_annotation"]) & (df["label"] == manual_label)].shape[0]
            print("Num correct={}".format(num_correct))

            percentage_correct = 100 * num_correct / df[df["manual_annotation"] == manual_label].shape[0]
            print(f"Cluster {i} Manual Label {manual_label} Percentage correct {percentage_correct}")
            dist = distance_df.iloc[indices]
            df["manual_annotation_confidence"].iloc[indices] = _manual_confidence * convert_distance_to_confidence(dist)
            df["distance_to_confidence"].iloc[indices] = convert_distance_to_confidence(dist)
        else:
            # TODO write code for unknown
            pass
        print("********************************")
