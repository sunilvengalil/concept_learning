from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

def plot_elbow(run_ids):
    for run_id in run_ids:
        latent_vectors = latent_vectors[run_id]
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(1, 20))

        visualizer.fit(latent_vectors)
        visualizer.show(exp_config.ANALYSIS_PATH + "elbow_curve.jpg")