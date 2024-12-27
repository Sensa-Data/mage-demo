def anomaly_train_kmeans(ts_training_data, window_size=5, n_clusters=2, anomaly_threshold=0.95, model_path=None):
  """
  Train and detect anomalies in a time series using KMeans clustering.

  Parameters:
  - ts_training_data: Input time series data for training
  - window_size: Size of sliding window for creating embeddings
  - n_clusters: Number of clusters for KMeans
  - anomaly_threshold: Percentile threshold for identifying anomalies
  - model_path: Optional path to save the trained model

  Returns:
  - Dictionary containing anomaly indices, windows, cluster labels, and trained model
  """

  import numpy as np
  import pandas as pd
  from sklearn.preprocessing import StandardScaler
  from sklearn.cluster import KMeans
  import joblib

  # Create sliding window embeddings
  embeddings = []
  for i in range(len(ts_training_data) - window_size + 1):
      window = ts_training_data[i:i+window_size]
      embeddings.append(window)

  # Standardize embeddings
  scaler = StandardScaler()
  scaled_embeddings = scaler.fit_transform(embeddings)

  # Perform K-Means Clustering
  kmeans = KMeans(n_clusters=n_clusters, random_state=42)
  cluster_labels = kmeans.fit_predict(scaled_embeddings)  # embeddings, scaled_embeddings

  # Calculate distances to cluster centroids
  distances = kmeans.transform(scaled_embeddings)
  min_distances = distances[np.arange(len(distances)), cluster_labels]

  # Determine anomaly threshold
  threshold = np.percentile(min_distances, anomaly_threshold * 100)

  # Identify anomalies
  anomaly_mask = min_distances > threshold
  anomaly_indices = np.where(anomaly_mask)[0]

  """
  Flagging anomaly data points
  """
  # is_anomaly = np.zeros(len(ts_training_data), dtype=bool)
  # for idx in anomaly_indices:
  #     is_anomaly[idx:idx+window_size] = True

  if model_path:
      joblib.dump({
          'kmeans_model': kmeans,
          'scaler': scaler,
          'window_size': window_size,
          'n_clusters': n_clusters,
          'anomaly_threshold': anomaly_threshold
      }, model_path)
      print(f"model saved to: {model_path}")

  return {
      'anomaly_indices': anomaly_indices,
      'anomaly_windows': [ts_training_data[idx:idx+window_size] for idx in anomaly_indices],
      'cluster_labels': cluster_labels,
      # 'is_anomaly': is_anomaly,
  }


def anomaly_plot_kmeans(ts_data, results):
  import matplotlib.pyplot as plt

  print(f"Number of anomaly windows detected: {len(results['anomaly_indices'])}")
  plt.figure(figsize=(15,6))
  plt.plot(ts_data)
  plt.scatter(results['anomaly_indices'],
              ts_data[results['anomaly_indices']],
              color='red', label='Anomalies')
  plt.title('Time Series Anomaly Detection using K-Means')
  plt.xlabel('Time')
  plt.ylabel('Value')
  plt.legend()
  plt.show()


def anomaly_predict_kmeans_single(ts_value, model_path):

  import numpy as np
  import joblib

  model_metadata = joblib.load(model_path)

  kmeans_model = model_metadata.get('kmeans_model')
  scaler = model_metadata.get('scaler')
  window_size = model_metadata.get('window_size')
  n_clusters = model_metadata.get('n_clusters')
  anomaly_threshold = model_metadata.get('anomaly_threshold')

  print(kmeans_model, scaler, window_size, n_clusters, anomaly_threshold)

  initial_window = np.zeros(4)
  testing_embeddings = np.concatenate([initial_window, [ts_value]])
  ts_testing_data = testing_embeddings.reshape(1, -1)
  print(ts_testing_data)

#   for i in range(len(ts_testing_data) - window_size + 1):
#       window = ts_testing_data[i:i+window_size]
#       testing_embeddings.append(window)

  # Standardize embeddings using the pre-fitted scaler
  testing_scaled_embeddings = scaler.transform(ts_testing_data)

  # Predict cluster labels
  testing_cluster_labels = kmeans_model.predict(testing_scaled_embeddings)

  # Calculate distances to cluster centroids
  testing_distances = kmeans_model.transform(testing_scaled_embeddings)
  testing_min_distances = testing_distances[np.arange(len(testing_distances)), testing_cluster_labels]

  # Determine anomaly threshold
  threshold = np.percentile(testing_min_distances, anomaly_threshold * 100)

  # Identify anomalies
  anomaly_mask = testing_min_distances > threshold
  anomaly_indices = np.where(anomaly_mask)[0]

  """
  Flagging anomaly data points
  """
  is_anomaly = np.zeros(len(ts_testing_data), dtype=bool)
  for idx in anomaly_indices:
      is_anomaly[idx:idx+window_size] = True

  return {
      'anomaly_indices': anomaly_indices,
      'anomaly_windows': [ts_testing_data[idx:idx+window_size] for idx in anomaly_indices],
      'cluster_labels': testing_cluster_labels,
      'distances': testing_min_distances,
      'is_anomaly': is_anomaly,
  }

def anomaly_predict_kmeans(ts_testing_data, model_path):

  import numpy as np
  import joblib

  model_metadata = joblib.load(model_path)

  kmeans_model = model_metadata.get('kmeans_model')
  scaler = model_metadata.get('scaler')
  window_size = model_metadata.get('window_size')
  n_clusters = model_metadata.get('n_clusters')
  anomaly_threshold = model_metadata.get('anomaly_threshold')

  print(kmeans_model, scaler, window_size, n_clusters, anomaly_threshold)

  testing_embeddings = []
  for i in range(len(ts_testing_data) - window_size + 1):
      window = ts_testing_data[i:i+window_size]
      testing_embeddings.append(window)

  # Standardize embeddings using the pre-fitted scaler
  testing_scaled_embeddings = scaler.transform(testing_embeddings)

  # Predict cluster labels
  testing_cluster_labels = kmeans_model.predict(testing_scaled_embeddings)

  # Calculate distances to cluster centroids
  testing_distances = kmeans_model.transform(testing_scaled_embeddings)
  testing_min_distances = testing_distances[np.arange(len(testing_distances)), testing_cluster_labels]

  # Determine anomaly threshold
  threshold = np.percentile(testing_min_distances, anomaly_threshold * 100)

  # Identify anomalies
  anomaly_mask = testing_min_distances > threshold
  anomaly_indices = np.where(anomaly_mask)[0]

  """
  Flagging anomaly data points
  """
  is_anomaly = np.zeros(len(ts_testing_data), dtype=bool)
  for idx in anomaly_indices:
      is_anomaly[idx:idx+window_size] = True

  return {
      'anomaly_indices': anomaly_indices,
      'anomaly_windows': [ts_testing_data[idx:idx+window_size] for idx in anomaly_indices],
      'cluster_labels': testing_cluster_labels,
      'distances': testing_min_distances,
      'is_anomaly': is_anomaly,
  }