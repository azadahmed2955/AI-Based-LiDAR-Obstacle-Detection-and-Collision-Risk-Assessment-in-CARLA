# Clustering Comparison Results

Compared methods:
- PCL Euclidean Cluster Extraction
- Open3D DBSCAN

Dataset:
- 148 LiDAR frames
- Same preprocessed CARLA LiDAR input for both methods

## Results

| Metric | PCL Euclidean | Open3D DBSCAN |
|---|---:|---:|
| Average runtime | 3.211 ms | 2.707 ms |
| Minimum runtime | 0.285 ms | 0.138 ms |
| Maximum runtime | 9.020 ms | 8.168 ms |
| Runtime standard deviation | 2.305 ms | 1.718 ms |
| Average cluster count | 7.94 | 7.55 |
| Cluster-count standard deviation | 1.42 | 1.50 |
| Faster frames | 44 | 104 |
| Average DBSCAN noise points | N/A | 36.40 |
| Average DBSCAN noise percentage | N/A | 10.59% |

The two methods produced the same cluster count in 58 of 148 frames (39.2%).

Preliminary observation:
DBSCAN was faster in this experiment and explicitly identified noise points.
PCL produced a slightly more stable cluster count.

Object-separation quality still needs to be evaluated before deciding which method is more suitable.
