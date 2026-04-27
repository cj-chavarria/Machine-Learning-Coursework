import numpy as np
from scipy.spatial import KDTree


class DBSCAN():
    def __init__(self, epsilon: float, min_pts: int):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.seen_points = set()
        self.current_cluster = 0
        self.labels = np.empty(0)
    
    def _is_core(self, neighbors_list):
        return len(neighbors_list) >= self.min_pts
    
    def _expand_cluster(self, queue_idx, queue, labels, neighbors):
        while queue_idx < len(queue):
            point_idx = queue[queue_idx]
            
            if labels[point_idx] == -1:
                labels[point_idx] = self.current_cluster
                
            if not point_idx in self.seen_points:
                self.seen_points.add(point_idx)
                
                if self._is_core(neighbors[point_idx]):
                    labels[point_idx] = self.current_cluster
                    queue.extend(neighbors[point_idx])
            
            queue_idx += 1 
    
    def _was_seen(self, point_idx):
        if point_idx in self.seen_points:
            return True
        self.seen_points.add(point_idx)
        return False
    
    def fit(self, X):
        tree = KDTree(X)
        neighbors = tree.query_ball_point(X, r=self.epsilon)
        self.labels = np.full(len(X), -1)
        
        for i, neighbors_list in enumerate(neighbors):
            if not self._was_seen(i):
                if self._is_core(neighbors_list):
                    queue = list()
                    self.labels[i] = self.current_cluster
                    queue.extend(neighbors_list)
                    queue_idx = 0
                    
                    self._expand_cluster(queue_idx=queue_idx, queue=queue,
                                         labels=self.labels, neighbors=neighbors)
                    
                    self.current_cluster += 1
        
        return self.labels
