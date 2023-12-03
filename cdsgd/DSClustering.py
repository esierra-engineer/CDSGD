from typing import Union
import subprocess
import sys
import ClusteringSelector as cs


subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/Sergio-P/DSGD.git"])
from dsgd.DSClassifierMultiQ import DSClassifierMultiQ

class DSClustering(DSClassifierMultiQ):
    """
    this class is in charge of de clustering of the data
    """
    def __init__(self, data, cluster:  Union[int, None] = None,
                 most_voted: bool = False, min_iter: int = 50,
                 max_iter: int = 400, debug_mode: bool = True,
                 lossfn: str = "MSE", num_workers: int = 0,
                 min_dloss: float = 1e-7):
        self.data = data
        selector = cs.ClusteringSelector(self.data, cluster)
        selector.select_best_clustering()
        self.cluster_labels_df = selector.get_cluster_labels_df()
        if not most_voted:
            self.best = selector.get_best_labels()
        else:
            self.best = selector.get_most_voted()        
        self.cluster = cluster if cluster is not None else len(set(self.best))
        super().__init__(self.cluster, min_iter=min_iter,
                         max_iter=max_iter, debug_mode=debug_mode,
                         lossfn="lossfn", num_workers=num_workers,
                         min_dloss=min_dloss)
