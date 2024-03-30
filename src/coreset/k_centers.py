import logging

import torch
from datasets import Dataset
from sklearn.cluster import KMeans
from transformers import PreTrainedModel, PreTrainedTokenizer

from .coreset_utils import get_embeddings, l2_dist

logger = logging.getLogger(__name__)


class FastKMeans:
    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        verbose: int = 0,
        seed: int | None = None,
    ):
        """Fast KMeans clustering algorithm
        Args:
            n_clusters (int): Number of clusters
            max_iter (int): Maximum number of iterations of the k-means algorithm
            mode (str): Distance metric ('euclidean' or 'cosine')
            verbose (int): Verbosity mode
            seed (int, optional): Random seed

        Example:
            >>> kmeans = FastKMeans(n_clusters=10)
            >>> cluster_ids = kmeans.fit_predict(data)
            >>> cluster_center_ids = kmeans.cluster_center_ids(data)
        """

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            max_iter=self.max_iter,
            n_init=1,
            verbose=self.verbose,
            random_state=seed,
        )

    @torch.no_grad()
    def fit_predict(self, data: torch.Tensor) -> tuple[list[int], list[int]]:
        """Clustering data and predict the cluster index for each sample
        Args:
            data (torch.Tensor): Data to cluster
        Returns:
            - cluster_ids (list[int]): Cluster index for each sample
            - cluster_center_ids (list[int]):
                Sample id of the nearest neighbor to cluster center
        """
        assert data.ndim == 2
        assert data.size(0) >= self.n_clusters

        data = data.cpu()

        logger.info("Clustering ...")
        cluster_ids = self.kmeans.fit_predict(data).tolist()
        assert len(cluster_ids) == len(data), f"{len(cluster_ids)} != {len(data)}"

        cluster_size = [cluster_ids.count(i) for i in range(self.n_clusters)]
        logger.info(f"Cluster size: {sorted(cluster_size)}")

        # check if there is empty cluster
        if 0 in cluster_size:
            # raise error if there is empty cluster
            self.empty_cluster_error_handling(data)

        cluster_center_ids = self._get_cluster_center_ids(data, cluster_ids)

        return cluster_ids, cluster_center_ids

    @torch.no_grad()
    def _get_cluster_center_ids(
        self, data: torch.Tensor, cluster_ids: list[int]
    ) -> list[int]:
        """Compute sample id of the nearest neighbor to cluster center"""

        assert len(cluster_ids) == len(data)

        def compute_cluster_center_id(
            data: torch.Tensor, target_cluster_id: int
        ) -> int:
            """Compute sample id of the nearest neighbor to cluster center"""
            assert 0 <= target_cluster_id < self.n_clusters, target_cluster_id
            assert (
                target_cluster_id in cluster_ids
            ), f"{target_cluster_id} not in {cluster_ids}"
            # select samples in the cluster
            cluster_sample_ids = [
                sample_id
                for sample_id, cluster_id in enumerate(cluster_ids)
                if cluster_id == target_cluster_id
            ]
            cluster_data = data[cluster_sample_ids]

            # compute cluster center
            centroid = cluster_data.mean(0)

            # compute distance to cluster center for each sample
            dists = l2_dist(cluster_data, centroid)

            # return sample id of the nearest neighbor to cluster center
            return cluster_sample_ids[torch.argmin(dists)]

        cluster_center_ids = [
            compute_cluster_center_id(data, target_cluster_id)
            for target_cluster_id in range(self.n_clusters)
        ]
        return cluster_center_ids

    def empty_cluster_error_handling(self, data: torch.Tensor):
        """Handling empty cluster error"""

        def count_unique_examples(data: torch.Tensor) -> int:
            unique_data = []
            for d in data:
                if not any(torch.all(d == ud) for ud in unique_data):
                    unique_data.append(d)
            return len(unique_data)

        # check if there is less unique examples than the number of clusters
        if count_unique_examples(data) < self.n_clusters:
            raise RuntimeError(
                "Less unique examples than the number of clusters ({} < {})".format(
                    count_unique_examples(data), self.n_clusters
                )
            )

        raise RuntimeError("Empty cluster was occurred during k-means clustering.")


def k_centers(
    dataset: Dataset,
    dpc: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sentence_keys: list[str],
    seed: int,
) -> Dataset:
    assert len(dataset) >= dpc

    # compute embeddings
    embeddings = get_embeddings(dataset, model, tokenizer, sentence_keys)

    # select k-centers for each label
    kmeans = FastKMeans(n_clusters=dpc, seed=seed)
    _, k_center_samples_ids = kmeans.fit_predict(embeddings)

    assert len(k_center_samples_ids) == dpc
    return dataset.select(k_center_samples_ids)
