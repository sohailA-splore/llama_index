"""Vespa vector store index.

An index that is built within Vespa.

"""
import logging
from typing import List, Any, Optional, Dict, Callable, Union
import json

from llama_index.vector_stores.utils import metadata_dict_to_node
from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

from llama_index.schema import BaseNode

from vespa.io import VespaResponse, VespaQueryResponse
from vespa.application import Vespa
from vespa.exceptions import VespaError

logger = logging.getLogger(__name__)


def _to_milvus_filter(standard_filters: MetadataFilters) -> List[str]:
    """Translate standard metadata filters to Milvus specific spec."""
    filters = []
    for filter in standard_filters.legacy_filters():
        if isinstance(filter.value, str):
            filters.append(str(filter.key) + " == " + '"' + str(filter.value) + '"')
        else:
            filters.append(str(filter.key) + " == " + str(filter.value))
    return filters


class VespaVectorStore(VectorStore):
    """The Vespa Vector Store.

    In this vector store we store the text, its embedding and
    a its metadata in a Milvus collection. This implementation
    allows the use of an already existing collection.
    It also supports creating a new one if the collection doesn't
    exist or if `overwrite` is set to True.

    Args:
        uri (str, optional): The URI to connect to, comes in the form of
            "http://address:port".
        token (str, optional): The token for log in. Empty if not using rbac, if
            using rbac it will most likely be "username:password".
        collection_name (str, optional): The name of the collection where data will be
            stored. Defaults to "llamalection".
        dim (int, optional): The dimension of the embedding vectors for the collection.
            Required if creating a new collection.
        embedding_field (str, optional): The name of the embedding field for the
            collection, defaults to DEFAULT_EMBEDDING_KEY.
        doc_id_field (str, optional): The name of the doc_id field for the collection,
            defaults to DEFAULT_DOC_ID_KEY.
        similarity_metric (str, optional): The similarity metric to use,
            currently supports IP and L2.
        consistency_level (str, optional): Which consistency level to use for a newly
            created collection. Defaults to "Strong".
        overwrite (bool, optional): Whether to overwrite existing collection with same
            name. Defaults to False.
        text_key (str, optional): What key text is stored in in the passed collection.
            Used when bringing your own collection. Defaults to None.
        index_config (dict, optional): The configuration used for building the
            Milvus index. Defaults to None.
        search_config (dict, optional): The configuration used for searching
            the Milvus index. Note that this must be compatible with the index
            type specified by `index_config`. Defaults to None.

    Raises:
        ImportError: Unable to import `pymilvus`.
        MilvusException: Error communicating with Milvus, more can be found in logging
            under Debug.

    Returns:
        MilvusVectorstore: Vectorstore that supports add, delete, and query.
    """
    stores_text: bool = True
    stores_node: bool = True

    def __init__(
            self,
            vespa_url: str,
            vespa_schema_name: str,
            vespa_namespace_name: str,
            max_queue_size: int = 41,
            vespa_cloud_secret_token: str = None,
            vespa_text_field: str = "name",
            vespa_embedding_gen_enabled: bool = False,
            vespa_embedding_field: str = "embedding",
            vespa_rank_profile: str = "default",
            vespa_embedding_dim: int = 384,
            vespa_doc_name: str = "doc",
            max_workers: int = 2,
            max_connections: int = 10) -> None:
        """Init params."""

        if vespa_cloud_secret_token is not None:
            self.vespa_app = Vespa(url=vespa_url, vespa_cloud_secret_token=vespa_cloud_secret_token)
        else:
            self.vespa_app = Vespa(url=vespa_url)

        assert 200 == self.vespa_app.get_application_status().status_code
        logger.debug("Vespa client connection with the Database")

        self.vespa_schema_name = vespa_schema_name
        self.vespa_namespace_name = vespa_namespace_name
        self.vespa_doc_name = vespa_doc_name
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        self.max_connections = max_connections
        self.vespa_text_field = vespa_text_field
        self.vespa_embedding_gen_enabled = vespa_embedding_gen_enabled
        self.vespa_embedding_field = vespa_embedding_field
        self.vespa_rank_profile = vespa_rank_profile
        self.vespa_embedding_dim = vespa_embedding_dim

    @property
    def get_client(self):
        return self.vespa_app

    def query(
            self,
            query: VectorStoreQuery,
            custom_args: Optional[Callable[[Dict, Union[VectorStoreQuery, None]], Dict]] = None,
            vespa_filter: Optional[List[Dict]] = None,
            **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by
            node_ids (Optional[List[str]]): list of node_ids to filter by
            output_fields (Optional[List[str]]): list of fields to return
            embedding_field (Optional[str]): name of embedding field
        """
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(f"Vespa does not support {query.mode} yet.")

        yql = f"select * from {self.vespa_schema_name} where userQuery() or " \
              f"({{targetHits:50,descending:true}}nearestNeighbor(embedding, e)) " \
              f"limit {int(query.similarity_top_k * 1.5)}"

        vespa_body = {
            "yql": yql,
            "query": query.query_str,
            "input.query(e)": query.query_embedding,
            "ranking.profile": self.vespa_rank_profile
        }

        if isinstance(custom_args, Dict):
            vespa_body = {
                "yql": custom_args.get("yql", yql),
                "query": query.query_str,
                "ranking.profile":  custom_args.get("ranking.profile", self.vespa_rank_profile)
            }
        with self.vespa_app.syncio() as session:
            response: VespaQueryResponse = session.query(body=vespa_body)
            if not response.is_successfull():
                raise ValueError(
                    f"Query request failed: {response.status_code}, response payload: {response.get_json()}")

        nodes = []
        similarities = []
        ids = []

        for hit in response.hits:
            if len(nodes) < query.similarity_top_k:
                node = metadata_dict_to_node(
                    {
                        "_node_content": json.dumps(hit["fields"]),
                        "_node_type": hit["fields"].get("sddocname", None)
                    }
                )
                nodes.append(node)
                similarities.append(hit['relevance'])
                ids.append(hit['id'])

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def add(self, nodes: List[BaseNode], **add_kwargs: Any, ) -> List[str]:
        """Add nodes to index."""

        def callback(response: VespaResponse, id: str):
            if not response.is_successfull():
                logger.error(f"Error when feeding document {id}: {response.get_json()}")
                raise VespaError(f"Error when feeding document {id}: {response.get_json()}")

        feed_data = [dict(fields={self.vespa_text_field: _node.text,
                                  self.vespa_embedding_field: _node.embedding
                                  }, id=f"{_node.id_}_{_node.hash}") for _node in nodes]

        self.vespa_app.feed_iterable(iter(feed_data), schema=self.vespa_schema_name,
                                     namespace=self.vespa_namespace_name,
                                     max_workers=self.max_workers, max_connections=self.max_connections,
                                     max_queue_size=self.max_queue_size,
                                     callback=callback)
        logger.info("Nodes added to Vespa")
        return [i['id'] for i in feed_data]

    def _check_node_in_vespa(self, node_id):
        data_id = f"id:{self.vespa_namespace_name}:{self.vespa_doc_name}::{node_id}"
        response = self.vespa_app.get_data(schema=self.vespa_schema_name,
                                           namespace=self.vespa_namespace_name,
                                           data_id=data_id,
                                           raise_on_not_found=False)

        if response.is_successfull():
            logger.info(f"Node exists in Vespa Index with data_id: {data_id}")
            return True
        logger.info(f"Unable to get Node in Vespa Index with data_id: {data_id}")
        return False

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with node_id.

        Args:
            node_id: str

        """
        if self._check_node_in_vespa(node_id):
            logger.debug(f"Started Delete Operation in Vespa Index for node_id: {node_id}")
            self.vespa_app.delete_data(schema=self.vespa_schema_name, data_id=node_id,
                                       namespace=self.vespa_namespace_name)
            assert False == self._check_node_in_vespa(node_id)
            logger.info(f"Successfully deleted Node from Vespa Index with node_id: {node_id}")
