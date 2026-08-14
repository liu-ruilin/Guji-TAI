import atexit
import json
import logging
import threading
from pathlib import Path

from FlagEmbedding import FlagModel
from pymilvus import MilvusClient
from tqdm import tqdm

from data.chain_of_thought.dataset_process import name2dataset_module

logger = logging.getLogger(__name__)

temp_dir = Path(__file__).parent.parent / "util" / "tmp"
temp_dir.mkdir(exist_ok=True)


class Interpret_vectordb:
    _global_lock = threading.Lock()
    _bge_model = None
    _clients = {}

    def __init__(self, settingname):
        with open(settingname, "r", encoding="utf-8") as f:
            self.setting = json.load(f)

        self.collection_name = "Interpret_lm"
        self.db_uri = str(temp_dir / self.setting["vectordb_path"])
        self._op_lock = threading.Lock()

        with self._global_lock:
            if self.__class__._bge_model is None:
                self.__class__._bge_model = FlagModel(
                    self.setting["embedding_model_path"], use_fp16=False
                )
            self.bge_model = self.__class__._bge_model

            if self.db_uri not in self.__class__._clients:
                self.__class__._clients[self.db_uri] = self._new_client()
            self.client = self.__class__._clients[self.db_uri]
        self.embedding_dim = self._infer_embedding_dim()
        self._ensure_collection_ready()

    def _new_client(self):
        # Keep one persistent client per db uri to avoid repeated grpc channel churn.
        return MilvusClient(self.db_uri)

    @staticmethod
    def _normalize_embedding(embeddings):
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        if isinstance(embeddings, tuple):
            embeddings = list(embeddings)
        if (
            isinstance(embeddings, list)
            and len(embeddings) == 1
            and isinstance(embeddings[0], (list, tuple))
        ):
            embeddings = embeddings[0]
        return embeddings

    def _encode_text(self, text):
        return self._normalize_embedding(self.bge_model.encode(text))

    def _infer_embedding_dim(self):
        configured_dim = self.setting.get("emb_dim")
        embeddings = self._encode_text("dimension probe")
        inferred_dim = len(embeddings)
        if configured_dim and int(configured_dim) != inferred_dim:
            logger.warning(
                "Configured emb_dim=%s does not match encoder output dimension=%s; using inferred dimension.",
                configured_dim,
                inferred_dim,
            )
        return inferred_dim

    def _get_collection_dimension(self):
        try:
            info = self.client.describe_collection(collection_name=self.collection_name)
        except Exception:
            logger.exception("Failed to describe collection %s", self.collection_name)
            return None

        if isinstance(info, dict):
            for key in ("dimension", "dim"):
                value = info.get(key)
                if value is not None:
                    return int(value)

            fields = info.get("fields") or info.get("schema", {}).get("fields") or []
            for field in fields:
                if field.get("name") != "vector":
                    continue
                params = field.get("params", {})
                if "dim" in params:
                    return int(params["dim"])
                if "dimension" in field:
                    return int(field["dimension"])
        return None

    def _create_collection(self):
        logger.info("---Creating vectordb---")
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_dim,
            metric_type="IP",
            consistency_level="Strong",
            auto_id=True,
        )
        self.emb_dataset()

    def _ensure_collection_ready(self):
        if not self.client.has_collection(collection_name=self.collection_name):
            self._create_collection()
            return

        existing_dim = self._get_collection_dimension()
        if existing_dim is not None and existing_dim != self.embedding_dim:
            logger.warning(
                "Existing vectordb dimension=%s does not match encoder output dimension=%s; rebuilding collection.",
                existing_dim,
                self.embedding_dim,
            )
            self.client.drop_collection(collection_name=self.collection_name)
            self._create_collection()
            return

        logger.info("---Vectordb already exists!---")

    def _refresh_client(self):
        with self._global_lock:
            old = self.__class__._clients.get(self.db_uri)
            try:
                if old is not None:
                    old.close()
            except Exception:
                pass
            self.__class__._clients[self.db_uri] = self._new_client()
            self.client = self.__class__._clients[self.db_uri]

    @staticmethod
    def _is_transient_grpc_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "too_many_pings" in msg
            or "goaway" in msg
            or "enhance_your_calm" in msg
            or "grpc_status:14" in msg
            or "unavailable" in msg
        )

    def _run_with_retry(self, op):
        try:
            return op()
        except Exception as exc:
            if not self._is_transient_grpc_error(exc):
                raise
            logger.warning("Milvus transient grpc error; recreating client and retrying once: %s", exc)
            self._refresh_client()
            return op()

    def add_data(self, dataset_info, data):
        text = self.build_search_text(data)
        embeddings = self._encode_text(text)
        row = {"vector": embeddings, "text": text, "data": data, "info": dataset_info["name"]}

        def _insert():
            return self.client.insert(collection_name=self.collection_name, data=row)

        with self._op_lock:
            self._run_with_retry(_insert)

    @staticmethod
    def build_search_text(data):
        ordered_keys = [
            "prompt",
            "question",
            "context",
            "source_text",
            "task_aware_target_tokens",
            "task_type",
            "sub_task",
            "domain",
        ]
        parts = []
        for key in ordered_keys:
            value = data.get(key, "")
            if value is not None and str(value).strip():
                parts.append(f"{key}: {value}")
        return "\n".join(parts)

    def search(self, query):
        embeddings = self._encode_text(query)

        def _search():
            return self.client.search(
                collection_name=self.collection_name,
                data=[embeddings],
                limit=self.setting["topk"],
                search_params={"metric_type": "IP", "params": {}},
                output_fields=["text", "data", "info"],
            )

        with self._op_lock:
            return self._run_with_retry(_search)

    def emb_dataset(self):
        for k, module in name2dataset_module.items():
            if k in ["GPT4o_data", "USEREDITINPUT"]:
                continue
            dataset = module.get_default_dataset()
            min_len = min(len(dataset), self.setting["max_num_pre_set"])
            for data in tqdm(dataset[:min_len], desc=f"Embedding dataset {k}"):
                self.add_data(
                    dataset_info=module.dataset_info,
                    data=module.get_processed_kvs(sample=data, keys=module.support_template_keys),
                )

    @classmethod
    def close_all_clients(cls):
        with cls._global_lock:
            for client in cls._clients.values():
                try:
                    client.close()
                except Exception:
                    pass
            cls._clients.clear()


atexit.register(Interpret_vectordb.close_all_clients)


if __name__ == "__main__":
    data = Interpret_vectordb("/gemini/data-3/lrl/interp/Know-MRI/embedding_utils/embedding_setting.json")
