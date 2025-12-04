# built-in dependencies
import os
import time
from typing import Optional, List

# FastAPI
from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    Form,
)

# External libs
from elasticsearch import Elasticsearch, NotFoundError

from .routes import (
    perform_represent,
    RepresentParams,
    default_image_max_size,
)


# 替代 logger
class SimpleLogger:
    def info(self, msg):
        print("[INFO]", msg)


logger = SimpleLogger()

# 替代 RuntimeException
RuntimeException = RuntimeError

router = APIRouter(prefix="/store", tags=["store"])

# ============
# ES Client Cache
# ============

es_client = None  # 用于缓存客户端
INDEX_PREFIX = "faces"

_model_name: str = "AdaFace"


# ============
# ES Utilities
# ============

def index_name(model_name: str) -> str:
    """faces_arcface => 小写索引名"""
    return f"{INDEX_PREFIX}_{model_name}".lower()


def create_es_client() -> Elasticsearch:
    """Create ES client based on env variables."""
    es_hosts = os.getenv("ELASTICSEARCH_HOSTS")
    es_password = os.getenv("ELASTIC_PASSWORD")

    if not es_hosts:
        raise RuntimeError("ELASTICSEARCH_HOSTS 未设置")

    return Elasticsearch(
        [es_hosts],
        http_auth=("elastic", es_password),
    )


def get_es(model_name: str, dims: int) -> Elasticsearch:
    """Get ES client and ensure index exists."""
    global es_client
    if es_client is not None:
        return es_client

    client = create_es_client()

    if not client.ping():
        raise RuntimeException("无法连接 Elasticsearch")

    idx = index_name(model_name)

    if not client.indices.exists(index=idx):
        logger.info(f"创建索引：{idx}")

        client.indices.create(
            index=idx,
            body={
                "mappings": {
                    "properties": {
                        "face_vector": {
                            "type": "dense_vector",
                            "dims": dims,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "key": {"type": "keyword"},
                        "extra": {"type": "text"}  # 新增扩展字段（任意长字符串）
                    }
                }
            },
        )

    es_client = client
    return client


# ============
# PUT
# ============

@router.post("/put", summary="存储人脸向量")
async def store_put(
        key: str = Form(..., description="唯一 key"),
        img: Optional[str] = Form(None),
        img_file: Optional[UploadFile] = File(None),
        extra: Optional[str] = Form(None),
        image_max_size: int = Form(default_image_max_size, ge=1),
        max_faces: int = Form(1, ge=1),
):
    """提取人脸 embedding 并存入 ES"""

    file_bytes = await img_file.read() if img_file else None

    params = RepresentParams(
        img=img,
        image_max_size=image_max_size,
        max_faces=max_faces,
    )

    if params.img is None and file_bytes is None:
        raise HTTPException(400, "必须提供 img 或 img_file")

    # 1) 取 embedding
    rep, code = perform_represent(params, file_bytes)
    if code != 200:
        raise HTTPException(code, rep)

    results = rep.get("results", [])
    if not results:
        raise HTTPException(400, "未检测到人脸")

    emb = results[0]["embedding"]
    dims = len(emb)

    # 2) 写入 ES
    es = get_es(_model_name, dims)

    es.index(
        index=index_name(_model_name),
        id=key,
        document={
            "key": key,
            "face_vector": emb,
            "extra": extra,
        },
    )

    return {"ok": True, "key": key}


# ============
# GET
# ============

@router.post("/get", summary="根据 key 获取 embedding")
async def store_get(
        key: str = Form(...),
):
    idx = index_name(_model_name)
    es = get_es(_model_name, 512)

    try:
        res = es.get(index=idx, id=key)
        return res["_source"]  # 自动包含 extra
    except NotFoundError:
        raise HTTPException(404, f"未找到: {key}")


# ============
# DELETE
# ============

@router.post("/delete", summary="删除一个 key")
async def store_delete(
        key: str = Form(...),
):
    idx = index_name(_model_name)
    es = get_es(_model_name, 512)

    try:
        es.delete(index=idx, id=key)
        return {"deleted": key}
    except NotFoundError:
        raise HTTPException(404, f"未找到: {key}")


# ============
# CLEAR
# ============

@router.post("/clear", summary="清空索引（删除全部向量）")
async def store_clear():
    idx = index_name(_model_name)
    es = get_es(_model_name, 512)

    if es.indices.exists(index=idx):
        es.indices.delete(index=idx)

    return {"cleared": True}


# ============
# SIZE
# ============

@router.post("/size", summary="索引向量数量")
async def store_size():
    idx = index_name(_model_name)
    es = get_es(_model_name, 512)

    if not es.indices.exists(index=idx):
        return {"size": 0}

    count = es.count(index=idx)
    return {"size": count["count"]}


# ============
# SEARCH
# ============

@router.post("/search", summary="基于图片搜索相似人脸")
async def store_search(
        max_size: int = Form(1, ge=1),
        img: Optional[str] = Form(None),
        img_file: Optional[UploadFile] = File(None),
        image_max_size: int = Form(default_image_max_size, ge=1),
):
    """提取 embedding → ES knn 搜索"""

    file_bytes = await img_file.read() if img_file else None

    params = RepresentParams(
        img=img,
        image_max_size=image_max_size,
        max_faces=1,
    )

    if params.img is None and file_bytes is None:
        raise HTTPException(400, "必须提供 img 或 img_file")

    # 1) embedding
    t0 = time.time()
    rep, code = perform_represent(params, file_bytes)
    if code != 200:
        raise HTTPException(code, rep)

    results = rep.get("results", [])
    if not results:
        raise HTTPException(400, "未检测到人脸")

    emb = results[0]["embedding"]
    dims = len(emb)

    # 2) knn 搜索
    es = get_es(_model_name, dims)

    body = {
        "knn": {
            "field": "face_vector",
            "query_vector": emb,
            "k": max_size,
            "num_candidates": 100
        },
        "_source": ["key", "extra"],
    }

    t1 = time.time()
    res = es.search(
        index=index_name(_model_name),
        size=max_size,
        body=body,
    )

    return {
        "time": {
            "represent": t1 - t0,
            "search": time.time() - t1,
        },
        "items": [
            {
                "key": hit["_source"]["key"],
                "score": hit["_score"],
                "extra": hit["_source"].get("extra"),
            }
            for hit in res["hits"]["hits"]
        ]
    }
