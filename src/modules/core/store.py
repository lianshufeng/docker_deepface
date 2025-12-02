# built-in dependencies
import os
import time
from typing import Optional

"""
向量存储与搜索接口说明（Elasticsearch）
- 路由前缀：/store，标签 store；索引命名 faces_{model_name}（小写）
- 环境变量：ELASTICSEARCH_HOSTS、ELASTIC_PASSWORD；首次使用自动创建 dense_vector 映射

通用提取参数：沿用 /v2/represent（img、image_max_size、detector_backend、model_name、
enforce_detection、align、anti_spoofing、max_faces），请求方式同 represent。

接口一览：
- POST /put    ：提取向量后写入 ES；额外字段 key（必填）
- POST /get    ：Body { key, model_name? } 按 key 取文档；未找到返回 {found: false}
- POST /remove ：Body { key, model_name? } 删除文档；不存在也返回 deleted
- POST /clean  ：Body { model_name? } 清空对应模型索引
- POST /size   ：Body { model_name? } 返回索引文档数量
- POST /search ：请求=提取参数 + max_size（默认 1）；先提取 embedding，再 KNN 搜索，
                返回 { time: {represent, search}, items: [{key, score}, ...] }
错误处理：
- 参数/输入校验失败返回 422；运行期错误统一 400（detail 为错误信息）。
"""

# project dependencies
from deepface.commons.logger import Logger
from elasticsearch import Elasticsearch, NotFoundError
# 3rd party dependencies
from fastapi import APIRouter, Body, HTTPException, Request, UploadFile, File
from pydantic import BaseModel, Field, ValidationError

from .routes import RepresentParams, _coerce_payload, _model_name, perform_represent

logger = Logger()
router = APIRouter(prefix="/store", tags=["store"])

# 创建索引前缀
index_pre_name = "faces"

# 缓存已创建的 ES 客户端
es_client = {}


class StoreKeyPayload(BaseModel):
    key: str = Field(..., description="要存储/获取的唯一文档 ID。")
    model_name: str = Field("ArcFace", description="用于推导索引名的模型名称。")


class StoreModelPayload(BaseModel):
    model_name: str = Field("ArcFace", description="用于推导索引名的模型名称。")


class StorePutParams(RepresentParams):
    key: str = Field(..., description="要存储的唯一文档 ID。")


class StoreSearchParams(RepresentParams):
    max_size: int = Field(1, ge=1, description="返回的最相近人脸数量。")


def indexName(model_name: str) -> str:
    return (index_pre_name + "_" + model_name).lower()


def create_es_client() -> Elasticsearch:
    # 读取环境变量
    es_hosts = os.getenv("ELASTICSEARCH_HOSTS")
    es_password = os.getenv("ELASTIC_PASSWORD")
    # 通过 http 方式连接
    client = Elasticsearch(
        [es_hosts],
        http_auth=("elastic", es_password),
    )
    return client


def get_es_client(model_name: str, dims: int) -> Elasticsearch:
    global es_client
    index_name = indexName(model_name)

    client = es_client.get(index_name)
    if client is not None:
        return client

    client = create_es_client()

    if not client.ping():
        raise ValueError("ES 连接失败。")

    # 注意：首次使用需创建索引
    if not client.indices.exists(index=index_name):
        client.indices.create(
            index=index_name,
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
                        "description": {"type": "text"},
                    }
                }
            },
        )
    es_client[index_name] = client
    return client


@router.post(
    "/put",
    summary="存储人脸向量",
    description="与 /v2/represent 相同的负载提取 embedding 后写入 Elasticsearch。",
)
async def put(
    request: Request,
    payload: Optional[StorePutParams] = Body(
        None, description="非表单上传时的 JSON 负载。"
    ),
    img: Optional[UploadFile] = File(None, description="表单字段 'img' 上传的图片文件。"),
):
    params, file_bytes = await _coerce_payload(request, payload, img, StorePutParams)
    if params.img is None and file_bytes is None:
        raise HTTPException(
            status_code=400,
            detail="请通过文件字段 img 上传图片，或在字段 img 中提供 Base64/路径/URL。",
        )

    try:
        rep, status_code = perform_represent(params, file_bytes)
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=rep)
        embedding = rep["results"][0]["embedding"]

        es = get_es_client(_model_name(params.model_name), len(embedding))
        es.index(
            index=indexName(_model_name(params.model_name)),
            id=params.key,
            document={"face_vector": embedding, "key": params.key},
        )
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))

    return rep


@router.post(
    "/get",
    summary="获取已存储的人脸向量",
    description="按 key 从对应模型索引中获取文档。",
)
async def get(payload: StoreKeyPayload = Body(...)):
    model_name = _model_name(payload.model_name)
    index_name = indexName(model_name)

    try:
        response = create_es_client().get(index=index_name, id=payload.key)
    except NotFoundError:
        return {"found": False}
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))
    return response.body


@router.post(
    "/remove",
    summary="删除已存储的人脸向量",
    description="按 key 从对应模型索引中删除文档。",
)
async def remove(payload: StoreKeyPayload = Body(...)):
    model_name = _model_name(payload.model_name)
    index_name = indexName(model_name)
    try:
        response = create_es_client().delete(index=index_name, id=payload.key)
    except NotFoundError:
        return {"result": "deleted"}
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))
    return response.body


@router.post(
    "/clean",
    summary="清空索引",
    description="删除指定模型索引下的全部文档。",
)
async def clean(payload: StoreModelPayload = Body(...)):
    model_name = _model_name(payload.model_name)
    index_name = indexName(model_name)
    try:
        response = create_es_client().delete_by_query(
            index=index_name,
            body={"query": {"match_all": {}}},
        )
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))
    return response.body


@router.post(
    "/size",
    summary="统计索引文档数",
    description="返回指定模型索引的文档数量。",
)
async def size(payload: StoreModelPayload = Body(...)):
    model_name = _model_name(payload.model_name)
    index_name = indexName(model_name)
    try:
        response = create_es_client().count(index=index_name, body={"query": {"match_all": {}}})
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))
    return response.body


@router.post(
    "/search",
    summary="搜索最近邻人脸向量",
    description="提取 embedding 后执行 KNN 搜索。",
)
async def search(
    request: Request,
    payload: Optional[StoreSearchParams] = Body(
        None, description="非表单上传时的 JSON 负载。"
    ),
    img: Optional[UploadFile] = File(None, description="表单字段 'img' 上传的图片文件。"),
):
    params, file_bytes = await _coerce_payload(request, payload, img, StoreSearchParams)
    if params.img is None and file_bytes is None:
        raise HTTPException(
            status_code=400,
            detail="请通过文件字段 img 上传图片，或在字段 img 中提供 Base64/路径/URL。",
        )

    ret = {"time": {"represent": 0.0, "search": 0.0}, "items": []}
    try:
        recordTime = time.time()
        rep, status_code = perform_represent(params, file_bytes)
        ret["time"]["represent"] = float(time.time() - recordTime)
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=rep)
        embedding = rep["results"][0]["embedding"]

        es = get_es_client(_model_name(params.model_name), len(embedding))
        body = {
            "knn": {
                "field": "face_vector",
                "query_vector": embedding,
                "k": params.max_size,
                "num_candidates": 100,
            },
            "fields": ["_id"],
            "_source": ["key"],
        }
        recordTime = time.time()
        res = es.search(index=indexName(_model_name(params.model_name)), size=params.max_size, body=body)
        ret["time"]["search"] = float(time.time() - recordTime)
        for hit in res["hits"]["hits"]:
            ret["items"].append({"key": hit["_id"], "score": hit["_score"]})
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))

    return ret
