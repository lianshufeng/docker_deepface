# built-in dependencies
import threading
from typing import Any, Dict, Optional, Tuple

"""
核心人脸特征接口说明
- 路由前缀：/v2，标签 deepface
- POST /represent：上传图片或提供 Base64/路径/URL 提取人脸向量；
  支持 detector_backend、model_name、enforce_detection、align、anti_spoofing、
  image_max_size（超限等比缩放）、max_faces 等参数。
输入形式：
- multipart/form-data：表单字段名 img 上传文件
- application/json：字段 img 传 Base64/文件路径/URL，其余参数同上
返回：
- 提取结果字典；若为 dict，附加 scale（缩放比例）、detector_backend、model_name。
错误：
- 参数校验失败返回 422；未提供图片或处理异常返回 400（detail 为错误信息）。
"""

import cv2
import numpy as np
# project dependencies
from deepface.api.src.modules.core import service
from deepface.commons.image_utils import load_image
from deepface.commons.logger import Logger
# 3rd party dependencies
from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, ValidationError

logger = Logger()
router = APIRouter(prefix="/v2", tags=["deepface"])

# 设置最大图片尺寸
default_image_max_size = 640

# 创建全局锁保证底层模型线程安全
lock = threading.Lock()


class RepresentParams(BaseModel):
    img: Optional[str] = Field(
        None,
        description="非上传文件时的图像输入：Base64、文件路径或 URL。",
    )
    image_max_size: int = Field(
        default_image_max_size,
        ge=1,
        description="处理后图像的最大宽高，超过时按长边等比缩放。",
    )
    detector_backend: str = Field("yunet", description="人脸检测后端，如 yunet、opencv。")
    model_name: str = Field("ArcFace", description="DeepFace 模型名称，默认 ArcFace。")
    enforce_detection: bool = Field(True, description="未检测到人脸时是否抛错。")
    align: bool = Field(True, description="提取向量前是否对人脸进行对齐。")
    anti_spoofing: bool = Field(False, description="是否启用防伪检测。")
    max_faces: int = Field(1, ge=1, description="最大处理人脸数量。")


def _model_name(value: Optional[str]) -> str:
    return (value or "ArcFace")


def imageCode(image: np.ndarray, image_max_size: int) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if height <= image_max_size and width <= image_max_size:
        return image, 1

    scale = image_max_size / width if width > height else image_max_size / height
    new_width = int(width * scale)
    new_height = int(height * scale)

    image = cv2.resize(image, (new_width, new_height))
    return image, scale


async def _coerce_payload(
    request: Request,
    payload: Optional[BaseModel],
    file: Optional[UploadFile],
    model_cls,
) -> Tuple[BaseModel, Optional[bytes]]:
    """
    同时支持 JSON 与 multipart/form-data。
    - JSON 直接由 payload 解析
    - 表单上传兼容旧版 Flask 的字段解析方式
    """
    file_bytes = await file.read() if file else None

    if payload is not None:
        return payload, file_bytes

    content_type = request.headers.get("content-type", "")
    data: Dict[str, Any] = {}

    if "multipart/form-data" in content_type:
        form = await request.form()
        for key, value in form.items():
            if isinstance(value, UploadFile) and key == "img" and file is None:
                file = value
            else:
                data[key] = value
        if file_bytes is None and isinstance(file, UploadFile):
            file_bytes = await file.read()
    else:
        try:
            data = await request.json()
        except Exception:
            data = {}

    try:
        params = model_cls.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())

    return params, file_bytes


def _extract_image(
    img_value: Optional[str], file_bytes: Optional[bytes], image_max_size: int
) -> Tuple[np.ndarray, float]:
    if file_bytes:
        image_array = np.frombuffer(file_bytes, dtype=np.uint8)
        img, scale = imageCode(cv2.imdecode(image_array, cv2.IMREAD_COLOR), image_max_size)
        if img is None:
            raise ValueError("上传的图片无效或为空。")
        return img, scale

    if img_value:
        buf, _ = load_image(img_value)
        img, scale = imageCode(buf, image_max_size)
        if img is None:
            raise ValueError("图片未检测到人脸。")
        return img, scale

    raise ValueError("未提供图片，请通过文件字段 img 或字段 img 的 Base64/路径/URL 传入。")


def perform_represent(params: RepresentParams, file_bytes: Optional[bytes]):
    with lock:
        img, scale = _extract_image(params.img, file_bytes, params.image_max_size)

        obj = service.represent(
            img_path=img,
            model_name=_model_name(params.model_name),
            detector_backend=params.detector_backend,
            enforce_detection=bool(params.enforce_detection),
            align=bool(params.align),
            anti_spoofing=bool(params.anti_spoofing),
            max_faces=int(params.max_faces),
        )

        logger.debug(obj)

        # 补充上下文字段，保持与原有输出兼容
        if not isinstance(obj, tuple):
            obj["scale"] = scale
            obj["detector_backend"] = params.detector_backend
            obj["model_name"] = _model_name(params.model_name)

        return obj, 200


@router.post(
    "/represent",
    summary="生成人脸向量",
    description="支持 multipart/form-data 或 JSON。表单字段 'img' 上传文件，或在 JSON/表单字段 'img' 传 Base64/路径/URL。",
)
async def represent(
    request: Request,
    payload: Optional[RepresentParams] = Body(
        None, description="非表单上传时的 JSON 负载。"
    ),
    img: Optional[UploadFile] = File(None, description="表单字段 'img' 上传的图片文件。"),
):
    params, file_bytes = await _coerce_payload(request, payload, img, RepresentParams)
    if params.img is None and file_bytes is None:
        raise HTTPException(
            status_code=400,
            detail="请通过文件字段 img 上传图片，或在字段 img 中提供 Base64/路径/URL。",
        )

    try:
        result, status_code = perform_represent(params, file_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=result)

    return result
