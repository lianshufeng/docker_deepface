import threading
from typing import Optional, Tuple, List

import base64
import cv2
import numpy as np
import requests
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    Request,
    HTTPException
)
from pydantic import BaseModel, Field
from insightface.app import FaceAnalysis


# ============================================
# 路由初始化
# ============================================
router = APIRouter(prefix="/v2", tags=["人脸特征向量接口"])

default_image_max_size = 640
lock = threading.Lock()
_face_app: Optional[FaceAnalysis] = None


# ============================================
# 模型加载：SCRFD + AdaFace
# ============================================
def get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is None:
        app = FaceAnalysis(name="antelopev2")
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
    return _face_app


# ============================================
# 工具：缩放图片
# ============================================
def resize_image(img: np.ndarray, max_size: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img, 1.0

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h)), scale


# ============================================
# 工具：从字符串加载图片
# ============================================
def load_image_from_str(img_value: str) -> Optional[np.ndarray]:
    img_value = img_value.strip()

    # URL
    if img_value.startswith("http://") or img_value.startswith("https://"):
        resp = requests.get(img_value, timeout=5)
        arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Base64
    if "base64," in img_value or len(img_value) > 200:
        if "base64," in img_value:
            img_value = img_value.split("base64,", 1)[1]
        data = base64.b64decode(img_value)
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 本地路径
    return cv2.imread(img_value)


# ============================================
# 入参
# ============================================
class RepresentParams(BaseModel):
    img: Optional[str] = Field(
        default=None,
        description="可选：Base64 / URL / 本地路径；若使用文件上传可为空"
    )
    image_max_size: int = Field(default=default_image_max_size)
    enforce_detection: bool = Field(default=True)
    max_faces: int = Field(default=1, ge=1)


# ============================================
# 读取图片（同步函数！不能是 async）
# ============================================
def read_image(
    img_value: Optional[str],
    file_bytes: Optional[bytes],
    max_size: int
):
    img: Optional[np.ndarray] = None

    if file_bytes:
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif img_value:
        img = load_image_from_str(img_value)

    if img is None:
        raise ValueError("未提供有效图片，请上传 img_file 或提供 Base64/URL/路径")

    return resize_image(img, max_size)


# ============================================
# 核心：提取人脸特征向量
# ============================================
def perform_represent(params: RepresentParams, file_bytes: Optional[bytes]):
    with lock:
        try:
            img, scale = read_image(params.img, file_bytes, params.image_max_size)
        except Exception as exc:
            return {"detail": str(exc)}, 400

        app = get_face_app()
        faces = app.get(img)

        if not faces:
            if params.enforce_detection:
                return {"detail": "未检测到人脸"}, 400
            return {
                "results": [],
                "scale": scale,
                "detector": "SCRFD",
                "model": "AdaFace",
            }, 200

        # 置信度排序
        faces_sorted = sorted(faces, key=lambda f: float(f.det_score), reverse=True)

        results = []
        for idx, face in enumerate(faces_sorted[: params.max_faces]):
            results.append({
                "embedding": face.normed_embedding.tolist(),
                "bbox": [int(x) for x in face.bbox],
                "score": float(face.det_score),
                "face_index": idx,
            })

        return {
            "results": results,
            "scale": scale,
            "detector": "SCRFD",
            "model": "AdaFace",
        }, 200


# ============================================
# API：/v2/represent
# ============================================
@router.post(
    "/represent",
    summary="【人脸特征向量提取】",
    description=(
        "使用 SCRFD + AdaFace 提取 512 维人脸向量。\n"
        "适用于监控、模糊、偏角、光照不稳定等真实场景。\n"
        "支持：文件上传 + Base64 + URL + 本地路径。"
    )
)
async def represent(
    request: Request,

    img: Optional[str] = Form(default=None),
    img_file: Optional[UploadFile] = File(
        default=None,
        description="上传图片文件（推荐）"
    ),

    image_max_size: int = Form(default_image_max_size),
    enforce_detection: bool = Form(True),
    max_faces: int = Form(1),
):
    file_bytes = await img_file.read() if img_file else None

    params = RepresentParams(
        img=img,
        image_max_size=image_max_size,
        enforce_detection=enforce_detection,
        max_faces=max_faces,
    )

    result, code = perform_represent(params, file_bytes)

    if code != 200:
        raise HTTPException(status_code=code, detail=result)

    return result
