# PUBG Item Recognition API Documentation

RESTful API服务，用于识别PUBG游戏截图中的物品并返回去重后的物品代码。

## 快速开始

### 使用Docker部署（推荐）

#### 1. 构建并启动服务

```bash
# 使用docker-compose（最简单）
docker-compose up -d

# 或者手动构建
docker build -t pubg-item-recognition .
docker run -d -p 8000:8000 --name pubg-api pubg-item-recognition
```

#### 2. 验证服务状态

```bash
curl http://localhost:8000/health
```

预期响应：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_items": 8642
}
```

#### 3. 停止服务

```bash
# docker-compose
docker-compose down

# 或手动停止
docker stop pubg-api
docker rm pubg-api
```

### 本地运行（开发模式）

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py

# 或使用uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API端点

### 1. 健康检查

**GET** `/`

简单的健康检查。

**响应示例：**
```json
{
  "status": "online",
  "service": "PUBG Item Recognition API",
  "version": "1.0.0"
}
```

---

### 2. 详细健康检查

**GET** `/health`

返回服务详细状态，包括模型加载情况和数据库信息。

**响应示例：**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database_items": 8642
}
```

---

### 3. 物品识别（主要API）

**POST** `/recognize`

上传一张或多张游戏截图，返回去重后的物品代码列表。

**请求参数：**
- `images`: 多个图片文件（multipart/form-data）
  - 支持格式：JPG, JPEG, PNG
  - 可上传多张图片

**响应格式：**
```json
{
  "item_codes": ["11010018", "11020019", "11030012", ...],
  "total_items_detected": 15,
  "unique_items": 12,
  "images_processed": 3
}
```

**字段说明：**
- `item_codes`: 去重后的物品代码数组（保持首次出现的顺序）
- `total_items_detected`: 所有图片中检测到的物品总数
- `unique_items`: 去重后的唯一物品数量
- `images_processed`: 处理的图片数量

---

## 使用示例

### cURL

**单张图片：**
```bash
curl -X POST "http://localhost:8000/recognize" \
  -F "images=@screenshot1.png"
```

**多张图片：**
```bash
curl -X POST "http://localhost:8000/recognize" \
  -F "images=@screenshot1.png" \
  -F "images=@screenshot2.png" \
  -F "images=@screenshot3.png"
```

### Python requests

```python
import requests

url = "http://localhost:8000/recognize"

# 单张图片
with open("screenshot1.png", "rb") as f:
    files = {"images": f}
    response = requests.post(url, files=files)
    print(response.json())

# 多张图片
files = [
    ("images", open("screenshot1.png", "rb")),
    ("images", open("screenshot2.png", "rb")),
    ("images", open("screenshot3.png", "rb"))
]
response = requests.post(url, files=files)
result = response.json()

print(f"识别到的物品代码: {result['item_codes']}")
print(f"总检测数: {result['total_items_detected']}")
print(f"唯一物品: {result['unique_items']}")
```

### JavaScript (fetch)

```javascript
async function recognizeItems(imageFiles) {
  const formData = new FormData();

  // 添加多个图片文件
  imageFiles.forEach(file => {
    formData.append('images', file);
  });

  const response = await fetch('http://localhost:8000/recognize', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();
  console.log('物品代码:', result.item_codes);
  console.log('总检测数:', result.total_items_detected);
  console.log('唯一物品:', result.unique_items);

  return result;
}

// 使用示例
const input = document.querySelector('input[type="file"]');
input.addEventListener('change', async (e) => {
  const files = Array.from(e.target.files);
  const result = await recognizeItems(files);
  console.log(result);
});
```

### Postman

1. 创建新请求：`POST http://localhost:8000/recognize`
2. 选择 **Body** → **form-data**
3. 添加多个 `images` 字段，类型选择 **File**
4. 上传图片文件
5. 点击 **Send**

---

## 自动API文档

FastAPI自动生成交互式API文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

可以直接在浏览器中测试API。

---

## 错误处理

### 400 Bad Request

**原因：**
- 未上传图片
- 图片格式不支持（仅支持JPG/PNG）

**示例响应：**
```json
{
  "detail": "Invalid file type: document.pdf. Only JPG/PNG allowed."
}
```

### 500 Internal Server Error

**原因：**
- 模型加载失败
- 图片处理错误
- 识别过程异常

**示例响应：**
```json
{
  "detail": "Recognition failed: YOLO model not found"
}
```

### 503 Service Unavailable

**原因：**
- 模型未加载
- 数据库未初始化

**解决方法：**
- 检查 `/health` 端点
- 确认模型文件存在
- 查看容器日志：`docker logs pubg-api`

---

## 性能优化

### 1. 模型预加载

默认情况下，模型在首次请求时加载。可以修改 `app.py` 在启动时预加载：

```python
@app.on_event("startup")
async def startup_event():
    # 立即加载模型
    get_recognizer()
```

### 2. 批量处理

一次上传多张图片比多次单张上传更高效：
- 模型只需加载一次
- 减少HTTP请求开销
- 自动去重处理

### 3. 资源限制

调整 `docker-compose.yml` 中的资源配置：

```yaml
deploy:
  resources:
    limits:
      cpus: '4'      # 增加CPU限制
      memory: 8G     # 增加内存限制
```

### 4. GPU加速（可选）

如果有NVIDIA GPU，修改：

1. **requirements.txt**: 使用 `faiss-gpu`
2. **Dockerfile**: 使用CUDA基础镜像
3. **docker-compose.yml**: 添加GPU支持

```yaml
services:
  pubg-api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 预期性能

基于CPU推理：
- **YOLOv8检测**: ~50-100ms/图片
- **CLIP特征提取**: ~20-30ms/物品
- **向量检索**: <1ms
- **总耗时**: ~500ms-1s/图片（取决于物品数量）

基于GPU推理：
- **总耗时**: ~100-300ms/图片

---

## 部署检查清单

部署前确认以下文件存在：

- [ ] `dataset/features_db.pkl` (1.4GB)
- [ ] `dataset/labels.json`
- [ ] `dataset/category_mapping.json`
- [ ] `dataset/item_code_mapping.json`
- [ ] `runs/detect/pubg_item_detection3/weights/best.pt`
- [ ] `app.py`
- [ ] `scripts/inference.py`
- [ ] `requirements.txt`
- [ ] `Dockerfile`
- [ ] `docker-compose.yml`

---

## 故障排查

### 容器无法启动

```bash
# 查看日志
docker logs pubg-api

# 常见问题：
# 1. 模型文件缺失 → 检查文件路径
# 2. 内存不足 → 增加Docker内存限制
# 3. 端口被占用 → 修改docker-compose.yml端口映射
```

### 识别精度低

- 确认上传的是清晰的游戏截图
- 检查图片中物品是否在网格布局中
- 调整YOLOv8置信度阈值（app.py中的 `conf_threshold`）

### 响应缓慢

- 首次请求会较慢（模型加载）
- 后续请求应该更快
- 考虑使用GPU加速
- 减少单次上传的图片数量

---

## 技术栈

- **Web框架**: FastAPI 0.104+
- **ASGI服务器**: Uvicorn
- **物品检测**: YOLOv8 (Ultralytics)
- **特征提取**: OpenAI CLIP (ViT-B/32)
- **向量检索**: NumPy (余弦相似度)
- **容器化**: Docker + Docker Compose

---

## 许可与支持

如有问题，请查看：
- FastAPI文档: https://fastapi.tiangolo.com/
- YOLO文档: https://docs.ultralytics.com/
- 项目CLAUDE.md文件
