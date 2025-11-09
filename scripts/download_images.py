import json
import os
import requests
from urllib.parse import urlparse
from pathlib import Path
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 配置
ORGANIZE_BY_CATEGORY = False  # 所有图片放在同一文件夹,便于训练
MAX_WORKERS = 20  # 并发线程数,可根据网络情况调整
CREATE_LABELS = True  # 是否创建标签文件

# 线程锁,用于安全地更新计数器和列表
lock = Lock()

# 创建数据集根目录
dataset_dir = Path("../dataset")
dataset_dir.mkdir(exist_ok=True)

# 图片统一保存在 dataset/images/ 文件夹
images_dir = dataset_dir / "images"
images_dir.mkdir(exist_ok=True)

# 读取JSON文件
with open("./all_updates.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 收集所有物品信息
items_info = {}  # key: image_url, value: item info
category_counts = defaultdict(int)

for update in data.get("updates", []):
    for item in update.get("items", []):
        if "image_url" in item and item["image_url"]:
            url = item["image_url"]
            if url not in items_info:  # 避免重复
                items_info[url] = {
                    'name': item.get('name', '未知'),
                    'primary_category': item.get('primary_category_cn', '其他'),
                    'secondary_category': item.get('secondary_category_cn', ''),
                    'rarity': item.get('rarity', '未知'),
                    'item_code': item.get('item_code', '')
                }
                category_counts[item.get('primary_category_cn', '其他')] += 1

print(f"找到 {len(items_info)} 个唯一的图片")
print(f"\n类别统计:")
for category, count in sorted(category_counts.items()):
    print(f"  {category}: {count}")

# 下载图片函数,带重试机制和请求头
def download_image(url, filepath, max_retries=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Referer': 'https://pubgitems.info/',
    }

    for attempt in range(max_retries):
        try:
            # 使用stream模式下载,避免大文件内存问题
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            # 分块写入文件
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # 等待1秒后重试
                continue
            else:
                raise e

# 单个下载任务函数
def download_single_item(url_info_tuple):
    """下载单个图片的任务函数"""
    url, info = url_info_tuple

    try:
        # 从URL中提取文件名
        filename = os.path.basename(urlparse(url).path)

        # 使用 item_code 作为文件名,统一保存在 images/ 文件夹
        item_code = info['item_code']
        if item_code:
            # 使用 item_code 作为文件名,保留原扩展名
            file_ext = os.path.splitext(filename)[1]
            new_filename = f"{item_code}{file_ext}"
        else:
            new_filename = filename

        filepath = images_dir / new_filename

        # 如果文件已存在且大小大于0,跳过
        if filepath.exists() and filepath.stat().st_size > 0:
            label_item = {
                'item_code': info['item_code'],
                'filename': new_filename,
                'name': info['name'],
                'primary_category': info['primary_category'],
                'secondary_category': info['secondary_category'],
                'rarity': info['rarity'],
                'url': url,
                'filepath': f"images/{new_filename}"
            }
            return {
                'status': 'skipped',
                'name': info['name'],
                'label': label_item if CREATE_LABELS else None
            }

        # 下载图片
        download_image(url, filepath)

        label_item = {
            'item_code': info['item_code'],
            'filename': new_filename,
            'name': info['name'],
            'primary_category': info['primary_category'],
            'secondary_category': info['secondary_category'],
            'rarity': info['rarity'],
            'url': url,
            'filepath': f"images/{new_filename}"
        }

        return {
            'status': 'success',
            'name': info['name'],
            'category': info['primary_category'],
            'label': label_item if CREATE_LABELS else None
        }

    except Exception as e:
        # 删除可能的不完整文件
        if 'filepath' in locals() and filepath.exists():
            filepath.unlink()

        return {
            'status': 'failed',
            'name': info['name'],
            'url': url,
            'error': str(e)
        }

# 多线程下载
success_count = 0
failed_count = 0
skipped_count = 0
failed_urls = []
labels_data = []

print(f"\n开始多线程下载图片 (线程数: {MAX_WORKERS})...")
print("=" * 70)

# 使用线程池执行下载任务
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # 提交所有任务
    future_to_item = {
        executor.submit(download_single_item, item): item
        for item in items_info.items()
    }

    # 处理完成的任务
    completed = 0
    total = len(items_info)

    for future in as_completed(future_to_item):
        completed += 1
        result = future.result()

        with lock:
            if result['status'] == 'success':
                success_count += 1
                print(f"[{completed}/{total}] 下载成功: {result['name']} ({result['category']})")
                if result['label']:
                    labels_data.append(result['label'])

            elif result['status'] == 'skipped':
                skipped_count += 1
                print(f"[{completed}/{total}] 跳过 (已存在): {result['name']}")
                if result['label']:
                    labels_data.append(result['label'])

            elif result['status'] == 'failed':
                failed_count += 1
                print(f"[{completed}/{total}] 下载失败: {result['name']} - {result['error']}")
                failed_urls.append({
                    'url': result['url'],
                    'name': result['name'],
                    'error': result['error']
                })

print(f"\n{'=' * 70}")
print(f"下载完成!")
print(f"成功: {success_count}")
print(f"跳过: {skipped_count}")
print(f"失败: {failed_count}")
print(f"总计: {success_count + skipped_count}/{total}")
print(f"图片保存在: {images_dir.absolute()}/")

# 保存标签文件
if CREATE_LABELS and labels_data:
    labels_file = dataset_dir / "labels.json"
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump({
            'total_images': len(labels_data),
            'categories': list(category_counts.keys()),
            'items': labels_data
        }, f, ensure_ascii=False, indent=2)
    print(f"\n标签文件已保存到: {labels_file}")

    # 创建类别映射文件
    category_map_file = dataset_dir / "category_mapping.json"
    categories = sorted(category_counts.keys())
    category_mapping = {cat: idx for idx, cat in enumerate(categories)}
    with open(category_map_file, "w", encoding="utf-8") as f:
        json.dump(category_mapping, f, ensure_ascii=False, indent=2)
    print(f"类别映射已保存到: {category_map_file}")

    # 创建 item_code 到索引的映射文件 (用于训练)
    item_code_map_file = dataset_dir / "item_code_mapping.json"
    unique_item_codes = sorted([item['item_code'] for item in labels_data if item['item_code']])
    item_code_to_idx = {code: idx for idx, code in enumerate(unique_item_codes)}
    idx_to_item_code = {idx: code for code, idx in item_code_to_idx.items()}

    with open(item_code_map_file, "w", encoding="utf-8") as f:
        json.dump({
            'item_code_to_idx': item_code_to_idx,
            'idx_to_item_code': idx_to_item_code,
            'total_classes': len(unique_item_codes)
        }, f, ensure_ascii=False, indent=2)
    print(f"Item Code 映射已保存到: {item_code_map_file}")
    print(f"总共 {len(unique_item_codes)} 个唯一的 item_code")

# 保存失败的URL到文件
if failed_urls:
    with open("failed_downloads.json", "w", encoding="utf-8") as f:
        json.dump(failed_urls, f, ensure_ascii=False, indent=2)
    print(f"\n失败的下载记录已保存到 failed_downloads.json")