#!/usr/bin/env python3
"""
脚本：将PUBG物品数据批量发送到API
从all_updates.json读取数据并POST到本地API服务
"""

import json
import requests
import time
from typing import Dict, List

# API配置
API_URL = "http://localhost:8080/api/v1/items"
HEADERS = {
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json',
    'Accept': '*/*',
    'Host': 'localhost:8080',
    'Connection': 'keep-alive'
}

# 文件路径
JSON_FILE = "pubg_update_39-2.json"


def load_json_data(file_path: str) -> Dict:
    """加载JSON文件"""
    print(f"正在读取文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"文件读取成功！总版本数: {data.get('total_versions', 0)}, 总物品数: {data.get('total_items', 0)}")
    return data


def prepare_item_data(item: Dict, version: str) -> Dict:
    """准备要发送的物品数据"""
    return {
        "primary_category_cn": item.get("primary_category_cn", ""),
        "secondary_category_cn": item.get("secondary_category_cn", ""),
        "item_code": item.get("item_code", ""),
        "rarity": item.get("rarity", "未知"),
        "name": item.get("name", ""),
        "version": version,
        "weapon_base_type": item.get("weapon_base_type", "")
    }


def send_item(item_data: Dict, index: int, total: int) -> bool:
    """发送单个物品到API"""
    try:
        response = requests.post(API_URL, headers=HEADERS, json=item_data, timeout=10)

        if response.status_code in [200, 201]:
            print(f"[{index}/{total}] ✓ 成功: {item_data['name']} (编号: {item_data['item_code']})")
            return True
        else:
            print(f"[{index}/{total}] ✗ 失败: {item_data['name']} - 状态码: {response.status_code}, 响应: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"[{index}/{total}] ✗ 错误: {item_data['name']} - {str(e)}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("PUBG物品数据批量上传脚本")
    print("=" * 60)

    # 加载数据
    try:
        data = load_json_data(JSON_FILE)
    except FileNotFoundError:
        print(f"错误: 文件 {JSON_FILE} 不存在！")
        return
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件解析失败 - {str(e)}")
        return

    # 收集所有物品
    all_items = []
    # for update in data.get("updates", []):
    version = data.get("version", "未知版本")
    items = data.get("items", [])
    for item in items:
        item_data = prepare_item_data(item, version)
        all_items.append(item_data)

    total_items = len(all_items)
    print(f"\n准备发送 {total_items} 个物品到API")
    print(f"目标URL: {API_URL}")

    # 询问是否继续
    confirm = input("\n是否继续? (y/n): ").strip().lower()
    if confirm != 'y':
        print("操作已取消")
        return

    # 发送数据
    print("\n开始发送数据...\n")
    success_count = 0
    failed_count = 0

    start_time = time.time()

    for index, item_data in enumerate(all_items, 1):
        if send_item(item_data, index, total_items):
            success_count += 1
        else:
            failed_count += 1

        # 添加小延迟，避免请求过快
        time.sleep(0.1)

    # 统计结果
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("上传完成！")
    print(f"总计: {total_items} 个物品")
    print(f"成功: {success_count} 个")
    print(f"失败: {failed_count} 个")
    print(f"耗时: {elapsed_time:.2f} 秒")
    print("=" * 60)


if __name__ == "__main__":
    main()