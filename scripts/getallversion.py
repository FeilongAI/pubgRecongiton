import requests
from bs4 import BeautifulSoup
import json
import re
import time
from typing import List, Dict, Optional
from pathlib import Path


class PUBGItemScraper:
    def __init__(self, base_url: str = "https://pubgitems.info"):
        self.base_url = base_url
        self.session = requests.Session()
        # 使用更完整的浏览器请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })

    def parse_version(self, version_str: str) -> str:
        """将版本字符串转换为标准格式，如 '4-1' -> '4.1'"""
        return version_str.replace('-', '.')

    def get_rarity_from_color(self, color: str) -> str:
        """根据颜色代码判断稀有度"""
        rarity_map = {
            '#777777': '经典',
            '#007470': '特殊',
            '#4e2f9b': '精英',
            '#b72314': '传奇',
            '#1f9c98': '特殊',
            '#7657C3': '精英',
            '#df4b3c': '传奇',
            '#9F9F9F': '经典'
        }
        return rarity_map.get(color, '未知')

    def parse_item_url(self, url: str) -> Dict[str, str]:
        """解析物品URL，提取类目信息"""
        parts = url.strip('/').split('/')

        result = {
            'primary_category': '',
            'secondary_category': '',
            'item_id': ''
        }

        if len(parts) >= 4:
            result['primary_category'] = parts[1]
            result['secondary_category'] = parts[2]
            result['item_id'] = parts[3]

        return result

    def get_category_name_cn(self, primary: str, secondary: str = None) -> Dict[str, str]:
        """获取类目的中文名称"""
        category_map = {
            'clothing': '服装',
            'weapons': '武器',
            'boxes': '盒子',
            'equip': '装备',
            'appearance': '外观',
            'other': '其他'
        }

        subcategory_map = {
            'torso': '上衣', 'legs': '腿部', 'feet': '鞋子', 'hands': '手部',
            'outer': '外套', 'head': '头部', 'mask': '面具', 'eyes': '眼部',
            'ar': '突击步枪', 'dmr': '精确射手步枪', 'sr': '狙击步枪', 'smg': '冲锋枪',
            'lmg': '轻机枪', 'shotgun': '霰弹枪', 'handgun': '手枪', 'melee': '近战',
            'charm': '挂饰', 'misc': '其他',
            'helmet': '头盔', 'vest': '防弹衣', 'backpack': '背包', 'belt': '腰带',
            'parachute': '降落伞',
            'chests': '宝箱', 'sets': '套装', 'crate': '宝箱', 'keys': '钥匙',
            'hair': '头发', 'face': '面容', 'makeup': '妆容', 'emotes': '表情',
            'contender': '竞争者',
            'vehicle': '载具', 'consumable': '消耗品', 'spray': '喷漆',
            'emblem': '徽章', 'nameplate': '名牌', 'pose': '姿势', 'lobby': '大厅',
            'unknown': '未知'
        }

        result = {
            'primary_category_cn': category_map.get(primary, primary),
            'secondary_category_cn': ''
        }

        if secondary:
            result['secondary_category_cn'] = subcategory_map.get(secondary, secondary)

        return result

    def get_all_update_versions(self, language: str = "zh-CN") -> List[str]:
        """获取所有更新版本列表"""
        url = f"{self.base_url}/{language}/updates"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            versions = []

            # 查找所有更新版本链接
            # 通常在 href="/zh-CN/updates/版本号" 的链接中
            update_links = soup.find_all('a', href=re.compile(r'/updates/[\d-]+'))

            for link in update_links:
                href = link.get('href', '')
                # 提取版本号
                match = re.search(r'/updates/([\d-]+)', href)
                if match:
                    version = match.group(1)
                    if version not in versions:
                        versions.append(version)

            # 按版本号排序
            versions.sort(key=lambda x: [int(n) for n in x.split('-')])

            return versions

        except requests.exceptions.RequestException as e:
            print(f"获取版本列表错误: {e}")
            return []

    def scrape_update_items(self, update_version: str = "4-2", language: str = "zh-CN") -> Dict:
        """抓取特定更新版本的物品信息"""
        url = f"{self.base_url}/{language}/updates/{update_version}"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            version = self.parse_version(update_version)

            total_items = 0
            rarity_stats = {}
            stats_div = soup.find('div', class_='rounded-lg bg-white/25')
            if stats_div:
                total_span = stats_div.find('span', class_='font-semibold')
                if total_span:
                    total_items = int(total_span.get_text(strip=True))

            items = []
            item_cards = soup.find_all('a', class_='bg-stone-700')

            for card in item_cards:
                item = {}
                href = card.get('href', '')
                item['url'] = self.base_url + href

                url_info = self.parse_item_url(href)
                item['primary_category'] = url_info['primary_category']
                item['secondary_category'] = url_info['secondary_category']
                item['item_id'] = url_info['item_id']

                category_names = self.get_category_name_cn(
                    url_info['primary_category'],
                    url_info['secondary_category']
                )
                item['primary_category_cn'] = category_names['primary_category_cn']
                item['secondary_category_cn'] = category_names['secondary_category_cn']

                img = card.find('img', src=True)
                if img:
                    item['image_url'] = img.get('src', '')
                    match = re.search(r'/(\d+)\.png', item['image_url'])
                    if match:
                        item['item_code'] = match.group(1)

                span_style = card.find('span', style=True)
                if span_style:
                    style = span_style.get('style', '')
                    color_match = re.search(r'--rc:(#[0-9A-Fa-f]+)', style)
                    if color_match:
                        color = color_match.group(1)
                        item['rarity'] = self.get_rarity_from_color(color)
                        item['rarity_color'] = color

                name_spans = card.find_all('span', class_='leading-tight')
                if name_spans:
                    item['name'] = name_spans[-1].get_text(strip=True)
                    if len(name_spans) > 1:
                        item['weapon_base_type'] = name_spans[0].get_text(strip=True)

                track_icon = card.find('img', alt='战场痕迹')
                item['has_battle_track'] = track_icon is not None

                items.append(item)

            for item in items:
                rarity = item.get('rarity', '未知')
                rarity_stats[rarity] = rarity_stats.get(rarity, 0) + 1

            return {
                'version': version,
                'version_code': update_version,
                'total_items': total_items,
                'actual_items_count': len(items),
                'rarity_statistics': rarity_stats,
                'items': items
            }

        except requests.exceptions.RequestException as e:
            print(f"请求错误 ({update_version}): {e}")
            return None

    def scrape_all_updates(self, language: str = "zh-CN", output_dir: str = "pubg_data", delay: float = 1.0):
        """抓取所有更新版本的数据"""
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 获取所有版本
        print("正在获取所有更新版本列表...")
        versions = self.get_all_update_versions(language)

        if not versions:
            print("未找到任何版本")
            return

        print(f"找到 {len(versions)} 个版本: {', '.join(versions)}")
        print(f"\n开始抓取数据...")
        print("=" * 70)

        all_data = []
        success_count = 0

        for i, version in enumerate(versions, 1):
            print(f"\n[{i}/{len(versions)}] 正在抓取版本 {version}...")

            data = self.scrape_update_items(version, language)

            if data:
                # 保存单个版本文件
                version_file = output_path / f"update_{version}.json"
                self.save_to_json(data, str(version_file))

                all_data.append(data)
                success_count += 1

                print(f"  [OK] 版本 {data['version']} - {data['actual_items_count']} 个物品")
            else:
                print(f"  [FAIL] 版本 {version} 抓取失败")

            # 延迟，避免请求过快
            if i < len(versions):
                time.sleep(delay)

        # 保存合并数据
        if all_data:
            merged_data = {
                'total_versions': len(all_data),
                'total_items': sum(d['actual_items_count'] for d in all_data),
                'updates': all_data
            }

            merged_file = output_path / "all_updates.json"
            self.save_to_json(merged_data, str(merged_file))

            print(f"\n{'=' * 70}")
            print(f"抓取完成！")
            print(f"成功: {success_count}/{len(versions)} 个版本")
            print(f"总物品数: {merged_data['total_items']}")
            print(f"数据已保存到: {output_dir}/")
            print(f"  - 单个版本文件: update_*.json")
            print(f"  - 合并文件: all_updates.json")

    def save_to_json(self, data: Dict, filename: str):
        """保存数据到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# 使用示例
if __name__ == "__main__":
    scraper = PUBGItemScraper()

    # 方式1: 抓取所有版本
    scraper.scrape_all_updates(
        language="zh-CN",
        output_dir="pubg_data",
        delay=1.0  # 每次请求间隔1秒
    )

    # 方式2: 只抓取指定版本
    # versions_to_scrape = ["4-1", "4-2", "4-3"]
    # for version in versions_to_scrape:
    #     data = scraper.scrape_update_items(version, "zh-CN")
    #     if data:
    #         scraper.save_to_json(data, f"pubg_update_{version}.json")