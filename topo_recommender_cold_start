import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gtda.mapper import make_mapper_pipeline, CubicalCover
from sklearn.cluster import DBSCAN
import networkx as nx
from matplotlib.font_manager import FontProperties
import platform
import random
from scipy.spatial import cKDTree
import os
from collections import Counter
import pickle
from multiprocessing import Pool, cpu_count
import faiss
import time

# 颜色映射定义为全局变量
category_colors = {
    "搞笑": "#FF9999",  # 红色系
    "美食": "#66B2FF",  # 蓝色系
    "知识": "#99FF99",  # 绿色系
    "时尚": "#FFCC99",  # 橙色系
    "舞蹈": "#CC99FF",  # 紫色系
    "游戏": "#FFFF99",  # 黄色系
    "旅行": "#99FFFF"   # 青色系
}

# 设置中文字体
system = platform.system()
if system == 'Windows':
    font_path = 'C:/Windows/Fonts/msyh.ttc'
elif system == 'Darwin':
    font_path = '/System/Library/Fonts/PingFang.ttc'
else:
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'

try:
    if os.path.exists(font_path):
        font_prop = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置中文字体时出错: {e}")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

class ShortVideoRecommender:
    """基于拓扑数据分析的短视频推荐系统"""
    
    def __init__(self):
        np.random.seed(42)
        self.features = None
        self.videos_df = None
        self.user_interactions = None
        self.graph = None
        self.node_to_videos = None
        self.node_categories = None
        self.video_pca = None
        self.G = None
    
    def create_synthetic_data(self, n_videos=200, n_users=500, n_features=20):
        """创建合成的短视频数据集和用户交互数据"""
        print("正在创建合成短视频数据集...")
        
        # 短视频分类
        categories = list(category_colors.keys())
        
        # 为每个分类创建特征中心
        category_centers = {cat: np.random.rand(n_features) for cat in categories}
        
        # 生成多模态特征（视觉、音频、文本等特征的组合）
        # 每个视频的特征是对应类别中心加上一些随机噪声
        features = np.zeros((n_videos, n_features))
        video_categories = []
        video_names = []
        video_durations = []  # 视频时长（秒）
        video_popularity = []  # 视频本身的受欢迎程度
        
        for i in range(n_videos):
            # 随机选择一个类别
            category = random.choice(categories)
            video_categories.append(category)
            
            # 生成偏向该类别中心的特征
            noise = 0.3 * np.random.randn(n_features)
            features[i] = 0.7 * category_centers[category] + noise
            
            # 生成视频名称
            video_names.append(f"视频{i+1}")
            
            # 生成视频时长（5秒到60秒之间）
            video_durations.append(np.random.randint(5, 61))
            
            # 生成视频本身的热度值（0到100之间）
            video_popularity.append(np.random.randint(1, 101))
        
        # 创建视频数据框
        self.videos_df = pd.DataFrame({
            'video_id': range(n_videos),
            'title': video_names,
            'category': video_categories,
            'duration': video_durations,
            'popularity': video_popularity
        })
        
        # 保存特征矩阵
        self.features = features
        
        # 生成用户交互数据
        # 包括：观看时长比例、是否完播、点赞、评论、收藏、分享
        interactions = []
        
        # 为每个用户分配一个偏好向量
        user_preferences = np.random.rand(n_users, n_features)
        
        # 确保某些用户对某些类别的偏好更强
        for u in range(n_users):
            # 随机选择1-3个喜欢的类别
            fav_categories = np.random.choice(categories, size=np.random.randint(1, 4), replace=False)
            # 加强对这些类别的偏好
            for cat in fav_categories:
                user_preferences[u] += 0.5 * category_centers[cat]
        
        # 为每个用户生成一些交互记录
        for u in range(n_users):
            # 每个用户交互的视频数量（10到30个）
            n_interactions = np.random.randint(10, 31)
            
            # 选择视频（偏向于与用户偏好更匹配的视频）
            # 计算用户偏好与所有视频的相似度
            similarities = np.dot(features, user_preferences[u])
            
            # 基于相似度的概率选择视频
            probs = np.exp(similarities) / np.sum(np.exp(similarities))
            watched_videos = np.random.choice(
                n_videos, 
                size=n_interactions, 
                replace=False, 
                p=probs
            )
            
            for v in watched_videos:
                # 相似度越高，交互行为越积极
                similarity = np.dot(features[v], user_preferences[u])
                
                # 观看时长比例（0到1之间）
                watch_ratio = min(1.0, max(0.1, 0.5 + 0.5 * similarity + 0.2 * np.random.randn()))
                
                # 是否完播（观看比例大于0.9视为完播）
                completed = watch_ratio > 0.9
                
                # 交互行为（点赞、评论、收藏、分享）
                # 相似度和完播情况会影响这些行为的概率
                base_prob = similarity if completed else 0.7 * similarity
                
                liked = np.random.random() < (base_prob * 0.8)
                commented = np.random.random() < (base_prob * 0.3)
                favorited = np.random.random() < (base_prob * 0.5)
                shared = np.random.random() < (base_prob * 0.2)
                
                interactions.append({
                    'user_id': u,
                    'video_id': v,
                    'watch_ratio': watch_ratio,
                    'completed': completed,
                    'liked': liked,
                    'commented': commented,
                    'favorited': favorited,
                    'shared': shared
                })
        
        self.user_interactions = pd.DataFrame(interactions)
        
        print(f"创建了{n_videos}个短视频和{len(interactions)}条用户交互记录")
        return self.videos_df, self.user_interactions, self.features
    
    def calculate_engagement_score(self, row):
        """根据用户交互计算参与度分数"""
        score = row['watch_ratio'] * 0.4  # 观看时长比例
        if row['completed']:
            score += 0.2  # 完播奖励
        if row['liked']:
            score += 0.1  # 点赞
        if row['commented']:
            score += 0.1  # 评论
        if row['favorited']:
            score += 0.1  # 收藏
        if row['shared']:
            score += 0.1  # 分享
        return score
    
    def build_topological_representation(self):
        """构建短视频数据的拓扑表示"""
        print("\n构建拓扑表示...")
        
        # 使用PCA降维以便可视化
        pca = PCA(n_components=2)
        self.video_pca = pca.fit_transform(self.features)
        
        # 使用Mapper构建拓扑表示
        mapper = make_mapper_pipeline(
            filter_func=pca,  # 使用PCA作为过滤函数
            cover=CubicalCover(n_intervals=8, overlap_frac=0.3),  # 立方体覆盖
            clusterer=DBSCAN(eps=0.5, min_samples=2),  # 聚类算法
            verbose=True
        )
        
        # 应用Mapper获取拓扑图
        self.graph = mapper.fit_transform(self.features)
        print(f"拓扑图节点数量: {self.graph.vcount()}")
        print(f"拓扑图边数量: {self.graph.ecount()}")
        
        # 将igraph转换为networkx以便后续分析
        self.G = nx.Graph()
        
        # 添加节点
        for i in range(self.graph.vcount()):
            self.G.add_node(i)
        
        # 添加边
        for edge in self.graph.es:
            self.G.add_edge(edge.source, edge.target)
        
        # 计算每个节点包含的视频
        self.node_to_videos = {i: [] for i in range(self.graph.vcount())}
        
        layout = self.graph.layout_fruchterman_reingold()
        node_positions = np.array(layout.coords)
        
        # 为每个视频分配节点
        kdtree = cKDTree(node_positions)
        for video_idx in range(len(self.features)):
            # 找到最近的节点
            _, closest_node = kdtree.query(self.video_pca[video_idx])
            # 将视频添加到节点
            self.node_to_videos[closest_node].append(video_idx)
        
        # 分析每个节点的主要视频类别
        self.node_categories = []
        node_popularity = []  # 记录每个节点的平均热度
        
        for node_idx in range(self.graph.vcount()):
            video_indices = self.node_to_videos[node_idx]
            if video_indices:
                # 获取节点中视频的类别
                node_category_list = [self.videos_df.iloc[i]['category'] for i in video_indices]
                # 获取节点中视频的平均热度
                node_avg_popularity = np.mean([self.videos_df.iloc[i]['popularity'] for i in video_indices])
                
                # 获取最常见的类别
                category_counter = Counter(node_category_list)
                most_common_category = category_counter.most_common(1)[0][0] if category_counter else "未知"
                
                self.node_categories.append(most_common_category)
                node_popularity.append(node_avg_popularity)
            else:
                self.node_categories.append("未知")
                node_popularity.append(0)
        
        # 可视化拓扑图
        plt.figure(figsize=(14, 12))
        pos = {i: (layout.coords[i][0], layout.coords[i][1]) for i in range(self.graph.vcount())}
        
        # 绘制边
        nx.draw_networkx_edges(self.G, pos, alpha=0.5, width=1.5)
        
        # 绘制节点，根据类别着色，根据热度调整大小
        for category in category_colors:
            category_nodes = [i for i, cat in enumerate(self.node_categories) if cat == category]
            if category_nodes:
                node_sizes = [30 + node_popularity[i]/2 for i in category_nodes]
                nx.draw_networkx_nodes(
                    self.G, pos, 
                    nodelist=category_nodes,
                    node_color=category_colors[category],
                    node_size=node_sizes,
                    alpha=0.8,
                    label=category
                )
        
        # 添加节点标签
        labels = {i: str(i) for i in range(self.graph.vcount())}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=9)
        
        plt.title('短视频内容的拓扑表示', fontsize=20)
        plt.legend(fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('短视频拓扑结构.png', dpi=300)
        
        # 可视化视频在PCA空间中的分布
        plt.figure(figsize=(12, 10))
        
        for category in category_colors:
            category_indices = self.videos_df[self.videos_df['category'] == category].index
            popularity = self.videos_df.iloc[category_indices]['popularity'].values
            
            # 点的大小根据热度调整
            sizes = 20 + popularity / 3
            
            plt.scatter(
                self.video_pca[category_indices, 0],
                self.video_pca[category_indices, 1],
                color=category_colors[category],
                label=category,
                alpha=0.7,
                s=sizes
            )
        
        plt.title('短视频在特征空间的分布', fontsize=20)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig('短视频特征空间.png', dpi=300)
        
        return self.G, self.node_to_videos, self.node_categories
    
    def recommend_videos(self, user_id, n_recommendations=10):
        """基于拓扑结构为用户推荐短视频"""
        print(f"\n为用户 {user_id} 推荐短视频:")
        
        # 获取用户的交互历史
        user_history = self.user_interactions[self.user_interactions['user_id'] == user_id]
        
        if len(user_history) == 0:
            print("该用户没有交互记录，无法提供推荐")
            return []
        
        # 计算用户对每个视频的参与度
        user_history['engagement'] = user_history.apply(self.calculate_engagement_score, axis=1)
        
        # 获取用户已观看的视频，按参与度排序
        watched_videos = user_history.sort_values('engagement', ascending=False)
        watched_video_ids = watched_videos['video_id'].astype(int).tolist()
        
        # 显示用户偏好
        top_watched = watched_videos.head(5)
        print("\n用户偏好分析:")
        for _, row in top_watched.iterrows():
            video_id = int(row['video_id'])
            video = self.videos_df.iloc[video_id]
            print(f"- {video['title']} ({video['category']}): 参与度 {row['engagement']:.2f}")
        
        # 统计用户偏好的类别
        watched_categories = [self.videos_df.iloc[int(vid)]['category'] for vid in watched_videos['video_id']]
        category_counter = Counter(watched_categories)
        
        print("\n类别偏好:")
        for category, count in category_counter.most_common():
            percentage = count / len(watched_categories) * 100
            print(f"- {category}: {percentage:.1f}%")
        
        # 找出包含用户喜欢视频的节点
        user_nodes = set()
        for node, videos in self.node_to_videos.items():
            if any(vid in watched_video_ids for vid in videos):
                user_nodes.add(node)
        
        # 找出相关节点（用户节点及其邻居）
        relevant_nodes = set(user_nodes)
        for node in user_nodes:
            relevant_nodes.update(self.G.neighbors(node))
        
        # 根据用户偏好计算节点分数
        node_scores = {}
        for node in relevant_nodes:
            # 节点中的视频类别
            videos_in_node = self.node_to_videos[node]
            if not videos_in_node:
                continue
                
            # 类别分数：该节点中视频类别与用户偏好的匹配程度
            node_categories = [self.videos_df.iloc[vid]['category'] for vid in videos_in_node]
            category_match = sum(category_counter.get(cat, 0) for cat in node_categories) / len(node_categories)
            
            # 连接分数：该节点与用户已访问节点的连接强度
            connection_score = sum(1 for unode in user_nodes if self.G.has_edge(node, unode))
            
            # 多样性分数：引入一些多样性，稍微提高用户不常看的类别的分数
            diversity_factor = 1.0
            node_main_category = Counter(node_categories).most_common(1)[0][0]
            if category_counter.get(node_main_category, 0) < len(watched_videos) / (len(category_colors) * 2):
                diversity_factor = 1.2
            
            # 总分 = 类别匹配 + 连接强度 + 多样性
            node_scores[node] = (category_match + 0.5 * connection_score) * diversity_factor
        
        # 根据节点分数为用户推荐视频
        candidate_videos = []
        
        # 从评分高的节点中获取候选视频
        for node, score in sorted(node_scores.items(), key=lambda x: x[1], reverse=True):
            for video_id in self.node_to_videos[node]:
                if video_id not in watched_video_ids:
                    # 计算视频分数，考虑节点分数和视频自身属性
                    video_popularity = self.videos_df.iloc[video_id]['popularity'] / 100  # 归一化到0-1
                    video_score = score * 0.7 + video_popularity * 0.3
                    
                    candidate_videos.append((video_id, video_score, node))
        
        # 排序并去重
        candidate_videos.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        recommendations = []
        
        for video_id, score, node in candidate_videos:
            if video_id not in seen and len(recommendations) < n_recommendations:
                seen.add(video_id)
                recommendations.append((video_id, score, node))
        
        # 输出推荐结果
        print("\n推荐视频:")
        for video_id, score, node in recommendations:
            video = self.videos_df.iloc[video_id]
            print(f"- {video['title']} ({video['category']}): 推荐度 {score:.2f}, 节点 {node}, 热度 {video['popularity']}")
        
        # 可视化用户历史和推荐视频
        self.visualize_recommendations(user_id, watched_video_ids, [r[0] for r in recommendations])
        
        return recommendations
    
    def visualize_recommendations(self, user_id, history_videos, recommended_videos):
        """可视化用户历史和推荐视频"""
        plt.figure(figsize=(14, 12))
        
        # 可视化拓扑图背景
        layout = self.graph.layout_fruchterman_reingold()
        pos = {i: (layout.coords[i][0], layout.coords[i][1]) for i in range(self.graph.vcount())}
        
        # 先绘制所有边
        nx.draw_networkx_edges(self.G, pos, alpha=0.2, width=1.0)
        
        # 然后绘制所有节点
        for i in range(self.graph.vcount()):
            nx.draw_networkx_nodes(
                self.G, pos, 
                nodelist=[i],
                node_color=category_colors.get(self.node_categories[i], 'gray'),
                node_size=50,
                alpha=0.5
            )
        
        # 高亮用户历史节点
        user_nodes = set()
        for node, videos in self.node_to_videos.items():
            if any(vid in history_videos for vid in videos):
                user_nodes.add(node)
        
        nx.draw_networkx_nodes(
            self.G, pos, 
            nodelist=list(user_nodes),
            node_color='blue',
            node_size=100,
            alpha=0.8,
            label='用户历史'
        )
        
        # 高亮推荐视频节点
        rec_nodes = set()
        for node, videos in self.node_to_videos.items():
            if any(vid in recommended_videos for vid in videos):
                rec_nodes.add(node)
        
        nx.draw_networkx_nodes(
            self.G, pos, 
            nodelist=list(rec_nodes),
            node_color='red',
            node_size=100,
            alpha=0.8,
            label='推荐视频'
        )
        
        # 添加节点标签
        labels = {i: str(i) for i in user_nodes.union(rec_nodes)}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=9)
        
        # 在PCA空间中可视化用户历史和推荐
        plt.figure(figsize=(12, 10))
        
        # 绘制所有视频点（小点，透明）
        for category in category_colors:
            category_indices = self.videos_df[self.videos_df['category'] == category].index
            plt.scatter(
                self.video_pca[category_indices, 0],
                self.video_pca[category_indices, 1],
                color=category_colors[category],
                label=category,
                alpha=0.3,
                s=30
            )
        
        # 绘制用户历史视频（大的黑色圆圈）
        plt.scatter(
            self.video_pca[history_videos, 0],
            self.video_pca[history_videos, 1],
            color='black',
            marker='o',
            s=150,
            label='用户历史',
            facecolors='none',
            linewidth=2
        )
        
        # 绘制推荐视频（红色星星）
        plt.scatter(
            self.video_pca[recommended_videos, 0],
            self.video_pca[recommended_videos, 1],
            color='red',
            marker='*',
            s=200,
            label='推荐视频'
        )
        
        # 添加标签
        for i in history_videos[:5]:  # 只显示前5个历史视频标签
            plt.annotate(
                self.videos_df.iloc[i]['title'],
                (self.video_pca[i, 0], self.video_pca[i, 1]),
                fontsize=9,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        for i in recommended_videos[:5]:  # 只显示前5个推荐标签
            plt.annotate(
                self.videos_df.iloc[i]['title'],
                (self.video_pca[i, 0], self.video_pca[i, 1]),
                fontsize=9,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.title(f'用户{user_id}的短视频偏好与推荐', fontsize=18)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'用户{user_id}短视频推荐.png', dpi=300)
        
        print(f"\n可视化结果已保存到'用户{user_id}短视频推荐.png'")

    def save_topological_model(self, filepath):
        """保存拓扑模型到文件
        
        Args:
            filepath: 模型保存路径
        """
        model_data = {
            'graph': self.graph,
            'node_to_videos': self.node_to_videos,
            'node_categories': self.node_categories,
            'video_pca': self.video_pca,
            'G': self.G,
            'features': self.features
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"拓扑模型已保存到 {filepath}")
    
    def load_topological_model(self, filepath):
        """从文件加载拓扑模型
        
        Args:
            filepath: 模型文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.graph = model_data['graph']
        self.node_to_videos = model_data['node_to_videos']
        self.node_categories = model_data['node_categories']
        self.video_pca = model_data['video_pca']
        self.G = model_data['G']
        self.features = model_data['features']
        print(f"已加载拓扑模型 {filepath}")

class ColdStartRecommender:
    """针对上亿用户的短视频冷启动推荐系统"""
    
    def __init__(self, video_features=None, video_metadata=None, topo_model_path=None):
        """初始化冷启动推荐系统
        
        Args:
            video_features: 视频特征矩阵，形状为 (n_videos, n_features)
            video_metadata: 视频元数据 DataFrame
            topo_model_path: 预训练拓扑模型路径，如果提供则加载
        """
        self.video_features = video_features
        self.video_metadata = video_metadata
        self.topo_recommender = ShortVideoRecommender()
        self.feature_index = None
        self.user_cluster_index = None
        self.user_clusters = {}  # 用户聚类映射
        self.cold_start_nodes = []  # 冷启动推荐的节点集合
        self.category_videos = {}  # 按类别索引的视频
        
        if topo_model_path and os.path.exists(topo_model_path):
            self.topo_recommender.load_topological_model(topo_model_path)
            # 确保topo_recommender中的videos_df被设置
            if self.topo_recommender.videos_df is None and video_metadata is not None:
                self.topo_recommender.videos_df = video_metadata
            self.init_category_index()
    
    def init_category_index(self):
        """初始化类别视频索引"""
        if self.topo_recommender.videos_df is not None:
            # 按类别组织视频
            for category in category_colors.keys():
                category_indices = self.topo_recommender.videos_df[
                    self.topo_recommender.videos_df['category'] == category
                ].index.tolist()
                self.category_videos[category] = category_indices
    
    def build_feature_index(self):
        """构建视频特征索引用于快速检索"""
        if self.video_features is None and self.topo_recommender.features is not None:
            self.video_features = self.topo_recommender.features
        
        if self.video_features is not None:
            n_features = self.video_features.shape[1]
            # 使用FAISS构建索引
            self.feature_index = faiss.IndexFlatL2(n_features)
            # 添加所有视频特征到索引
            self.feature_index.add(self.video_features.astype('float32'))
            print(f"已构建视频特征索引，包含 {self.feature_index.ntotal} 个视频")
    
    def precompute_cold_start_nodes(self, top_k=10):
        """预计算冷启动推荐的高质量节点
        
        Args:
            top_k: 每个类别选择的节点数量
        """
        if not hasattr(self.topo_recommender, 'node_categories') or self.topo_recommender.node_categories is None:
            print("错误：未找到拓扑模型的节点类别数据")
            return
        
        # 按类别组织节点
        category_nodes = {}
        for category in category_colors.keys():
            category_nodes[category] = [
                i for i, cat in enumerate(self.topo_recommender.node_categories) 
                if cat == category
            ]
        
        # 计算每个节点的质量分数
        node_scores = {}
        for node_idx in range(self.topo_recommender.graph.vcount()):
            # 获取节点中的视频
            videos = self.topo_recommender.node_to_videos.get(node_idx, [])
            if not videos:
                continue
            
            # 计算节点中视频的平均热度和多样性
            if hasattr(self.topo_recommender, 'videos_df') and self.topo_recommender.videos_df is not None:
                avg_popularity = np.mean([
                    self.topo_recommender.videos_df.iloc[vid]['popularity'] 
                    for vid in videos
                ])
                
                # 类别多样性
                categories = [
                    self.topo_recommender.videos_df.iloc[vid]['category'] 
                    for vid in videos
                ]
                category_diversity = len(set(categories)) / len(categories) if categories else 0
                
                # 节点连接度（作为内容关联性的代理）
                connectivity = self.topo_recommender.G.degree(node_idx)
                
                # 综合分数 = 热度 * 0.6 + 多样性 * 0.2 + 连接度 * 0.2
                node_scores[node_idx] = avg_popularity * 0.6 + category_diversity * 100 * 0.2 + connectivity * 0.2
        
        # 为每个类别选择最佳节点
        self.cold_start_nodes = []
        for category, nodes in category_nodes.items():
            if not nodes:
                continue
                
            # 按分数排序
            sorted_nodes = sorted(
                [(node, node_scores.get(node, 0)) for node in nodes],
                key=lambda x: x[1],
                reverse=True
            )
            
            # 选取top_k个节点
            top_nodes = [node for node, _ in sorted_nodes[:top_k]]
            self.cold_start_nodes.extend(top_nodes)
        
        print(f"已预计算 {len(self.cold_start_nodes)} 个冷启动推荐节点")
    
    def cold_start_recommendation(self, user_features=None, category_preference=None, n_recommendations=10):
        """为新用户提供冷启动推荐
        
        Args:
            user_features: 用户特征向量，如果有
            category_preference: 用户类别偏好字典，如{'搞笑': 0.7, '美食': 0.3}
            n_recommendations: 推荐数量
            
        Returns:
            推荐视频列表，每项为(video_id, score)
        """
        start_time = time.time()
        
        # 如果没有预计算冷启动节点，则进行预计算
        if not self.cold_start_nodes:
            self.precompute_cold_start_nodes()
        
        # 根据用户特征或类别偏好进行推荐
        if category_preference:
            # 根据类别偏好选择视频
            return self._recommend_by_category(category_preference, n_recommendations)
        elif user_features is not None:
            # 根据用户特征选择最相似视频
            return self._recommend_by_features(user_features, n_recommendations)
        else:
            # 默认冷启动：使用预计算的高质量节点
            return self._recommend_default(n_recommendations)
    
    def _recommend_by_category(self, category_preference, n_recommendations):
        """根据用户类别偏好推荐视频"""
        # 按偏好权重选择视频
        candidates = []
        start_time = time.time()
        
        # 优先选择冷启动节点中的视频
        for node in self.cold_start_nodes:
            videos = self.topo_recommender.node_to_videos.get(node, [])
            node_category = self.topo_recommender.node_categories[node]
            
            # 如果节点类别在用户偏好中
            preference_weight = category_preference.get(node_category, 0)
            if preference_weight > 0:
                for vid in videos:
                    # 获取视频热度
                    popularity = self.topo_recommender.videos_df.iloc[vid]['popularity'] / 100
                    # 计算分数 = 类别偏好 * 0.7 + 热度 * 0.3
                    score = preference_weight * 0.7 + popularity * 0.3
                    candidates.append((vid, score))
        
        # 如果冷启动节点没有足够视频，从类别视频中直接选择
        if len(candidates) < n_recommendations * 2:
            for category, weight in category_preference.items():
                if category in self.category_videos:
                    for vid in self.category_videos[category][:20]:  # 只看前20个
                        popularity = self.topo_recommender.videos_df.iloc[vid]['popularity'] / 100
                        score = weight * 0.7 + popularity * 0.3
                        candidates.append((vid, score))
        
        # 排序并去重
        candidates.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        recommendations = []
        
        for vid, score in candidates:
            if vid not in seen and len(recommendations) < n_recommendations:
                seen.add(vid)
                recommendations.append((vid, score))
        
        print(f"冷启动推荐完成，耗时: {time.time() - start_time:.2f}秒")
        return recommendations
    
    def _recommend_by_features(self, user_features, n_recommendations):
        """根据用户特征向量推荐相似视频"""
        # 如果没有构建特征索引，先构建
        if self.feature_index is None:
            self.build_feature_index()
            
        start_time = time.time()
        
        if self.feature_index is not None:
            # 查询最相似的视频
            user_features = np.array([user_features], dtype='float32')
            distances, indices = self.feature_index.search(user_features, n_recommendations * 2)
            
            # 处理结果
            candidates = []
            for i, idx in enumerate(indices[0]):
                vid = int(idx)
                # 计算分数为相似度的倒数（距离越小越好）
                score = 1.0 / (1.0 + distances[0][i])
                candidates.append((vid, score))
            
            # 结合视频热度调整分数
            final_candidates = []
            for vid, score in candidates:
                if vid < len(self.topo_recommender.videos_df):
                    popularity = self.topo_recommender.videos_df.iloc[vid]['popularity'] / 100
                    # 调整分数 = 相似度 * 0.7 + 热度 * 0.3
                    adjusted_score = score * 0.7 + popularity * 0.3
                    final_candidates.append((vid, adjusted_score))
            
            # 排序并限制推荐数量
            final_candidates.sort(key=lambda x: x[1], reverse=True)
            recommendations = final_candidates[:n_recommendations]
            
            print(f"特征匹配推荐完成，耗时: {time.time() - start_time:.2f}秒")
            return recommendations
        
        # 如果没有特征索引，使用默认推荐
        return self._recommend_default(n_recommendations)
    
    def _recommend_default(self, n_recommendations):
        """默认冷启动推荐：使用预计算的高质量节点"""
        candidates = []
        start_time = time.time()
        
        # 从冷启动节点中获取视频
        for node in self.cold_start_nodes:
            videos = self.topo_recommender.node_to_videos.get(node, [])
            for vid in videos:
                # 获取视频热度作为分数
                popularity = self.topo_recommender.videos_df.iloc[vid]['popularity'] / 100
                candidates.append((vid, popularity))
        
        # 排序并去重
        candidates.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        recommendations = []
        
        for vid, score in candidates:
            if vid not in seen and len(recommendations) < n_recommendations:
                seen.add(vid)
                recommendations.append((vid, score))
        
        print(f"默认冷启动推荐完成，耗时: {time.time() - start_time:.2f}秒")
        return recommendations
    
    def batch_recommend(self, user_preferences, n_recommendations=10, n_workers=None):
        """批量为多个用户生成推荐
        
        Args:
            user_preferences: 列表，每项为(user_id, category_preference)或(user_id, user_features)
            n_recommendations: 每个用户的推荐数量
            n_workers: 并行工作进程数，默认为CPU核心数
        
        Returns:
            字典，键为user_id，值为推荐列表
        """
        if n_workers is None:
            n_workers = cpu_count()
        
        start_time = time.time()
        results = {}
        
        # 预计算冷启动节点（如果尚未计算）
        if not self.cold_start_nodes:
            self.precompute_cold_start_nodes()
        
        # 多线程处理用户
        with Pool(processes=n_workers) as pool:
            jobs = []
            
            for user_data in user_preferences:
                user_id = user_data[0]
                preference = user_data[1]
                
                # 根据preference类型决定推荐方法
                if isinstance(preference, dict):
                    # 类别偏好
                    job = pool.apply_async(
                        self._recommend_by_category,
                        (preference, n_recommendations)
                    )
                else:
                    # 用户特征
                    job = pool.apply_async(
                        self._recommend_by_features,
                        (preference, n_recommendations)
                    )
                
                jobs.append((user_id, job))
            
            # 收集结果
            for user_id, job in jobs:
                results[user_id] = job.get()
        
        total_time = time.time() - start_time
        print(f"批量推荐完成，为 {len(user_preferences)} 个用户生成推荐，总耗时: {total_time:.2f}秒")
        return results
    
    def evaluate_cold_start(self, test_interactions, n_recommendations=10):
        """评估冷启动推荐效果
        
        Args:
            test_interactions: 测试数据，包含真实用户互动
            n_recommendations: 推荐数量
            
        Returns:
            评估指标字典
        """
        # 实现冷启动评估逻辑
        pass

def main():
    # 创建推荐系统实例
    topo_recommender = ShortVideoRecommender()
    
    # 创建合成数据
    videos_df, interactions_df, features = topo_recommender.create_synthetic_data(n_videos=500, n_users=1000)
    
    print("\n数据集概览:")
    print("\n视频数据:")
    print(videos_df.head())
    
    print("\n各类别视频数量:")
    print(videos_df['category'].value_counts())
    
    # 构建拓扑表示
    topo_recommender.build_topological_representation()
    
    # 保存拓扑模型
    topo_recommender.save_topological_model('topo_model.pkl')
    
    # 创建冷启动推荐系统
    cold_start = ColdStartRecommender(
        video_features=features,
        video_metadata=videos_df,
        topo_model_path='topo_model.pkl'
    )
    
    # 确保videos_df被正确设置
    if cold_start.topo_recommender.videos_df is None:
        cold_start.topo_recommender.videos_df = videos_df
    
    # 预计算冷启动节点
    cold_start.precompute_cold_start_nodes()
    
    # 测试冷启动推荐 - 类别偏好
    print("\n基于类别偏好的冷启动推荐:")
    category_preference = {"搞笑": 0.6, "知识": 0.4}
    recommendations = cold_start.cold_start_recommendation(
        category_preference=category_preference,
        n_recommendations=5
    )
    
    for vid, score in recommendations:
        video = videos_df.iloc[vid]
        print(f"- {video['title']} ({video['category']}): 推荐度 {score:.2f}, 热度 {video['popularity']}")
    
    # 测试批量推荐
    print("\n测试批量冷启动推荐:")
    # 模拟10000个用户的偏好
    user_preferences = []
    categories = list(category_colors.keys())
    
    for i in range(100):
        # 随机生成类别偏好
        n_categories = random.randint(1, 3)
        selected_categories = random.sample(categories, n_categories)
        
        preference = {}
        for cat in selected_categories:
            preference[cat] = random.random()
            
        # 归一化偏好权重
        total = sum(preference.values())
        if total > 0:
            for cat in preference:
                preference[cat] /= total
                
        user_preferences.append((i, preference))
    
    # 批量推荐
    batch_results = cold_start.batch_recommend(
        user_preferences,
        n_recommendations=5,
        n_workers=4
    )
    
    # 展示前三个用户的推荐结果
    for user_id in list(batch_results.keys())[:3]:
        print(f"\n用户 {user_id} 的推荐:")
        for vid, score in batch_results[user_id]:
            video = videos_df.iloc[vid]
            print(f"- {video['title']} ({video['category']}): 推荐度 {score:.2f}")
    
    print("\n冷启动推荐系统总结:")
    print("1. 基于拓扑数据分析预计算高质量内容节点，实现高效冷启动")
    print("2. 支持基于类别偏好和用户特征的个性化冷启动")
    print("3. 利用FAISS实现大规模特征索引和快速检索")
    print("4. 通过多进程并行处理支持上亿用户的批量推荐")
    print("5. 冷启动推荐考虑内容热度、多样性和拓扑结构")

if __name__ == "__main__":
    main() 
