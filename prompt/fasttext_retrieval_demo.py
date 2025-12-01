#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastText实现检索功能演示

本文件演示如何使用FastText实现类似原始检索功能的文本检索系统，
包括文本向量表示、相似度计算和结果排序等核心功能。
"""

import numpy as np
import fasttext
import fasttext.util
import pandas as pd
from typing import List, Dict, Any, Optional


class FastTextRetriever:
    """
    使用FastText实现的文本检索系统
    
    该系统可以将文本转换为向量表示，并基于向量相似度进行检索和排序。
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化FastText检索器
        
        参数:
            model_path: FastText预训练模型路径，若为None则下载wiki预训练模型
        """
        self.model = None
        self.records = []
        self.record_vectors = None
        
        # 加载FastText模型
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str]):
        """
        加载FastText模型
        
        参数:
            model_path: 模型路径
        """
        try:
            if model_path:
                self.model = fasttext.load_model(model_path)
            else:
                print("下载预训练FastText模型...")
                # 下载英文wiki预训练模型
                fasttext.util.download_model('en', if_exists='ignore')
                self.model = fasttext.load_model('cc.en.300.bin')
            print("FastText模型加载成功!")
        except Exception as e:
            print(f"警告: FastText模型加载失败: {e}")
            print("将使用简单的TF-IDF作为替代")
            self._init_tfidf_fallback()
    
    def _init_tfidf_fallback(self):
        """
        初始化TF-IDF作为FastText不可用时的备选方案
        """
        self.vocab = {}
        self.idf = {}
        self.doc_count = 0
    
    def fit(self, records: List[Dict[str, Any]]):
        """
        训练检索器，为所有记录生成向量表示
        
        参数:
            records: 包含文本字段的记录列表
        """
        self.records = records
        self.doc_count = len(records)
        
        # 如果没有text字段，尝试使用其他文本字段
        text_field = 'text' if any('text' in r for r in records) else next(
            (k for k in records[0].keys() if isinstance(records[0][k], str)), None
        )
        
        if not text_field:
            raise ValueError("记录中没有找到文本字段")
        
        self.text_field = text_field
        
        # 为TF-IDF备选方案构建词汇表
        self._build_vocab()
        
        # 生成记录向量
        self.record_vectors = np.array([
            self._get_text_vector(record[text_field]) 
            for record in records
        ])
        
        print(f"已为{len(records)}条记录生成向量表示")
    
    def _build_vocab(self):
        """
        构建词汇表和IDF统计
        """
        # 统计每个词出现在多少文档中
        doc_freq = {}
        
        for record in self.records:
            text = record.get(self.text_field, '').lower()
            words = set(text.split())
            
            for word in words:
                doc_freq[word] = doc_freq.get(word, 0) + 1
                self.vocab[word] = len(self.vocab)
        
        # 计算IDF
        for word, freq in doc_freq.items():
            self.idf[word] = np.log(self.doc_count / (freq + 1)) + 1
    
    def _get_text_vector(self, text: str) -> np.ndarray:
        """
        获取文本的向量表示
        
        参数:
            text: 输入文本
        
        返回:
            文本的向量表示
        """
        if self.model:
            # 使用FastText获取向量
            return self.model.get_sentence_vector(text)
        else:
            # 使用TF-IDF备选方案
            return self._get_tfidf_vector(text)
    
    def _get_tfidf_vector(self, text: str) -> np.ndarray:
        """
        获取文本的TF-IDF向量表示
        
        参数:
            text: 输入文本
        
        返回:
            TF-IDF向量
        """
        words = text.lower().split()
        tf = {}
        
        # 计算TF
        for word in words:
            tf[word] = tf.get(word, 0) + 1
        
        # 归一化TF
        max_tf = max(tf.values()) if tf else 1
        
        # 创建向量
        vector = np.zeros(len(self.vocab))
        for word, count in tf.items():
            if word in self.vocab:
                tf_value = count / max_tf
                vector[self.vocab[word]] = tf_value * self.idf.get(word, 1)
        
        return vector
    
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        基于查询文本检索记录
        
        参数:
            query: 查询文本
            **kwargs: 额外的过滤参数
                - sort: 排序配置，包含value(排序字段)和ordering(排序方向)
                - 其他键值对用于过滤
        
        返回:
            排序后的检索结果
        """
        # 生成查询向量
        query_vector = self._get_text_vector(query)
        
        # 计算相似度
        similarities = self._compute_similarities(query_vector)
        
        # 过滤记录
        filtered_indices = self._filter_records(**kwargs)
        
        # 对过滤后的结果按相似度排序
        results = []
        for idx in filtered_indices:
            record = self.records[idx].copy()
            record['similarity'] = similarities[idx]
            results.append(record)
        
        # 应用排序
        if 'sort' in kwargs:
            key = kwargs['sort']['value']
            reverse = kwargs['sort']['ordering'] == 'descend'
            results = sorted(results, key=lambda x: x.get(key, 0), reverse=reverse)
        else:
            # 默认按相似度降序排序
            results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        return results
    
    def _compute_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """
        计算查询向量与所有记录向量的相似度
        
        参数:
            query_vector: 查询向量
        
        返回:
            相似度数组
        """
        # 归一化向量
        query_norm = np.linalg.norm(query_vector)
        record_norms = np.linalg.norm(self.record_vectors, axis=1)
        
        # 计算余弦相似度
        dot_products = np.dot(self.record_vectors, query_vector)
        
        # 避免除零错误
        denominators = np.maximum(query_norm * record_norms, 1e-10)
        
        similarities = dot_products / denominators
        
        return similarities
    
    def _filter_records(self, **kwargs) -> List[int]:
        """
        根据条件过滤记录
        
        参数:
            **kwargs: 过滤条件
        
        返回:
            符合条件的记录索引列表
        """
        filtered_indices = []
        
        for idx, record in enumerate(self.records):
            select = True
            
            # 处理requirement过滤
            if record.get("requirement"):
                if "status" not in kwargs or kwargs["status"] != record["requirement"]:
                    continue
            
            # 处理其他过滤条件
            for k, v in kwargs.items():
                if k in ["sort", "status"]:  # 跳过排序和已经处理的status
                    continue
                
                if k not in record:
                    select = False
                    break
                
                # 处理特殊操作符
                if isinstance(v, dict):
                    if "operator" in v:
                        # 安全地执行操作符比较
                        try:
                            lhs = record[k]
                            rhs = v["value"]
                            
                            # 构建比较表达式
                            if v["operator"] == ">":
                                if not (lhs > rhs):
                                    select = False
                                    break
                            elif v["operator"] == "<":
                                if not (lhs < rhs):
                                    select = False
                                    break
                            elif v["operator"] == ">=":
                                if not (lhs >= rhs):
                                    select = False
                                    break
                            elif v["operator"] == "<=":
                                if not (lhs <= rhs):
                                    select = False
                                    break
                            elif v["operator"] == "==":
                                if not (lhs == rhs):
                                    select = False
                                    break
                            elif v["operator"] == "!=":
                                if not (lhs != rhs):
                                    select = False
                                    break
                        except Exception:
                            select = False
                            break
                    elif "value" in v:
                        if v["value"] == "无上限" and record[k] != 1000:
                            select = False
                            break
                        elif str(record[k]) != str(v["value"]):
                            select = False
                            break
                elif str(record[k]) != str(v):
                    select = False
                    break
            
            if select:
                filtered_indices.append(idx)
        
        return filtered_indices


def create_sample_data() -> List[Dict[str, Any]]:
    """
    创建示例数据用于演示
    
    返回:
        示例记录列表
    """
    return [
        {
            "id": 1,
            "text": "Deep learning is a subset of machine learning",
            "price": 100,
            "category": "technology",
            "requirement": "premium"
        },
        {
            "id": 2,
            "text": "FastText is a library for efficient text classification",
            "price": 200,
            "category": "nlp",
            "requirement": "standard"
        },
        {
            "id": 3,
            "text": "Transformer models have revolutionized NLP",
            "price": 150,
            "category": "nlp",
            "requirement": "premium"
        },
        {
            "id": 4,
            "text": "Vector similarity search is essential for recommendation systems",
            "price": 300,
            "category": "search",
            "requirement": "standard"
        },
        {
            "id": 5,
            "text": "Word embeddings capture semantic relationships between words",
            "price": 250,
            "category": "nlp",
            "requirement": "premium"
        }
    ]


def main():
    """
    主函数，演示FastText检索功能
    """
    print("="*80)
    print("使用FastText实现检索功能演示")
    print("="*80)
    
    # 创建示例数据
    print("\n1. 创建示例数据:")
    data = create_sample_data()
    print(f"   创建了{len(data)}条示例记录")
    
    # 初始化检索器
    print("\n2. 初始化FastText检索器:")
    retriever = FastTextRetriever()
    
    # 训练检索器
    print("\n3. 训练检索器:")
    retriever.fit(data)
    
    # 执行检索
    print("\n4. 执行检索:")
    
    # 示例1: 简单文本查询
    print("\n   示例1: 简单文本查询 'natural language processing'")
    results1 = retriever.retrieve("natural language processing")
    for i, result in enumerate(results1[:3], 1):
        print(f"     {i}. ID: {result['id']}, 相似度: {result['similarity']:.4f}, 文本: {result['text'][:50]}...")
    
    # 示例2: 带过滤条件的查询
    print("\n   示例2: 带过滤条件的查询 (category='nlp')")
    results2 = retriever.retrieve("word embeddings", category="nlp")
    for i, result in enumerate(results2, 1):
        print(f"     {i}. ID: {result['id']}, 相似度: {result['similarity']:.4f}, 类别: {result['category']}")
    
    # 示例3: 带排序的查询
    print("\n   示例3: 带排序的查询 (按price升序)")
    results3 = retriever.retrieve(
        "machine learning", 
        sort={"value": "price", "ordering": "ascend"}
    )
    for i, result in enumerate(results3, 1):
        print(f"     {i}. ID: {result['id']}, 价格: {result['price']}, 相似度: {result['similarity']:.4f}")
    
    # 示例4: 复杂查询（带操作符）
    print("\n   示例4: 复杂查询 (price>150)")
    results4 = retriever.retrieve(
        "deep learning", 
        price={"operator": ">", "value": 150}
    )
    for i, result in enumerate(results4, 1):
        print(f"     {i}. ID: {result['id']}, 价格: {result['price']}, 相似度: {result['similarity']:.4f}")
    
    print("\n5. 与原始检索功能的对比:")
    print("   - 原始功能: 基于精确字段匹配和排序")
    print("   - FastText实现: 增加了文本语义理解和向量相似度计算")
    print("   - 优势: 能够处理语义相似但字面不同的查询")
    print("   - 扩展性: 可以轻松添加更多文本字段和复杂的相似度计算策略")


if __name__ == "__main__":
    main()
