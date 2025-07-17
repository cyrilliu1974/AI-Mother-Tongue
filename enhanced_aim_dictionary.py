import json
import os
from datetime import datetime
import time # For timestamp in reflection records

class EnhancedAIMDictionary:
    def __init__(self, file_path="enhanced_aim_dictionary.json"):
        self.file_path = file_path
        self.aim_entries = {}  # {aim_id: {aim_sequence, human_label, first_seen_round, contexts, evolution_trace}}
        self.reflection_records = {}  # {aim_id: [reflection_data_list]}
        self.unified_records_list = [] # Store all unified records for analysis later

        self._load_from_file() # Load existing data

    def _load_from_file(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    self.aim_entries = data.get('aim_entries', {})
                    self.reflection_records = data.get('reflection_records', {})
                    self.unified_records_list = data.get('unified_records_list', [])
                    print(f"[EnhancedAIMDictionary] Loaded {len(self.aim_entries)} AIM entries and {len(self.unified_records_list)} unified records from {self.file_path}")
            except json.JSONDecodeError:
                print(f"Warning: {self.file_path} is corrupted or empty. Starting with empty dictionary.")

    def _generate_aim_id(self, aim_sequence, context, round_num, agent_id):
        # 創建一個更具描述性的AIM_ID，結合序列、回合、代理ID和部分上下文
        # 由於 aim_sequence 是列表，直接用 str(aim_sequence)
        # context 可以是 dict, 這裡簡化為字符串形式
        context_str = str(context.get('context', '')) # 確保上下文為字符串，避免複雜對象作為ID一部分
        return f"{agent_id}_R{round_num}_{str(aim_sequence)}"

    def add_entry_with_reflection(self, aim_sequence, human_interpretation, context, 
                                 agent_reflection_data, round_num, agent_id):
        
        aim_id = self._generate_aim_id(aim_sequence, context, round_num, agent_id)
        
        if aim_id not in self.aim_entries:
            self.aim_entries[aim_id] = {
                'aim_sequence': aim_sequence,
                'human_label': human_interpretation,
                'first_seen_round': round_num,
                'contexts': [], # List of dicts: {'context_detail': ..., 'round': ..., 'success_rate': ..., 'semantic_stability': ...}
                'evolution_trace': [] # 可以用於追蹤AIM語義的演化
            }
        
        # 添加當前上下文詳細信息 (簡化 success_rate 和 semantic_stability 為佔位符)
        current_context_detail = {
            'context_detail': context.get('context_detail', context), # 嘗試獲取詳細上下文，否則用原始context
            'round': round_num,
            'success_rate': self._compute_success_rate(aim_id, round_num), # 簡化計算
            'semantic_stability': self._compute_semantic_stability(aim_id, round_num) # 簡化計算
        }
        self.aim_entries[aim_id]['contexts'].append(current_context_detail)
        
        # 關聯反省記錄
        if aim_id not in self.reflection_records:
            self.reflection_records[aim_id] = []
            
        self.reflection_records[aim_id].append({
            'round': round_num,
            'agent_id': agent_id,
            'reflection_data': agent_reflection_data,
            'human_label': human_interpretation,
            'timestamp': datetime.now().isoformat() # 使用 ISO 格式字符串
        })
        
        return aim_id

    def add_unified_record(self, record):
        """添加完整的統一記錄，方便後續分析"""
        self.unified_records_list.append(record)

    def _compute_success_rate(self, aim_id, current_round):
        # 這是簡化計算，實際需要追踪每個AIM的行為結果（獎勵）
        # 假設如果被解釋為C且獎勵高則成功，D且獎勵高則成功
        # 為了DEMO，返回一個模擬值
        return round(0.5 + 0.5 * (current_round % 100) / 100, 2) # 模擬隨時間變化

    def _compute_semantic_stability(self, aim_id, current_round):
        # 簡化計算：基於特定AIM_ID歷史human_label的一致性
        # 實際需要遍歷 self.reflection_records[aim_id] 統計 human_label 的分佈
        # 為了DEMO，返回一個模擬值
        return round(0.7 + 0.3 * (current_round % 50) / 50, 2) # 模擬隨時間變化

    def save(self):
        data_to_save = {
            'aim_entries': self.aim_entries,
            'reflection_records': self.reflection_records,
            'unified_records_list': self.unified_records_list
        }
        with open(self.file_path, "w") as f:
            json.dump(data_to_save, f, indent=2)
        print(f"[EnhancedAIMDictionary] Saved {len(self.aim_entries)} AIM entries and {len(self.unified_records_list)} unified records to {self.file_path}")