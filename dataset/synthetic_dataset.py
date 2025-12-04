from dataset import Dataset, QAFormat
from typing import List, Dict, Any
import numpy as np
import os
import json

class SyntheticDataset(Dataset):
    """Abstract base class for dataset implementations"""

    def __init__(self):
        self.name="SyntheticDataset"
        self.load()    
    
    def load(self) -> None:
        """Load the dataset"""
        with open('sample_dataset.json', 'r') as f:
            sample_dataset = json.load(f)
        self.train=sample_dataset
        self.dev=sample_dataset
        self.test=sample_dataset

    def _convert_to_qaformat(self, data: List[Dict[str, Any]]) -> List[QAFormat]:
        qa_data = []
        for item in data:
            answers_grailqa=item.get("answer", [])
            answers=[]
            for ans in answers_grailqa:
                answers.append(ans.get("answer_argument", ""))
            
            qa_item = QAFormat(
                question_id=str(item.get("qid", "")),
                question_text=item.get("question", ""),
                answers=answers,
                metadata=item.copy()
            )
            qa_data.append(qa_item)
        return qa_data

    def get_train_data(self) -> List[QAFormat]:
        return self._convert_to_qaformat(self.train)
    
    def get_dev_data(self) -> List[QAFormat]:
        return self._convert_to_qaformat(self.dev)
    
    def get_test_data(self) -> List[QAFormat]:
        return self._convert_to_qaformat(self.test)
    
    def __str__(self) -> str:
        return self.name


    
