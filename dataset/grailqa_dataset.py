from dataset import Dataset, QAFormat
from typing import List, Dict, Any
import numpy as np
import os
import json

class GrailQADataset(Dataset):
    """Abstract base class for dataset implementations"""

    def __init__(self, data_path: str = os.path.join("data","grailqa","GrailQA_v1.0")):
        self.name="GrailQADataset"
        self.train_path=os.path.join(data_path,"grailqa_v1.0_train.json")
        self.dev_path=os.path.join(data_path,"grailqa_v1.0_dev.json")
        self.test_path=os.path.join(data_path,"grailqa_v1.0_test_public.json")

        self.load()    
    
    def load(self) -> None:
        """Load the dataset"""
        with open(self.train_path, 'r') as f:
            self.train = json.load(f)
        with open(self.dev_path, 'r') as f:
            self.dev = json.load(f)
        with open(self.test_path, 'r') as f:
            self.test = json.load(f)

    def _convert_to_qaformat(self, data: List[Dict[str, Any]]) -> List[QAFormat]:
        qa_data = []
        for item in data:
            answers_grailqa=item.get("answers", [])
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

    
