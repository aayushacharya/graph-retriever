from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from pydantic import BaseModel, Field


class QAFormat(BaseModel):
    """Data model for a single question in the dataset"""
    question_id: str = Field(description="Unique identifier for the question")
    question_text: str = Field(description="Text of the question")
    answers: List[str] = Field(description="List of correct answers")
    metadata: Dict[str, Any] = Field(description="Additional metadata for the question")

class Dataset(ABC):
    """Abstract base class for dataset implementations"""
    
    @abstractmethod
    def load(self) -> None:
        """Load the dataset"""
        pass

    @abstractmethod
    def get_train_data(self) -> List[QAFormat]:
        """Get training data"""
        pass

    @abstractmethod
    def get_dev_data(self) -> List[QAFormat]:
        """Get development data"""
        pass

    @abstractmethod
    def get_test_data(self) -> List[QAFormat]:
        """Get test data"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass



