from typing import Optional
from pydantic import BaseModel


class ContentInput(BaseModel):
    title: str
    num_movie: Optional[int] = 20


class CollaborativeInput(BaseModel):
    user_id: int
    movie_id: int


class HybridInput(BaseModel):
    user_id: int
    title: str
    num_movie: Optional[int] = 20
