# -*- coding: utf-8 -*-
"""
| **@created on:** 10/5/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** 
"""

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Column, Integer, String, Binary, TIMESTAMP, Float, ForeignKey
from app.db.base_class import Base

if TYPE_CHECKING:
    from .item import Item  # noqa: F401


class AnnotatedImages(Base):
    __tablename__ = "annotated_images"
    id = Column(Integer, primary_key=True, index=True)
    rawImageId = Column(Integer, ForeignKey("raw_images.id"))
    uniqueId = Column(String)
    label = Column(String)
    probability = Column(Float)
    clarity = Column(Float)
    timestamp = Column(TIMESTAMP)
    userEmail = Column(String, ForeignKey("user.email"))
