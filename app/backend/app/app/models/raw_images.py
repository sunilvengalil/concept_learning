# -*- coding: utf-8 -*-
"""
| **@created on:** 9/18/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** 
"""

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Column, Integer, String, Binary, TIMESTAMP, Float
from sqlalchemy.orm import relationship

from app.db.base_class import Base

if TYPE_CHECKING:
    from .item import Item  # noqa: F401


class RawImages(Base):
    __tablename__ = "raw_images"
    id = Column(Integer, primary_key=True, index=True)
    experiment = Column(String, index=True)
    epoch = Column(Integer)
    step = Column(Integer)
    batch = Column(Integer)
    uniqueId = Column(String, unique=True)
    image = Column(Binary)
    timestamp = Column(TIMESTAMP)
    avgProbability = Column(Float, default=0.0)
    avgClarity = Column(Float, default=0.0)
    annotationScore = Column(Float, default=0.0)
    fileName = Column(String)
    evalImageId = Column(Integer)
    totalAnnotations = Column(Integer, default=0)
