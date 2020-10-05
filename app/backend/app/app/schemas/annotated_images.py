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

from typing import Optional
from pydantic import BaseModel
from datetime import datetime


# Shared properties
class AnnotatedImagesBase(BaseModel):
    annotationId: Optional[int] = None
    rawImageId: Optional[int] = None
    uniqueId: Optional[str] = None
    label: Optional[str] = None
    probability: Optional[float] = None
    clarity: Optional[float] = None
    userId: Optional[int] = None
    timestamp: Optional[datetime] = None


# Properties shared by models stored in DB
class AnnotatedImagesInDBBase(AnnotatedImagesBase):
    class Config:
        orm_mode = True


# Properties to receive via API on creation
class AnnotatedImagesCreate(AnnotatedImagesBase):
    pass


# Properties to receive via API on update
class AnnotatedImagesUpdate(AnnotatedImagesBase):
    pass


# Properties to return to client
class AnnotatedImages(AnnotatedImagesBase):
    pass
