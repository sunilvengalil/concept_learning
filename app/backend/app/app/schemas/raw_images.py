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
from typing import Optional

from pydantic import BaseModel


# Shared properties
class RawImagesBase(BaseModel):
    experiment: Optional[str] = None
    uniqueId: Optional[str] = None


# Properties shared by models stored in DB
class RawImagesInDBBase(RawImagesBase):

    class Config:
        orm_mode = True


# Properties to receive via API on creation
class RawImagesCreate(RawImagesBase):
    pass


# Properties to receive via API on update
class RawImagesUpdate(RawImagesBase):
    pass


# Properties to return to client
class RawImages(RawImagesInDBBase):
    pass