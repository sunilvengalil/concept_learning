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
from typing import Optional, Callable

from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import Binary
from pydantic.types import StrBytes
from typing import BinaryIO
from base64 import b64encode


# Shared properties
class RawImagesBase(BaseModel):
    rawImageId: Optional[str] = None
    experiment: Optional[str] = None
    epoch:Optional[int]=None
    step: Optional[int] = None
    batch:Optional[int]=None
    timestamp:Optional[datetime]=None
    image:Optional[bytes]=None

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