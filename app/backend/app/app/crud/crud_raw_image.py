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

from typing import List

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.raw_images import RawImages
from app.schemas.raw_images import RawImagesCreate, RawImagesUpdate


class CRUDRawImages(CRUDBase[RawImages, RawImagesCreate, RawImagesUpdate]):
    pass


raw_images = CRUDRawImages(RawImages)
