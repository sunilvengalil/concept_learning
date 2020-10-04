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
from base64 import b64encode


class CRUDRawImages(CRUDBase[RawImages, RawImagesCreate, RawImagesUpdate]):

    def get_next_image(self, db: Session):
        # Fetch the least sample with annotation score (ascending order)
        rm = db.query(self.model).order_by(self.model.annotationScore.asc()).first()

        # Encode image to base64 string for JSON
        rm.image = b64encode(rm.image).decode('utf-8')
        return rm


raw_images = CRUDRawImages(RawImages)
