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

from app.crud.base import CRUDBase
from app.models.annotated_image import AnnotatedImages
from app.schemas.annotated_images import AnnotatedImagesCreate, AnnotatedImagesUpdate


class CRUDAnnotatedImages(CRUDBase[AnnotatedImages, AnnotatedImagesCreate, AnnotatedImagesUpdate]):
    pass


annotated_images = CRUDAnnotatedImages(AnnotatedImages)