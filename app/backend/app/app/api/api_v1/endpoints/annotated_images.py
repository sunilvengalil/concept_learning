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

from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import typing

from app import crud, models, schemas
from app.api import deps

router = APIRouter()


@router.post("/", response_model=str)
def push_annotated_images(
        annotations: typing.List[schemas.AnnotatedImages],
        db: Session = Depends(deps.get_db),
        current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve items.
    """
    # if crud.user.is_superuser(current_user):
    ids = []
    try:
        for annotation in annotations:
            annotated_image_response = crud.annotated_images.create(db=db, obj_in=annotation)
            ids.append(annotated_image_response.annotationId)
        return f"Annotations: {ids} Insertion successful"
    except Exception as e:
        return e.__str__()
