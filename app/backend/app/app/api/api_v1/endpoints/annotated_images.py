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
            # Insert a record in annotated_images table
            annotated_image_response = crud.annotated_images.create(db=db, obj_in=annotation)

            # Update user annotations:
            user = crud.user.get_by_email(db=db, email=current_user.email)
            user.totalAnnotations += 1
            db.commit()

            # Update raw image annotation score and total annotations
            raw_image = crud.raw_images.get(db=db, id=annotation.rawImageId)
            # Moving average and equal weightage for probability and clarity

            if raw_image.totalAnnotations != 0:
                current_avg_score = ((1.0 * annotation.probability / 100) + (1.0 * annotation.clarity / 100))/2
                raw_image.annotationScore = (raw_image.annotationScore * raw_image.totalAnnotations + current_avg_score) /  (raw_image.totalAnnotations + 1)
                raw_image.totalAnnotations += 1
            else:
                current_avg_score = (1.0 * annotation.probability / 100) + (1.0 * annotation.clarity / 100)
                raw_image.annotationScore = current_avg_score / 2
                raw_image.totalAnnotations += 1
            db.commit()
            ids.append(annotated_image_response.id)
        return f"Annotations: {ids} Insertion successful"
    except Exception as e:
        return e.__str__()
