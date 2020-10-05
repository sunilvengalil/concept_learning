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
from app.utils import running_average
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
    Push annotated image
    """

    ids = []
    try:
        for annotation in annotations:
            # Insert a record in annotated_images table
            annotation.userEmail = current_user.email
            annotated_image_response = crud.annotated_images.create(db=db, obj_in=annotation)

            # Update user annotations:
            user = crud.user.get_by_email(db=db, email=current_user.email)
            user.totalAnnotations += 1
            db.commit()

            # Update raw image annotation score and total annotations
            raw_image = crud.raw_images.get(db=db, id=annotation.rawImageId)

            # Update avgProbability
            if raw_image.avgProbability == 0:
                raw_image.avgProbability = annotation.probability / 100
            else:
                raw_image.avgProbability = running_average(previous_avg=raw_image.avgProbability,
                                                           current_val=annotation.probability / 100,
                                                           total_records=raw_image.totalAnnotations)

            # Update avgClarity
            if raw_image.avgClarity == 0:
                raw_image.avgClarity = annotation.clarity / 100
            else:
                raw_image.avgClarity = running_average(previous_avg=raw_image.avgClarity,
                                                       current_val=annotation.clarity / 100,
                                                       total_records=raw_image.totalAnnotations)

            # Moving average and equal weightage for probability and clarity
            if raw_image.totalAnnotations == 0:
                current_avg_score = (1.0 * annotation.probability / 100) + (1.0 * annotation.clarity / 100)
                raw_image.annotationScore = current_avg_score / 2

            else:
                current_avg_score = ((1.0 * annotation.probability / 100) + (1.0 * annotation.clarity / 100)) / 2
                raw_image.annotationScore = running_average(previous_avg=raw_image.annotationScore,
                                                            current_val=current_avg_score,
                                                            total_records=raw_image.totalAnnotations)
            raw_image.totalAnnotations += 1
            db.commit()
            ids.append(annotated_image_response.id)
        return f"Annotations: {ids} Insertion successful"
    except Exception as e:
        return e.__str__()
