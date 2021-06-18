from clearn.analysis import ImageConcept


def create_concept_dict(image_list):
    concept_dict = dict()
    for digit, concepts in image_list.items():
        for concept in concepts:
            if digit in concept_dict.keys():
                concept_dict[digit].append(ImageConcept(concept[0],
                                                        concept[1],
                                                        concept[2],
                                                        concept[3],
                                                        concept[4],
                                                        concept[5],
                                                        concept[6])
                                           )
            else:
                concept_dict[digit] = [ImageConcept(concept[0],
                                                    concept[1],
                                                    concept[2],
                                                    concept[3],
                                                    concept[4],
                                                    concept[5],
                                                    concept[6])

                                       ]
    return concept_dict
