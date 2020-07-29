def get_first_key(data_dict):
    if data_dict is None or len(data_dict) == 0:
        first_key = None
    else:
        first_key = next(iter(data_dict))
    print(first_key)
    return first_key
