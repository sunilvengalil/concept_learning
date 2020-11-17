# TODO modify this to get other ways of combining multiple rows (maximum if there is more than one,
#  evidence assembler logic)

"""Returns true if all of the specified column in the  in the row has same value"""


def get_combined_annotation(row, column_names):
    text_0 = row[column_names[0]]
    all_same = True
    for _column_name in column_names[1:]:
        all_same = all_same and (row[_column_name] == text_0)
        if not all_same:
            return False
    return all_same


def has_multiple_value(_column_name, row):
    """ Returns true if the row has multiple words separated by space """
    if len(row[_column_name].split()) > 1:
        return True
    else:
        return False


""" An aggregation function, that can be used with `groupby` which converts a list of strings to single string
 separated by space """


def space_separated_string(x):
    x_as_list = list(x)
    if len(x_as_list) > 1:
        if len(set(x)) == 1:
            return x_as_list[0]
        else:
            x_as_list = x.values.tolist()
            return " ".join(x_as_list)
    else:
        return x
