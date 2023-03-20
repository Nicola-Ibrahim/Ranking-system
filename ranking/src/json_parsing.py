import datetime
from json import JSONEncoder

import dateutil.parser


class DateTimeEncoder(JSONEncoder):
    # Override the default method
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()


def decode_date_time(space_dict) -> dict:
    """Decode string date to be datetime object

    Args:
        space_dict (_type_): space's date dict

    Returns:
        dict: a new formatted space dict
    """
    if "start" in space_dict:
        space_dict["start"] = dateutil.parser.parse(space_dict["start"])

    if "end" in space_dict:
        space_dict["end"] = dateutil.parser.parse(space_dict["end"])

    return space_dict
