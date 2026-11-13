
import numpy as np

def add_booking_features(data):
    data = data.copy()

    data["total_nights"] = (
        data["stays_in_weekend_nights"] +
        data["stays_in_week_nights"]
    )

    data["total_guests"] = (
        data["adults"] +
        data["children"] +
        data["babies"]
    )

    data["is_family"] = (
        (data["children"] + data["babies"]) > 0
    ).astype(int)

    return data


def convert_to_string(data):
    return np.asarray(data).astype(str)
