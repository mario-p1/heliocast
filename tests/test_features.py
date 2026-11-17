import pandas as pd

from solar_predictor.features import resample_to_hourly


def test_resample_to_hourly():
    df = pd.DataFrame(
        {
            "value": {
                "2020-06-10 00:00": 1.0,
                "2020-06-10 00:30": 2.0,
                "2020-06-10 01:00": 4.0,
                "2020-06-10 01:30": 6.0,
                "2020-06-10 02:00": 8.0,
            }
        }
    )
    df.index = pd.to_datetime(df.index, utc=True)

    result = resample_to_hourly(df)

    expected = pd.DataFrame(
        {
            "value": {
                "2020-06-10 00:00": 1.0,
                "2020-06-10 01:00": 3.0,
                "2020-06-10 02:00": 7.0,
            }
        }
    )
    expected.index = pd.to_datetime(expected.index, utc=True)
    pd.testing.assert_frame_equal(result, expected, check_freq=False)
