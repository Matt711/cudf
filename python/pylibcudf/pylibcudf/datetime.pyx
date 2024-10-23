# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.datetime cimport (
    add_calendrical_months as cpp_add_calendrical_months,
    ceil_datetimes as cpp_ceil_datetimes,
    datetime_component,
    day_of_year as cpp_day_of_year,
    days_in_month as cpp_days_in_month,
    extract_datetime_component as cpp_extract_datetime_component,
    extract_day as cpp_extract_day,
    extract_hour as cpp_extract_hour,
    extract_microsecond_fraction as cpp_extract_microsecond_fraction,
    extract_millisecond_fraction as cpp_extract_millisecond_fraction,
    extract_minute as cpp_extract_minute,
    extract_month as cpp_extract_month,
    extract_nanosecond_fraction as cpp_extract_nanosecond_fraction,
    extract_quarter as cpp_extract_quarter,
    extract_second as cpp_extract_second,
    extract_weekday as cpp_extract_weekday,
    extract_year as cpp_extract_year,
    floor_datetimes as cpp_floor_datetimes,
    is_leap_year as cpp_is_leap_year,
    last_day_of_month as cpp_last_day_of_month,
    round_datetimes as cpp_round_datetimes,
    rounding_frequency,
)

from pylibcudf.libcudf.datetime import \
    datetime_component as DatetimeComponent  # no-cython-lint
from pylibcudf.libcudf.datetime import \
    rounding_frequency as RoundingFrequency  # no-cython-lint

from cython.operator cimport dereference

from .column cimport Column


cpdef Column extract_year(
    Column input
):
    """
    Extract the year from a datetime column.

    For details, see :cpp:func:`extract_year`.

    Parameters
    ----------
    input : Column
        The column to extract the year from.

    Returns
    -------
    Column
        Column with the extracted years.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_year(input.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_month(
    Column input
):
    """
    Extract the month from a datetime column.

    For details, see :cpp:func:`extract_month`.

    Parameters
    ----------
    input : Column
        The column to extract the month from.

    Returns
    -------
    Column
        Column with the extracted months.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_month(input.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_day(
    Column input
):
    """
    Extract the day from a datetime column.

    For details, see :cpp:func:`extract_day`.

    Parameters
    ----------
    input : Column
        The column to extract the day from.

    Returns
    -------
    Column
        Column with the extracted days.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_day(input.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_weekday(
    Column input
):
    """
    Extract the weekday from a datetime column.

    For details, see :cpp:func:`extract_weekday`.

    Parameters
    ----------
    input : Column
        The column to extract the weekday from.

    Returns
    -------
    Column
        Column with the extracted weekdays.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_weekday(input.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_hour(
    Column input
):
    """
    Extract the hour from a datetime column.

    For details, see :cpp:func:`extract_hour`.

    Parameters
    ----------
    input : Column
        The column to extract the hour from.

    Returns
    -------
    Column
        Column with the extracted hours.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_hour(input.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_minute(
    Column input
):
    """
    Extract the minute from a datetime column.

    For details, see :cpp:func:`extract_minute`.

    Parameters
    ----------
    input : Column
        The column to extract the minute from.

    Returns
    -------
    Column
        Column with the extracted minutes.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_minute(input.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_second(
    Column input
):
    """
    Extract the second from a datetime column.

    For details, see :cpp:func:`extract_second`.

    Parameters
    ----------
    input : Column
        The column to extract the second from.

    Returns
    -------
    Column
        Column with the extracted seconds.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_second(input.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_millisecond_fraction(
    Column input
):
    """
    Extract the millisecond from a datetime column.

    For details, see :cpp:func:`extract_millisecond_fraction`.

    Parameters
    ----------
    input : Column
        The column to extract the millisecond from.

    Returns
    -------
    Column
        Column with the extracted milliseconds.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_millisecond_fraction(input.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_microsecond_fraction(
    Column input
):
    """
    Extract the microsecond fraction from a datetime column.

    For details, see :cpp:func:`extract_microsecond_fraction`.

    Parameters
    ----------
    input : Column
        The column to extract the microsecond fraction from.

    Returns
    -------
    Column
        Column with the extracted microsecond fractions.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_microsecond_fraction(input.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_nanosecond_fraction(
    Column input
):
    """
    Extract the nanosecond fraction from a datetime column.

    For details, see :cpp:func:`extract_nanosecond_fraction`.

    Parameters
    ----------
    input : Column
        The column to extract the nanosecond fraction from.

    Returns
    -------
    Column
        Column with the extracted nanosecond fractions.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_nanosecond_fraction(input.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_datetime_component(
    Column input,
    datetime_component component
):
    """
    Extract a datetime component from a datetime column.

    For details, see :cpp:func:`cudf::extract_datetime_component`.

    Parameters
    ----------
    input : Column
        The column to extract the component from.
    component : DatetimeComponent
        The datetime component to extract.

    Returns
    -------
    Column
        Column with the extracted component.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_datetime_component(input.view(), component)
    return Column.from_libcudf(move(result))

cpdef Column ceil_datetimes(
    Column input,
    rounding_frequency freq
):
    """
    Round datetimes up to the nearest multiple of the given frequency.

    For details, see :cpp:func:`ceil_datetimes`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    freq : rounding_frequency
        The frequency to round up to.

    Returns
    -------
    Column
        Column of the same datetime resolution as the input column.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_ceil_datetimes(input.view(), freq)
    return Column.from_libcudf(move(result))

cpdef Column floor_datetimes(
    Column input,
    rounding_frequency freq
):
    """
    Round datetimes down to the nearest multiple of the given frequency.

    For details, see :cpp:func:`floor_datetimes`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    freq : rounding_frequency
        The frequency to round down to.

    Returns
    -------
    Column
        Column of the same datetime resolution as the input column.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_floor_datetimes(input.view(), freq)
    return Column.from_libcudf(move(result))

cpdef Column round_datetimes(
    Column input,
    rounding_frequency freq
):
    """
    Round datetimes to the nearest multiple of the given frequency.

    For details, see :cpp:func:`round_datetimes`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    freq : rounding_frequency
        The frequency to round to.

    Returns
    -------
    Column
        Column of the same datetime resolution as the input column.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_round_datetimes(input.view(), freq)
    return Column.from_libcudf(move(result))

cpdef Column add_calendrical_months(
    Column input,
    ColumnOrScalar months,
):
    """
    Adds or subtracts a number of months from the datetime
    type and returns a timestamp column that is of the same
    type as the input timestamps column.

    For details, see :cpp:func:`add_calendrical_months`.

    Parameters
    ----------
    input : Column
        The column of input timestamp values.
    months : ColumnOrScalar
        The number of months to add.

    Returns
    -------
    Column
        Column of computed timestamps.
    """
    if not isinstance(months, (Column, Scalar)):
        raise TypeError("Must pass a Column or Scalar")

    cdef unique_ptr[column] result

    with nogil:
        result = cpp_add_calendrical_months(
            input.view(),
            months.view() if ColumnOrScalar is Column else
            dereference(months.get())
        )
    return Column.from_libcudf(move(result))

cpdef Column day_of_year(Column input):
    """
    Computes the day number since the start of
    the year from the datetime. The value is between
    [1, {365-366}].

    For details, see :cpp:func:`day_of_year`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.

    Returns
    -------
    Column
        Column of day numbers.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_day_of_year(input.view())
    return Column.from_libcudf(move(result))

cpdef Column is_leap_year(Column input):
    """
    Check if the year of the given date is a leap year.

    For details, see :cpp:func:`is_leap_year`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.

    Returns
    -------
    Column
        Column of bools indicating whether the given year
        is a leap year.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_is_leap_year(input.view())
    return Column.from_libcudf(move(result))

cpdef Column last_day_of_month(Column input):
    """
    Computes the last day of the month.

    For details, see :cpp:func:`last_day_of_month`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.

    Returns
    -------
    Column
        Column of ``TIMESTAMP_DAYS`` representing the last day
        of the month.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_last_day_of_month(input.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_quarter(Column input):
    """
    Returns the quarter (ie. a value from {1, 2, 3, 4})
    that the date is in.

    For details, see :cpp:func:`extract_quarter`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.

    Returns
    -------
    Column
        Column indicating which quarter the date is in.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_quarter(input.view())
    return Column.from_libcudf(move(result))

cpdef Column days_in_month(Column input):
    """
    Extract the number of days in the month.

    For details, see :cpp:func:`days_in_month`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.

    Returns
    -------
    Column
        Column of the number of days in the given month.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_days_in_month(input.view())
    return Column.from_libcudf(move(result))
