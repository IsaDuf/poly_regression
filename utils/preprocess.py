import numpy as np
import torch
import doctest
import warnings


def standardize(data, data_mean=None, data_std=None):
    """Standardize input by centering at zero (subtract mean) and dividing by standard deviation.

    (`data` - `mean`)/`std`

    Parameters
    ----------
    data : array_like
        Data to standardize.
    data_mean : float
        Mean of data
    data_std : float
        Standard deviation of data

    Returns
    -------
    data_out : array_like
        Standardized `data`
    """
    data_out = data.copy()

    flag = False

    # Center at 0
    if data_mean is None:
        flag = True
        data_mean = np.mean(data, axis=0, keepdims=True)
    data_out -= data_mean

    # Make std = 1
    if data_std is None:
        data_std = np.std(data, axis=0, keepdims=True)
    data_out = data / data_std

    if flag:
        return data_out, data_mean, data_std
    else:
        return data_out


def circmean(data, dim=None, high=np.pi, low=-np.pi):
    """Get the circular mean of the data

    Parameters
    ----------
    data:
        data we want the mean of
    dim:
        axis along which means are computed. Default computes the mean of the flattened array.
    low: float or int
        lowest value on the circular line (range)
    high : float or int
        highest value on circular line (range

    Returns
    -------
    data_out :
        angular mean of the data.
        Will return same type as `data` except for ints that are converted to floats.


    Examples
    -------
    >>> a = np.float64(-361)
    >>> circmean(a, low=0, high=360)
    359.0

    >>> a = np.float64(360)
    >>> circmean(a, low=0, high=360)
    0.0

    >>> a = [-361]
    >>> circmean(a, low=0, high=360)
    359.0

    >>> a = np.array([358.345, 362.345, -898, 6.35])
    >>> circmean(a, low=0, high=360)
    2.5204185683871594

    >>> a = np.array([358.345, 362.345, -898, 6.35])
    >>> b = a.reshape(2,2)
    >>> b
    array([[ 358.345,  362.345],
           [-898.   ,    6.35 ]])
    >>> circmean(b, low=0, high=360)
    2.5204185683871594

    >>> circmean(b, low=0, high=360, dim=0)
    array([270.1725,   4.3475])

    >>> circmean(b, low=0, high=360, dim=1)
    array([ 0.345, 94.175])

    >>> tensor_b = torch.tensor(b)
    >>> tensor_b
    tensor([[ 358.3450,  362.3450],
            [-898.0000,    6.3500]], dtype=torch.float64)

    >>> circmean(tensor_b, low=0, high=360)
    tensor(2.5204, dtype=torch.float64)

    >>> circmean(tensor_b, low=0, high=360, dim=0)
    tensor([270.1725,   4.3475], dtype=torch.float64)

    >>> circmean(tensor_b, low=0, high=360, dim=1)
    tensor([ 0.3450, 94.1750], dtype=torch.float64)

    >>> a = [-11*np.pi, 6*np.pi, 11*np.pi]
    >>> circmean(a, low=-np.pi, high=np.pi)
    3.141592653589793

    """
    circ_range = high - low
    if circ_range <= 0:
        raise ValueError('range must have a positive value -> (high-low) > 0')

    data_out, data_type, data_shape = convert_to(data)
    if data_shape == (0,):
        raise ValueError("The mean value of NaN is undefined")

    # rotate to range and convert to radian:
    data_out = rotate_to_range(data_out, low, high)
    data_out = scale_to_range(data_out, new_low=-np.pi, new_high=np.pi, old_low=low, old_high=high)

    if 'torch.Tensor' in data_type:
        this_sum = torch.sum
        this_sin = torch.sin
        this_cos = torch.cos
        this_atan2 = torch.atan2
        if dim is None:
            dim = ()

    else:
        this_sum = np.sum
        this_sin = np.sin
        this_cos = np.cos
        this_atan2 = np.arctan2

    data_out = this_atan2(this_sum(this_sin(data_out), dim), this_sum(this_cos(data_out), dim))

    # rotate to range and convert back to range:
    data_out = rotate_to_range(data_out, low=-np.pi, high=np.pi)
    data_out = scale_to_range(data_out, new_low=low, new_high=high, old_low=-np.pi, old_high=np.pi)

    return data_out


def rotate_to_range(data, low=-np.pi, high=np.pi):
    """Rotate points around a circle of circumference to be within specified range.

     Rotate points around a circle of circumference to be within given range [`low`, `high`)

    Parameters
    ----------
    data:
        data to be rotated to range
    low: float or int
        lowest value on the circular line
    high : float or int
        highest value on circular line

    Returns
    -------
    data_out :
        data with rotated elements to range.
        Will return same type as `data` except for ints that are converted to floats.


    Examples
    -------
    >>> a = np.float64(-361)
    >>> rotate_to_range(a, low=0, high=360)
    359.0

    >>> a = np.float64(360)
    >>> rotate_to_range(a, low=0, high=360)
    0.0

    >>> a = [-361]
    >>> rotate_to_range(a, low=0, high=360)
    [359.0]

    >>> a = np.array([358.345, 362.345, -898, 6.35])
    >>> rotate_to_range(a, low=0, high=360)
    array([358.345,   2.345, 182.   ,   6.35 ])

    >>> a = np.array([358.345, 362.345, -898, 6.35])
    >>> b = a.reshape(2,2)
    >>> rotate_to_range(b, low=0, high=360)
    array([[358.345,   2.345],
           [182.   ,   6.35 ]])

    >>> a = np.array([-11*np.pi, 6*np.pi, 11*np.pi])
    >>> rotate_to_range(a, low=-np.pi, high=np.pi)
    array([-3.14159265,  0.        ,  3.14159265])

    The values returned are in range [`low`, `high`). In this case, because floating point precision,
    rotating `11*np.pi` yields a number that is smaller than np.pi, although it appears to by equal.

    >>> a = np.array([])
    >>> rotate_to_range(a, low=-np.pi, high=np.pi)
    array([], dtype=float64)

    """
    circ_range = high - low

    if circ_range <= 0:
        raise ValueError('range must have a positive value -> (high-low) > 0')

    data_out, data_type, data_shape = convert_to(data)

    if 'torch.Tensor' in data_type:
        this_any = torch.any
        this_to_type = torch.Tensor.to
        this_type = torch.float
    else:
        this_any = np.any
        this_to_type = np.ndarray.astype
        this_type = float

    while this_any(data_out >= high+100*circ_range):
        warnings.warn("High values are far from max_range, this is likely caused by an upstream problem"
                      " such as exploding gradients")
        data_out = data_out - 100*circ_range * this_to_type((data_out >= high+100*circ_range), this_type)

    while this_any(data_out < low-100*circ_range):
        warnings.warn("Low values are far from min_range, this is likely caused by an upstream problem"
                      " such as exploding gradients")
        data_out = data_out + 100*circ_range * this_to_type((data_out < low-100*circ_range), this_type)

    while this_any(data_out >= high+10*circ_range):
        data_out = data_out - 10*circ_range * this_to_type((data_out >= high+10*circ_range), this_type)

    while this_any(data_out < low-10*circ_range):
        data_out = data_out + 10*circ_range * this_to_type((data_out < low-10*circ_range), this_type)

    while this_any(data_out >= high):
        data_out = data_out - circ_range * this_to_type((data_out >= high), this_type)
    while this_any(data_out < low):
        data_out = data_out + circ_range * this_to_type((data_out < low), this_type)

    data_out = convert_back(data_out, data_type, data_shape)

    return data_out


# def circ_rounded(data, val=0.125, low=1.0, high=6.0):
#     """Rounds elements of data to closest value on a circle where low = high.
#
#     Rounds elements of `data` to closest value on a circular range [`low`, `high`) (i.e. where `low` = `high`)
#     For example, the range of the circle in degrees goes from `low` = 0 to `high` = 360 degrees.
#     Values are first rotated within the range, then rounded to the closest step of length val within the range.
#
#     Parameters
#     ---------- --- needs update
#     data: numpy float, int, float, of array
#         value to be rounded
#     val : float or int
#         step size to round to (length of interval between two rounded values)
#     low : float or int
#         lowest value on the circular line
#     high : float or int
#         highest value on circular line
#
#     Returns
#     -------
#     data_out :  ndarray or tensor
#         Rounded data elements within range [low, high) in type
#
#     Examples
#     -------
#     >>> a = np.array([-361])
#     >>> circ_rounded(a, val=5, low=0, high=360)
#     array([0.])
#
#     >>> a = np.array([358.45, 362.345, -898, 6.35])
#     >>> circ_rounded(a, val=1, low=0, high=360)
#     array([358.,   2., 182.,   6.])
#
#     >>> a = np.array([358.45, 362.345, -898, 6.35])
#     >>> circ_rounded(a, val=0.125, low=0, high=360)
#     array([358.5  ,   2.375, 182.   ,   6.375])
#
#     >>> a = np.array([358.45, 362.345, -898, 6.35])
#     >>> b = a.reshape(2,2)
#     >>> circ_rounded(b, low=0, high=360)
#     array([[358.5  ,   2.375],
#            [182.   ,   6.375]])
#
#     >>> a = np.array([-11*np.pi, 6*np.pi, 11*np.pi])
#     >>> circ_rounded(a, np.pi, low=-np.pi, high=np.pi)
#     array([-3.14159265,  0.        , -3.14159265])
#
#     >>> a = np.array([])
#     >>> circ_rounded(a)
#     array([], dtype=float64)
#
#     """
#     # limit range of data to circumference value range
#     data_out = rotate_to_range(data, low=low, high=high)
#
#     data_out, data_type, data_shape = convert_to(data_out)
#
#     if 'torch.Tensor' in str(type(data_out)):
#         this_round = torch.round
#     else:
#         this_round = np.around
#
#     # fixes minor numerical inconsistencies that occurs at lower and upper bound of range
#     data_out[data_out == high] = high - 1e-8
#     data_out[data_out == low] = low + 1e-8
#
#     # round values
#     data_out = val * this_round(data_out/val)
#
#     # get consistent value outputs
#     data_out[data_out == -0] = 0
#     data_out[data_out == high] = low
#
#     data_out = convert_back(data_out, data_type, data_shape)
#
#     return data_out
#
#
def label_to_class(data, num_map_class=5, low=1, high=6.0):
    """Discretize continuous labels to corresponding class labels based on range intervals of equal width.

    Discretize continuous label `data` to corresponding class label based on range intervals of equal width.
    Intervals' widths are calculated by dividing the data range by the number of classes we wish to map to.
    Elements of data that are out of range are rotated before they are discretized.
    Classes are integers of range [0, `num_map_class`-1].

    NOTE: `data` will be rotated to range but is not rescaled.
           Make sure data is properly scaled before calling this function.

    Parameters
    ----------
    data : array_like
        continuous labels we wish to map to classes labels
    num_map_class : int
        number of classes (bins) to map to
    low : float or int
        lowest value on the circular line
    high : float or int
        highest value on circular line

    Returns
    -------
    data_class: ndarray
        ndarray of dtype int containing the discretized class labels (range [0, num_map_class-1])
        for each of the elements
        of data.

    Examples
    -------
    >>> a = np.array([0.2, 5.3, 2.4, 8.6, 5.1, 3.1, 1.2, 1.1])
    >>> label_to_class(a, num_map_class=2, low=-2.5, high=2.5)
    array([1, 1, 1, 0, 1, 0, 1, 1])

    >>> label_to_class(a, 5, -2.5, 2.5)
    array([2, 2, 4, 1, 2, 0, 3, 3])

    >>> a = np.array([358.45, 362.345, -898, 6.35])
    >>> label_to_class(a, 5, -2.5, 2.5)
    array([0, 4, 4, 3])

    >>> label_to_class(a, 10, 0, 360)
    array([9, 0, 5, 0])

    Rescalling a to the label range:
    >>> a = scale_to_range(a, -2.5, 2.5, 0, 360)
    >>> label_to_class(a, 5, -2.5, 2.5)
    array([4, 0, 2, 0])

    >>> label_to_class(a, 10, -2.5, 2.5)
    array([9, 0, 5, 0])

    >>> a = np.array([358.45, 362.345, -898, 6.35])
    >>> b = a.reshape(2,2)
    >>> label_to_class(b, 10, 0, 360)
    array([[9, 0],
           [5, 0]])

    >>> a = np.array([])
    >>> label_to_class(a, 10, 0, 360)
    array([], dtype=int64)
    """

    data_out, data_type, data_shape = convert_to(data)

    if 'torch.Tensor' in data_type:
        data_copy = data_out.detach()
        this_zeros_like = torch.zeros_like
        this_arange = torch.arange
        this_logical_and = torch.logical_and

    else:
        data_copy = data_out.copy()
        this_zeros_like = np.zeros_like
        this_arange = np.arange
        this_logical_and = np.logical_and

    data_class = this_zeros_like(data_copy, dtype=int)

    # rotate elements to range
    data_copy = rotate_to_range(data_copy, low=low, high=high)
    data_copy[data_copy == high] = low

    # set intervals (bins) boundaries
    bins_range = (high - low)/num_map_class
    bins_bound = this_arange(low+bins_range, high+bins_range, bins_range)

    for cl in range(1, num_map_class):
        data_class[this_logical_and(data_copy >= bins_bound[cl-1], data_copy < bins_bound[cl])] = cl

    data_class = convert_back(data_class, data_type, data_shape)

    return data_class


def scale_to_range(data, new_low=1, new_high=6.0, old_low=-np.pi, old_high=np.pi):
    """Scales data from a range to another.

    Scales `data` from a range [`old_low`, `old_high`] to another [`new_low`, `new_high`].
    Default is used in data generation (`loader.py`) where the labels are in range [-pi, pi] and are rescaled to
    [min_range, max_range] and inversly.

    Parameters
    ----------
    data : array like
        data to rescale
    new_low : float or int
        lower bound of new range
    new_high : float or int
        upper bound of new range
    old_low : float or int
        lower bound of old range
    old_high : float or int
        upper bound of old range

    Returns
    -------
    data_scaled : ndarray or tensor
        data rescaled to new range

    Examples
    -------
    >>> a = np.array([-np.pi, -np.pi/2, 0 , np.pi/2, np.pi])
    >>> scale_to_range(a, -np.pi, np.pi)
    array([-3.14159265, -1.57079633,  0.        ,  1.57079633,  3.14159265])

    >>> scale_to_range(a)
    array([1.  , 2.25, 3.5 , 4.75, 6.  ])

    >>> scale_to_range(a, -2.5, 2.5)
    array([-2.5 , -1.25,  0.  ,  1.25,  2.5 ])

    >>> scale_to_range(a, 0, 360)
    array([  0.,  90., 180., 270., 360.])

    >>> a_rotate = rotate_to_range(a, 0, 2*np.pi)
    >>> scale_to_range(a_rotate, 0, 360, 0, 2*np.pi)
    array([180., 270.,   0.,  90., 180.])

    """

    return (data - old_low) * (new_high - new_low) / (old_high - old_low) + new_low


# def utc_to_labels(data, time_zone='US/Eastern'):
#     """
#     Converts UTC timestamp to labels hour, day, week with respect to `time_zone`.
#
#
#     Parameters
#     ----------
#     data : float or int
#         the UTC timestamp to convert to labels
#
#     time_zone : str
#         time zone of the desired labels. Default is 'US/Eastern'. Set to None for
#         local time zone. See pytz module for list of timezone of run `pytz.all_timezones`.
#
#     Returns
#     -------
#
#
#     """
#
#     tz = timezone(time_zone)
#     date_time = datetime.datetime.fromtimestamp(data)
#     date_time = date_time.astimezone(tz)
#
#     hour = np.float32(date_time.time().hour + date_time.time().minute/60)
#     day = np.float32(date_time.weekday())
#     week = np.float32(date_time.isocalendar()[1])
#
#     return hour, day, week
#
#
def convert_to(data):
    """Convert data to numpy array

    Parameters
    ----------
    data
        data to be converted

    Returns
    -------
    data_out: ndarray or tensor
        data casted in ndarray, or clone of tensor
    data_type: str
        original type of data
    data_shape: tuple
        shape of ndarray
    """

    data_type = str(type(data))
    if 'numpy' in data_type:
        dt = data.dtype
    else:
        dt = 'float64'

    if 'torch' not in data_type:
        data_shape = np.array(data).shape
        data_out = np.array(data, dtype=dt, ndmin=1)

    else:
        data_shape = data.shape
        data_out = data.clone()

    return data_out, data_type, data_shape


def convert_back(data_out, data_type, data_shape):
    """Convert back data_out into original type

    Convert back `data_out` in its original `data_type`, except for int that are converted in floats.

    Parameters
    ----------
    data_out : ndarray or tensor
        data to convert back of its original type
    data_type : str
        original type of `data`
    data_shape : tuple
        shape of an ndarray if no minimum dimensions are set. Checks for scalar types.


    Returns
    -------
    data_out
        `data_out` casted in its original `data_type`, except for int that are converted in floats.

    """

    if data_shape == () and 'torch' not in data_type:
        data_out = data_out[0]

    if 'numpy' not in data_type and 'torch' not in data_type:
        data_out = data_out.tolist()

    return data_out


if __name__ == "__main__":

    doctest.testmod()
