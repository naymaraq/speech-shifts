import decimal
import json
import os
import yaml

from dateutil import parser
import subprocess

__all__ = [
    "get_env",
    "get_envbool",
    "get_envint",
    "get_envfloat",
    "get_envdecimal",
    "get_envdate",
    "get_envdatetime",
    "get_envlist",
    "get_envdict",
    "CoercionError",
    "RequiredSettingMissingError",
]


class CoercionError(Exception):
    """Custom error raised when a value cannot be coerced."""

    def __init__(self, key, value, func):
        msg = "Unable to coerce '{}={}' using {}.".format(key, value, func.__name__)
        super(CoercionError, self).__init__(msg)


class RequiredSettingMissingError(Exception):
    """Custom error raised when a required env var is missing."""

    def __init__(self, key):
        msg = "Required env var '{}' is missing.".format(key)
        super(RequiredSettingMissingError, self).__init__(msg)


def _get_env(key, default=None, coerce=lambda x: x, required=False):
    """
    Return env var coerced into a type other than string.
    This function extends the standard os.getenv function to enable
    the coercion of values into data types other than string (all env
    vars are strings by default).
    Args:
        key: string, the name of the env var to look up
    Kwargs:
        default: the default value to return if the env var does not exist. NB the
            default value is **not** coerced, and is assumed to be of the correct type.
        coerce: a function that is used to coerce the value returned into
            another type
        required: bool, if True, then a RequiredSettingMissingError error is raised
            if the env var does not exist.
    Returns the env var, passed through the coerce function
    """
    try:
        value = os.environ[key]
    except KeyError:
        if required is True:
            raise RequiredSettingMissingError(key)
        else:
            return default

    try:
        return coerce(value)
    except Exception:
        raise CoercionError(key, value, coerce)


# standard type coercion functions
def _bool(value):
    if isinstance(value, bool):
        return value

    return not (value is None or value.lower() in ("false", "0", "no", "n", "f", "none"))


def _int(value):
    return int(value)


def _float(value):
    return float(value)


def _decimal(value):
    return decimal.Decimal(value)


def _dict(value):
    return json.loads(value)


def _datetime(value):
    return parser.parse(value)


def _date(value):
    return parser.parse(value).date()


def get_env(key, *default, **kwargs):
    """
    Return env var.
    This is the parent function of all other get_foo functions,
    and is responsible for unpacking args/kwargs into the values
    that _get_env expects (it is the root function that actually
    interacts with environ).
    Args:
        key: string, the env var name to look up.
        default: (optional) the value to use if the env var does not
            exist. If this value is not supplied, then the env var is
            considered to be required, and a RequiredSettingMissingError
            error will be raised if it does not exist.
    Kwargs:
        coerce: a func that may be supplied to coerce the value into
            something else. This is used by the default get_foo functions
            to cast strings to builtin types, but could be a function that
            returns a custom class.
    Returns the env var, coerced if required, and a default if supplied.
    """
    assert len(default) in (0, 1), "Too many args supplied."
    func = kwargs.get('coerce', lambda x: x)
    required = len(default) == 0
    default = default[0] if not required else None
    return _get_env(key, default=default, coerce=func, required=required)


def get_envbool(key, *default):
    """Return env var cast as boolean."""
    return get_env(key, *default, coerce=_bool)


def get_envint(key, *default):
    """Return env var cast as integer."""
    return get_env(key, *default, coerce=_int)


def get_envfloat(key, *default):
    """Return env var cast as float."""
    return get_env(key, *default, coerce=_float)


def get_envdecimal(key, *default):
    """Return env var cast as Decimal."""
    return get_env(key, *default, coerce=_decimal)


def get_envdate(key, *default):
    """Return env var as a date."""
    return get_env(key, *default, coerce=_date)


def get_envdatetime(key, *default):
    """Return env var as a datetime."""
    return get_env(key, *default, coerce=_datetime)


def get_envlist(key, *default, **kwargs):
    """Return env var as a list."""
    separator = kwargs.get('separator', ' ')
    return get_env(key, *default, coerce=lambda x: x.split(separator))


def get_envdict(key, *default):
    """Return env var as a dict."""
    return get_env(key, *default, coerce=_dict)


def is_global_rank_zero():
    """ Helper function to determine if the current process is global_rank 0 (the main process)
    """
    # Try to get the pytorch RANK env var
    # RANK is set by torch.distributed.launch
    rank = get_envint("RANK", None)
    if rank:
        return rank == 0

    # If not set by pytorch, we need to determine node_rank
    def get_node_rank():
        # Use an equivalent of pytorch lightning's determine_ddp_node_rank()
        node_rank = 0
        # First check if running on a slurm cluster
        # TODO: This check could probably be better
        num_slurm_tasks = get_envint("SLURM_NTASKS", 0)
        if num_slurm_tasks > 0:
            node_rank = get_envint("SLURM_NODEID", 0)
        else:
            node_rank_env = get_envint("NODE_RANK", None)
            group_rank = get_envint("GROUP_RANK", None)
            if group_rank:
                node_rank = group_rank
            # Take from NODE_RANK whenever available
            if node_rank_env:
                node_rank = node_rank_env
        return node_rank

    node_rank = get_node_rank()
    local_rank = get_envint("LOCAL_RANK", 0)
    return node_rank == 0 and local_rank == 0

def read_yaml(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config

def get_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except:
        return ""

def get_version_folder(dir_path, new=True):
    folders = os.listdir(dir_path)
    if len(folders) == 0:
        return os.path.join(dir_path, 'version_0')
    indexes = [int(folder.split('_')[-1]) for folder in folders]
    if new:
        return os.path.join(dir_path, 'version_' + str(max(indexes) + 1))
    return os.path.join(dir_path, 'version_' + str(max(indexes)))