from ._W_basic import W_basic
from ._W_dict import W_dict
from ._W_list import W_list
from ._W_io import W_io
from ._W_tools import W_tools
from ._W_string import W_string
from ._W_namedtuple import W_namedtuple

class W(W_basic, W_dict, W_list, W_io, W_tools, W_string, W_namedtuple):
    def __init__(self) -> None:
        super().__init__()