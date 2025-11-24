from .hopper import HopperEnv
from .swimmer import SwimmerEnv
from .ant import AntEnv
from .gap import GapEnv
from .walker import WalkerEnv
from .stair import StairEnv
from .stairhard import StairhardEnv
from .pusher import PusherEnv

env_dict = {
    'hopper': HopperEnv,
    'swimmer': SwimmerEnv,
    'ant': AntEnv,
    'gap': GapEnv,
    'walker': WalkerEnv,
    'stair': StairEnv,
    'stairhard': StairhardEnv,
    'pusher': PusherEnv,
}