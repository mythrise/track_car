"""Isolated preformal F2 experiment implementation.

This package is deliberately outside the frozen main-v1 scripts and
OpenTrackVLA source tree.  It implements only the Fable-approved
``L1+D2+AP2+F2`` candidate contract and must not be imported by any
main-v1 formal lifecycle (v8/v9 and successors).
"""

from .controller import (
    ACTION_FILTER_ESTIMAND,
    CONTROLLER_CONFIG_CONTRACT_SHA256,
    ActionFilterConfig,
    ActionFilterController,
    ActionFilterState,
    ActionFilterTransition,
    bind_controller_identity,
)
from .support import (
    ARCHITECTURE_LOCK,
    FROZEN_TRAIN_ROWS,
    FROZEN_TRAIN_SHA256,
    F2ContractError,
    FrozenSupportReceipt,
    build_frozen_support,
    build_frozen_support_from_payload,
)

__all__ = [
    "ACTION_FILTER_ESTIMAND",
    "ARCHITECTURE_LOCK",
    "CONTROLLER_CONFIG_CONTRACT_SHA256",
    "FROZEN_TRAIN_ROWS",
    "FROZEN_TRAIN_SHA256",
    "ActionFilterConfig",
    "ActionFilterController",
    "ActionFilterState",
    "ActionFilterTransition",
    "F2ContractError",
    "FrozenSupportReceipt",
    "build_frozen_support",
    "build_frozen_support_from_payload",
    "bind_controller_identity",
]
