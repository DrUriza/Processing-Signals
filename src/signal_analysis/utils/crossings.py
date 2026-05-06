from __future__                    import annotations
import pandas                      as pd
from signal_analysis.utils.helpers import to_series

# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          UTILS
# MODULE NAME:        crossings.py
# DESCRIPTION:        @brief Cross detection helpers for series and scalar levels
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.2$
# CHANGES:            22.04.2026 - Migrated to short banner comments.
#                     22.04.2026 - Refactored to class-based OOP.
# *****************************************************************************

class CrossingDetectors:
    # ***********************************************************************************************************************
    # Functionname:       CrossingDetectors.cross_above(a, b)
    #
    # @brief              Detect upward crossing of series a over series b.
    # @pre                Inputs must be coercible to aligned pandas Series.
    # @post               Returns boolean Series with True at upward cross points.
    # @param[in]          a: First input sequence
    #                     b: Second input sequence
    # @param[out]         out: Boolean crossing mask
    #
    # @callsequence       @startuml
    #                     title CrossingDetectors.cross_above
    #                     start
    #                       :Convert both inputs to pandas Series;
    #                       :Compare current values and prior values;
    #                       :Return upward crossing mask;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def cross_above(a, b) -> pd.Series:
        sa = to_series(a, name="a")
        sb = to_series(b, name="b")
        # Current bar above, previous bar at or below.
        return (sa > sb) & (sa.shift(1) <= sb.shift(1))

    # ***********************************************************************************************************************
    # Functionname:       CrossingDetectors.cross_below(a, b)
    #
    # @brief              Detect downward crossing of series a below series b.
    # @pre                Inputs must be coercible to aligned pandas Series.
    # @post               Returns boolean Series with True at downward cross points.
    # @param[in]          a: First input sequence
    #                     b: Second input sequence
    # @param[out]         out: Boolean crossing mask
    #
    # @callsequence       @startuml
    #                     title CrossingDetectors.cross_below
    #                     start
    #                       :Convert both inputs to pandas Series;
    #                       :Compare current values and prior values;
    #                       :Return downward crossing mask;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def cross_below(a, b) -> pd.Series:
        sa = to_series(a, name="a")
        sb = to_series(b, name="b")
        # Current bar below, previous bar at or above.
        return (sa < sb) & (sa.shift(1) >= sb.shift(1))

    # ***********************************************************************************************************************
    # Functionname:       CrossingDetectors.cross_level_up(series, level: float)
    #
    # @brief              Detect upward crossing through a scalar level.
    # @pre                Series must be coercible to pandas Series.
    # @post               Returns boolean Series with True at upward level-cross points.
    # @param[in]          series: Input sequence
    #                     level: Scalar threshold
    # @param[out]         out: Boolean crossing mask
    #
    # @callsequence       @startuml
    #                     title CrossingDetectors.cross_level_up
    #                     start
    #                       :Convert input to pandas Series;
    #                       :Compare current and prior values against level;
    #                       :Return upward level-cross mask;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def cross_level_up(series, level: float) -> pd.Series:
        s = to_series(series)
        return (s > level) & (s.shift(1) <= level)

    # ***********************************************************************************************************************
    # Functionname:       CrossingDetectors.cross_level_down(series, level: float)
    #
    # @brief              Detect downward crossing through a scalar level.
    # @pre                Series must be coercible to pandas Series.
    # @post               Returns boolean Series with True at downward level-cross points.
    # @param[in]          series: Input sequence
    #                     level: Scalar threshold
    # @param[out]         out: Boolean crossing mask
    #
    # @callsequence       @startuml
    #                     title CrossingDetectors.cross_level_down
    #                     start
    #                       :Convert input to pandas Series;
    #                       :Compare current and prior values against level;
    #                       :Return downward level-cross mask;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def cross_level_down(series, level: float) -> pd.Series:
        s = to_series(series)
        return (s < level) & (s.shift(1) >= level)

# ---------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ---------------------------------------------------------------------------
def cross_above(a, b) -> pd.Series:
    return CrossingDetectors.cross_above(a, b)

def cross_below(a, b) -> pd.Series:
    return CrossingDetectors.cross_below(a, b)

def cross_level_up(series, level: float) -> pd.Series:
    return CrossingDetectors.cross_level_up(series, level)

def cross_level_down(series, level: float) -> pd.Series:
    return CrossingDetectors.cross_level_down(series, level)

__all__ = ["CrossingDetectors", "cross_above", "cross_below", "cross_level_up", "cross_level_down"]
