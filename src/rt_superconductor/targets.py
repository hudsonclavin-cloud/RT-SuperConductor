"""Shared target windows for room-temperature superconductivity research."""

from dataclasses import dataclass


ROOM_TEMPERATURE_MIN_K = 273.15  # 0 °C
ROOM_TEMPERATURE_MAX_K = 298.15  # 25 °C
ATMOSPHERIC_PRESSURE_GPA = 0.000101325
# Provisional engineering-screening ceiling: 100 MPa (1,000 bar). A qualified
# pressure-vessel engineer must confirm the practical limit for any experiment.
ENGINEERING_PRESSURE_CEILING_GPA = 0.1


@dataclass(frozen=True, slots=True)
class TargetWindow:
    """A temperature/pressure window against which a candidate is screened."""

    name: str
    max_pressure_gpa: float
    min_tc_k: float = ROOM_TEMPERATURE_MIN_K
    max_tc_k: float = ROOM_TEMPERATURE_MAX_K

    def accepts(self, tc_k: float, pressure_gpa: float) -> bool:
        return (
            self.min_tc_k <= tc_k <= self.max_tc_k
            and 0 <= pressure_gpa <= self.max_pressure_gpa
        )


AMBIENT_TARGET = TargetWindow("ambient", ATMOSPHERIC_PRESSURE_GPA)
ENGINEERING_TARGET = TargetWindow(
    "engineering_pressure", ENGINEERING_PRESSURE_CEILING_GPA
)
