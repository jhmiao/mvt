from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import datetime as dt

@dataclass
class Nurse:
    id: int
    role: str                       # "RN" or "LVN"
    home_location: Union[Tuple[float, float], int]
    availability: List[Tuple[dt.datetime, dt.datetime]] = field(default_factory=list)
    name: Optional[str] = None
    max_hours_per_day: float = 8.0
    max_hours_per_week: float = 40.0

