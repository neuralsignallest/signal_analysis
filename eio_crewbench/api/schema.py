"""Pydantic v2 data contracts for the EIO crew-bench API boundary.

Four top-level types cross the boundary between the environment and an agent:

* :class:`VehicleState` - the simulator's full truth state (not agent-visible).
* :class:`Observation` - what the agent sees at a decision point.
* :class:`Action` - a discriminated union of the commands an agent may issue.
* :class:`ScenarioTemplate` - the generator-side description of one scenario.

This module holds data shapes only. No physics, no dynamics, no defaults that
encode engineering numbers: every engineering quantity lives in
``config/params.yaml`` with a unit, a source and a confidence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #


class StrictModel(BaseModel):
    """Base for every schema: unknown fields are an error, enums stay enums."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


NonNegFloat = Annotated[float, Field(ge=0)]
PosFloat = Annotated[float, Field(gt=0)]
UnitFloat = Annotated[float, Field(ge=0, le=1)]
NonNegInt = Annotated[int, Field(ge=0)]
PosInt = Annotated[int, Field(ge=1)]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")]


class CompartmentId(StrEnum):
    """The four pressurised compartments defined in ``config/topology.yaml``."""

    HABITATION = "habitation"
    NODE = "node"
    LAB = "lab"
    AIRLOCK = "airlock"


class HatchPosition(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class BedMode(StrEnum):
    """Operating mode of a CO2 sorbent bed."""

    ADSORB = "adsorb"
    DESORB = "desorb"
    STANDBY = "standby"
    ISOLATED = "isolated"


class Priority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# --------------------------------------------------------------------------- #
# VehicleState
# --------------------------------------------------------------------------- #


class CompartmentGas(StrictModel):
    """Molar inventory of one compartment."""

    n_o2_mol: NonNegFloat
    n_co2_mol: NonNegFloat
    n_n2_mol: NonNegFloat


class FanState(StrictModel):
    speed_frac: UnitFloat = Field(description="Fraction of rated speed, 0..1.")
    powered: bool = True


class ValveState(StrictModel):
    open_frac: UnitFloat = Field(description="Fraction open, 0 = shut, 1 = full open.")


class HatchState(StrictModel):
    position: HatchPosition


class BedState(StrictModel):
    mode: BedMode
    loading_frac: UnitFloat = Field(description="Fraction of sorbent capacity in use.")
    cycle_elapsed_s: NonNegFloat = 0.0


class VehicleState(StrictModel):
    """Full simulator truth state at time ``t_s``."""

    t_s: NonNegFloat
    compartments: dict[CompartmentId, CompartmentGas]
    fans: dict[Identifier, FanState] = Field(default_factory=dict)
    valves: dict[Identifier, ValveState] = Field(default_factory=dict)
    hatches: dict[Identifier, HatchState] = Field(default_factory=dict)
    beds: dict[Identifier, BedState] = Field(default_factory=dict)
    bus_load_w: NonNegFloat
    battery_soc: UnitFloat = Field(description="Battery state of charge, 0..1.")

    @model_validator(mode="after")
    def _all_compartments_present(self) -> VehicleState:
        missing = set(CompartmentId) - set(self.compartments)
        if missing:
            names = ", ".join(sorted(m.value for m in missing))
            raise ValueError(f"compartments missing entries for: {names}")
        return self


# --------------------------------------------------------------------------- #
# Action (discriminated union)
# --------------------------------------------------------------------------- #


class IsolationScope(StrEnum):
    COMPARTMENT = "compartment"
    COMPONENT = "component"


class SetFan(StrictModel):
    kind: Literal["set_fan"] = "set_fan"
    fan_id: Identifier
    speed_frac: UnitFloat


class SetValve(StrictModel):
    kind: Literal["set_valve"] = "set_valve"
    valve_id: Identifier
    open_frac: UnitFloat


class SetHatch(StrictModel):
    kind: Literal["set_hatch"] = "set_hatch"
    hatch_id: Identifier
    position: HatchPosition


class BedCommand(StrictModel):
    kind: Literal["bed_command"] = "bed_command"
    bed_id: Identifier
    mode: BedMode


class RunTest(StrictModel):
    """Run a built-in test or self-check on a component."""

    kind: Literal["run_test"] = "run_test"
    test_id: Identifier
    target_id: Identifier | None = None


class Isolate(StrictModel):
    """Isolate a compartment (close its hatches and ducts) or a component."""

    kind: Literal["isolate"] = "isolate"
    scope: IsolationScope
    target_id: Identifier


class VentOverboard(StrictModel):
    kind: Literal["vent_overboard"] = "vent_overboard"
    compartment: CompartmentId
    valve_id: Identifier | None = None
    duration_s: PosFloat


class RequestCrew(StrictModel):
    kind: Literal["request_crew"] = "request_crew"
    task: str = Field(min_length=1)
    compartment: CompartmentId | None = None
    priority: Priority = Priority.NORMAL


class RequestRobot(StrictModel):
    kind: Literal["request_robot"] = "request_robot"
    task: str = Field(min_length=1)
    compartment: CompartmentId | None = None
    robot_id: Identifier | None = None
    priority: Priority = Priority.NORMAL


class AskGround(StrictModel):
    kind: Literal["ask_ground"] = "ask_ground"
    question: str = Field(min_length=1)
    priority: Priority = Priority.NORMAL


class StartProcedure(StrictModel):
    kind: Literal["start_procedure"] = "start_procedure"
    procedure_id: Identifier
    from_step: NonNegInt = 0


class PauseProcedure(StrictModel):
    kind: Literal["pause_procedure"] = "pause_procedure"
    procedure_id: Identifier | None = Field(
        default=None, description="None pauses whichever procedure is running."
    )


class ResumeProcedure(StrictModel):
    kind: Literal["resume_procedure"] = "resume_procedure"
    procedure_id: Identifier | None = Field(
        default=None, description="None resumes whichever procedure is paused."
    )


class SetTimer(StrictModel):
    kind: Literal["set_timer"] = "set_timer"
    timer_id: Identifier
    duration_s: PosFloat
    label: str = ""


class Wait(StrictModel):
    kind: Literal["wait"] = "wait"
    duration_s: PosFloat


Action = Annotated[
    SetFan
    | SetValve
    | SetHatch
    | BedCommand
    | RunTest
    | Isolate
    | VentOverboard
    | RequestCrew
    | RequestRobot
    | AskGround
    | StartProcedure
    | PauseProcedure
    | ResumeProcedure
    | SetTimer
    | Wait,
    Field(discriminator="kind"),
]
"""Every command an agent may issue, discriminated on the ``kind`` field."""

ACTION_TYPES: tuple[type[StrictModel], ...] = (
    SetFan,
    SetValve,
    SetHatch,
    BedCommand,
    RunTest,
    Isolate,
    VentOverboard,
    RequestCrew,
    RequestRobot,
    AskGround,
    StartProcedure,
    PauseProcedure,
    ResumeProcedure,
    SetTimer,
    Wait,
)

ActionAdapter: TypeAdapter[Action] = TypeAdapter(Action)
"""Use ``ActionAdapter.validate_json`` / ``dump_json`` to (de)serialise an Action."""


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #


class AlarmSeverity(StrEnum):
    ADVISORY = "advisory"
    CAUTION = "caution"
    WARNING = "warning"
    EMERGENCY = "emergency"


class ServiceKind(StrEnum):
    CREW = "crew"
    ROBOT = "robot"
    GROUND = "ground"


class ServiceStatus(StrEnum):
    REQUESTED = "requested"
    ACKNOWLEDGED = "acknowledged"
    EN_ROUTE = "en_route"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    DENIED = "denied"


class ProcedureStatus(StrEnum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    ABORTED = "aborted"


class CommRegime(StrEnum):
    """Live state of the ground link as the agent experiences it."""

    NOMINAL = "nominal"
    DEGRADED = "degraded"
    DELAYED = "delayed"
    BLACKOUT = "blackout"


class ChannelHistory(StrictModel):
    """Time series for one telemetry channel. ``None`` marks a dropped sample."""

    unit: str = Field(min_length=1)
    t_s: list[NonNegFloat]
    values: list[float | None]

    @model_validator(mode="after")
    def _aligned_and_monotone(self) -> ChannelHistory:
        if len(self.t_s) != len(self.values):
            raise ValueError(
                f"t_s has {len(self.t_s)} samples but values has {len(self.values)}"
            )
        if any(b < a for a, b in zip(self.t_s, self.t_s[1:], strict=False)):
            raise ValueError("t_s must be non-decreasing")
        return self


class AlarmEvent(StrictModel):
    t_s: NonNegFloat
    alarm_id: Identifier
    severity: AlarmSeverity
    channel: Identifier | None = None
    message: str = ""
    acknowledged: bool = False


class ServiceCall(StrictModel):
    call_id: Identifier
    kind: ServiceKind
    status: ServiceStatus
    requested_t_s: NonNegFloat
    eta_s: NonNegFloat | None = None
    target_id: Identifier | None = None
    note: str = ""


class ProcedureState(StrictModel):
    procedure_id: Identifier
    status: ProcedureStatus
    step_index: NonNegInt
    n_steps: PosInt
    started_t_s: NonNegFloat | None = None

    @model_validator(mode="after")
    def _step_in_range(self) -> ProcedureState:
        if self.step_index > self.n_steps:
            raise ValueError(
                f"step_index {self.step_index} exceeds n_steps {self.n_steps}"
            )
        return self


class CommState(StrictModel):
    regime: CommRegime
    one_way_latency_s: NonNegFloat
    next_window_t_s: NonNegFloat | None = Field(
        default=None, description="Sim time the next contact window opens, if known."
    )


class ShieldRejection(StrictModel):
    """An action the safety shield refused to pass to the vehicle."""

    t_s: NonNegFloat
    action: Action
    rule_id: Identifier
    reason: str = Field(min_length=1)


class TimerState(StrictModel):
    timer_id: Identifier
    expires_t_s: NonNegFloat
    label: str = ""
    fired: bool = False


class Observation(StrictModel):
    """Everything the agent is shown at one decision point."""

    t_s: NonNegFloat
    channels: dict[Identifier, ChannelHistory] = Field(default_factory=dict)
    alarms: list[AlarmEvent] = Field(default_factory=list)
    service_calls: list[ServiceCall] = Field(default_factory=list)
    procedure: ProcedureState | None = None
    comm: CommState
    shield_rejections: list[ShieldRejection] = Field(default_factory=list)
    timers: list[TimerState] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# ScenarioTemplate
# --------------------------------------------------------------------------- #


class ScenarioFamily(StrEnum):
    NOMINAL = "nominal"
    CO2_SCRUBBING = "co2_scrubbing"
    O2_SUPPLY = "o2_supply"
    ATMOSPHERE_LEAK = "atmosphere_leak"
    VENTILATION = "ventilation"
    SENSOR_FAULT = "sensor_fault"
    POWER = "power"


class RootCause(StrEnum):
    NONE = "none"
    COMPONENT_DEGRADATION = "component_degradation"
    COMPONENT_FAILURE = "component_failure"
    SENSOR_BIAS = "sensor_bias"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_DROPOUT = "sensor_dropout"
    LEAK = "leak"
    BLOCKAGE = "blockage"
    SOFTWARE_FAULT = "software_fault"
    CREW_ERROR = "crew_error"


class SeverityBand(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SensorOverlayKind(StrEnum):
    NONE = "none"
    NOISE = "noise"
    BIAS = "bias"
    DRIFT = "drift"
    DROPOUT = "dropout"
    STUCK = "stuck"


class LatencyRegime(StrEnum):
    """Which comm-latency envelope the scenario is generated under."""

    LEO = "leo"
    LUNAR = "lunar"
    MARS = "mars"


class CrewSetting(StrEnum):
    UNCREWED = "uncrewed"
    CREW_ASLEEP = "crew_asleep"
    CREW_AWAKE = "crew_awake"
    CREW_EVA = "crew_eva"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SensorOverlay(StrictModel):
    """A sensor-fault overlay applied on top of the true channel values."""

    kind: SensorOverlayKind
    channels: list[Identifier] = Field(default_factory=list)
    magnitude: float | None = Field(
        default=None, description="Overlay magnitude in the channel's own units."
    )

    @model_validator(mode="after")
    def _none_is_empty(self) -> SensorOverlay:
        if self.kind is SensorOverlayKind.NONE and (self.channels or self.magnitude is not None):
            raise ValueError("overlay kind 'none' takes no channels or magnitude")
        if self.kind is not SensorOverlayKind.NONE and not self.channels:
            raise ValueError(f"overlay kind '{self.kind.value}' needs at least one channel")
        return self


class OnsetWindow(StrictModel):
    """Sim-time window inside which the fault onset is drawn."""

    start_s: NonNegFloat
    end_s: NonNegFloat

    @model_validator(mode="after")
    def _ordered(self) -> OnsetWindow:
        if self.end_s < self.start_s:
            raise ValueError(f"end_s {self.end_s} is before start_s {self.start_s}")
        return self


class ScenarioTemplate(StrictModel):
    """Generator-side description of one scenario draw."""

    template_id: Identifier
    family: ScenarioFamily
    root_cause: RootCause
    component: Identifier | None = Field(
        default=None, description="Faulted component id; None only when root_cause is 'none'."
    )
    severity: SeverityBand
    sensor_overlay: SensorOverlay
    onset: OnsetWindow
    latency: LatencyRegime
    crew: CrewSetting
    split: Split
    seed: int | None = None

    @model_validator(mode="after")
    def _component_matches_cause(self) -> ScenarioTemplate:
        if self.root_cause is RootCause.NONE and self.component is not None:
            raise ValueError("component must be None when root_cause is 'none'")
        if self.root_cause is not RootCause.NONE and self.component is None:
            raise ValueError(f"root_cause '{self.root_cause.value}' needs a component")
        return self


__all__ = [
    # building blocks
    "StrictModel",
    "Identifier",
    "CompartmentId",
    "HatchPosition",
    "BedMode",
    "Priority",
    # vehicle state
    "CompartmentGas",
    "FanState",
    "ValveState",
    "HatchState",
    "BedState",
    "VehicleState",
    # actions
    "IsolationScope",
    "SetFan",
    "SetValve",
    "SetHatch",
    "BedCommand",
    "RunTest",
    "Isolate",
    "VentOverboard",
    "RequestCrew",
    "RequestRobot",
    "AskGround",
    "StartProcedure",
    "PauseProcedure",
    "ResumeProcedure",
    "SetTimer",
    "Wait",
    "Action",
    "ACTION_TYPES",
    "ActionAdapter",
    # observation
    "AlarmSeverity",
    "ServiceKind",
    "ServiceStatus",
    "ProcedureStatus",
    "CommRegime",
    "ChannelHistory",
    "AlarmEvent",
    "ServiceCall",
    "ProcedureState",
    "CommState",
    "ShieldRejection",
    "TimerState",
    "Observation",
    # scenario template
    "ScenarioFamily",
    "RootCause",
    "SeverityBand",
    "SensorOverlayKind",
    "LatencyRegime",
    "CrewSetting",
    "Split",
    "SensorOverlay",
    "OnsetWindow",
    "ScenarioTemplate",
]
