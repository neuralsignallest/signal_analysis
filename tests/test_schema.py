"""Every schema in eio_crewbench.api.schema round-trips through JSON."""

from __future__ import annotations

import json
from typing import get_args

import pytest
from pydantic import ValidationError

from eio_crewbench.api import schema as s

# --------------------------------------------------------------------------- #
# Example instances - one per model class
# --------------------------------------------------------------------------- #

ACTION_EXAMPLES: list[s.StrictModel] = [
    s.SetFan(fan_id="fan_lab_1", speed_frac=0.75),
    s.SetValve(valve_id="valve_o2_lab", open_frac=0.0),
    s.SetHatch(hatch_id="hatch_node_airlock", position=s.HatchPosition.CLOSED),
    s.BedCommand(bed_id="bed_a", mode=s.BedMode.DESORB),
    s.RunTest(test_id="fan_bit", target_id="fan_lab_1"),
    s.Isolate(scope=s.IsolationScope.COMPARTMENT, target_id="airlock"),
    s.VentOverboard(compartment=s.CompartmentId.AIRLOCK, valve_id="vent_al", duration_s=30.0),
    s.RequestCrew(
        task="inspect fan_lab_1", compartment=s.CompartmentId.LAB, priority=s.Priority.HIGH
    ),
    s.RequestRobot(task="swap filter", compartment=s.CompartmentId.LAB, robot_id="ivr_1"),
    s.AskGround(question="Confirm bed_a desorb is safe on current bus load?"),
    s.StartProcedure(procedure_id="proc_co2_high", from_step=2),
    s.PauseProcedure(procedure_id="proc_co2_high"),
    s.ResumeProcedure(),
    s.SetTimer(timer_id="t_recheck", duration_s=600.0, label="re-check ppCO2"),
    s.Wait(duration_s=60.0),
]


def _compartment_gas() -> s.CompartmentGas:
    return s.CompartmentGas(n_o2_mol=850.0, n_co2_mol=1.5, n_n2_mol=3200.0)


def _vehicle_state() -> s.VehicleState:
    return s.VehicleState(
        t_s=120.0,
        compartments={cid: _compartment_gas() for cid in s.CompartmentId},
        fans={"fan_lab_1": s.FanState(speed_frac=0.8)},
        valves={"valve_o2_lab": s.ValveState(open_frac=0.5)},
        hatches={"hatch_node_lab": s.HatchState(position=s.HatchPosition.OPEN)},
        beds={"bed_a": s.BedState(mode=s.BedMode.ADSORB, loading_frac=0.3, cycle_elapsed_s=900.0)},
        bus_load_w=1450.0,
        battery_soc=0.92,
    )


def _channel_history() -> s.ChannelHistory:
    return s.ChannelHistory(unit="kPa", t_s=[0.0, 1.0, 2.0], values=[0.40, None, 0.42])


def _alarm() -> s.AlarmEvent:
    return s.AlarmEvent(
        t_s=2.0,
        alarm_id="ppco2_caution_lab",
        severity=s.AlarmSeverity.CAUTION,
        channel="ppco2_lab",
        message="ppCO2 above caution threshold",
    )


def _service_call() -> s.ServiceCall:
    return s.ServiceCall(
        call_id="sc_001",
        kind=s.ServiceKind.CREW,
        status=s.ServiceStatus.EN_ROUTE,
        requested_t_s=5.0,
        eta_s=300.0,
        target_id="fan_lab_1",
        note="crew member awake in habitation",
    )


def _procedure() -> s.ProcedureState:
    return s.ProcedureState(
        procedure_id="proc_co2_high",
        status=s.ProcedureStatus.RUNNING,
        step_index=2,
        n_steps=7,
        started_t_s=10.0,
    )


def _comm() -> s.CommState:
    return s.CommState(regime=s.CommRegime.DELAYED, one_way_latency_s=1.28, next_window_t_s=None)


def _shield_rejection() -> s.ShieldRejection:
    return s.ShieldRejection(
        t_s=3.0,
        action=s.VentOverboard(compartment=s.CompartmentId.LAB, duration_s=10.0),
        rule_id="no_vent_occupied",
        reason="lab is occupied",
    )


def _timer() -> s.TimerState:
    return s.TimerState(timer_id="t_recheck", expires_t_s=720.0, label="re-check ppCO2")


def _observation() -> s.Observation:
    return s.Observation(
        t_s=120.0,
        channels={"ppco2_lab": _channel_history()},
        alarms=[_alarm()],
        service_calls=[_service_call()],
        procedure=_procedure(),
        comm=_comm(),
        shield_rejections=[_shield_rejection()],
        timers=[_timer()],
    )


def _sensor_overlay() -> s.SensorOverlay:
    return s.SensorOverlay(kind=s.SensorOverlayKind.BIAS, channels=["ppco2_lab"], magnitude=0.05)


def _onset() -> s.OnsetWindow:
    return s.OnsetWindow(start_s=600.0, end_s=1800.0)


def _scenario_template() -> s.ScenarioTemplate:
    return s.ScenarioTemplate(
        template_id="co2_bed_degradation_01",
        family=s.ScenarioFamily.CO2_SCRUBBING,
        root_cause=s.RootCause.COMPONENT_DEGRADATION,
        component="bed_a",
        severity=s.SeverityBand.MODERATE,
        sensor_overlay=_sensor_overlay(),
        onset=_onset(),
        latency=s.LatencyRegime.LUNAR,
        crew=s.CrewSetting.CREW_ASLEEP,
        split=s.Split.TRAIN,
        seed=42,
    )


EXAMPLES: list[s.StrictModel] = [
    _compartment_gas(),
    s.FanState(speed_frac=0.8),
    s.ValveState(open_frac=0.5),
    s.HatchState(position=s.HatchPosition.OPEN),
    s.BedState(mode=s.BedMode.ADSORB, loading_frac=0.3),
    _vehicle_state(),
    *ACTION_EXAMPLES,
    _channel_history(),
    _alarm(),
    _service_call(),
    _procedure(),
    _comm(),
    _shield_rejection(),
    _timer(),
    _observation(),
    _sensor_overlay(),
    _onset(),
    _scenario_template(),
]

ALL_MODEL_CLASSES: list[type[s.StrictModel]] = sorted(
    {
        obj
        for name in s.__all__
        if isinstance(obj := getattr(s, name), type)
        and issubclass(obj, s.StrictModel)
        and obj is not s.StrictModel
    },
    key=lambda c: c.__name__,
)


# --------------------------------------------------------------------------- #
# Coverage: every exported model has an example, every action type is listed
# --------------------------------------------------------------------------- #


def test_every_exported_model_has_an_example() -> None:
    covered = {type(m) for m in EXAMPLES}
    missing = [c.__name__ for c in ALL_MODEL_CLASSES if c not in covered]
    assert not missing, f"no round-trip example for: {missing}"


def test_action_union_lists_every_action_type() -> None:
    union_members = set(get_args(get_args(s.Action)[0]))
    assert union_members == set(s.ACTION_TYPES)
    assert {type(a) for a in ACTION_EXAMPLES} == set(s.ACTION_TYPES)


def test_action_kinds_are_unique_literals() -> None:
    kinds = [get_args(t.model_fields["kind"].annotation)[0] for t in s.ACTION_TYPES]
    assert len(kinds) == len(set(kinds)), f"duplicate discriminators: {kinds}"


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", EXAMPLES, ids=lambda m: type(m).__name__)
def test_model_round_trips_through_json(model: s.StrictModel) -> None:
    text = model.model_dump_json()
    back = type(model).model_validate_json(text)
    assert back == model
    assert json.loads(back.model_dump_json()) == json.loads(text)
    # and via a plain dict, as an HTTP layer would hand it over
    assert type(model).model_validate(json.loads(text)) == model


@pytest.mark.parametrize("action", ACTION_EXAMPLES, ids=lambda a: type(a).__name__)
def test_action_round_trips_through_discriminated_union(action: s.StrictModel) -> None:
    text = s.ActionAdapter.dump_json(action)
    back = s.ActionAdapter.validate_json(text)
    assert type(back) is type(action)
    assert back == action


@pytest.mark.parametrize("cls", [s.VehicleState, s.Observation, s.ScenarioTemplate])
def test_top_level_models_emit_json_schema(cls: type[s.StrictModel]) -> None:
    schema = cls.model_json_schema()
    assert schema["type"] == "object"
    assert "properties" in schema


def test_action_union_emits_json_schema_with_discriminator() -> None:
    schema = s.ActionAdapter.json_schema()
    assert "oneOf" in schema
    assert schema["discriminator"]["propertyName"] == "kind"
    assert len(schema["oneOf"]) == len(s.ACTION_TYPES)


# --------------------------------------------------------------------------- #
# Validation rules
# --------------------------------------------------------------------------- #


def test_unknown_action_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        s.ActionAdapter.validate_python({"kind": "launch_missiles", "target": "x"})


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        s.Wait(duration_s=1.0, reason="extra")  # type: ignore[call-arg]


def test_vehicle_state_needs_every_compartment() -> None:
    partial = {s.CompartmentId.LAB: _compartment_gas()}
    with pytest.raises(ValidationError, match="missing entries"):
        s.VehicleState(t_s=0.0, compartments=partial, bus_load_w=0.0, battery_soc=1.0)


def test_channel_history_arrays_must_align() -> None:
    with pytest.raises(ValidationError, match="samples"):
        s.ChannelHistory(unit="kPa", t_s=[0.0, 1.0], values=[1.0])
    with pytest.raises(ValidationError, match="non-decreasing"):
        s.ChannelHistory(unit="kPa", t_s=[1.0, 0.0], values=[1.0, 2.0])


def test_unit_fractions_are_bounded() -> None:
    with pytest.raises(ValidationError):
        s.SetFan(fan_id="f", speed_frac=1.5)
    with pytest.raises(ValidationError):
        s.VehicleState(
            t_s=0.0,
            compartments={cid: _compartment_gas() for cid in s.CompartmentId},
            bus_load_w=0.0,
            battery_soc=-0.1,
        )


def test_scenario_component_must_match_root_cause() -> None:
    base = _scenario_template().model_dump()
    base.update(root_cause="none", component="bed_a")
    with pytest.raises(ValidationError, match="component must be None"):
        s.ScenarioTemplate.model_validate(base)
    base.update(root_cause="leak", component=None)
    with pytest.raises(ValidationError, match="needs a component"):
        s.ScenarioTemplate.model_validate(base)


def test_sensor_overlay_none_takes_no_channels() -> None:
    with pytest.raises(ValidationError):
        s.SensorOverlay(kind=s.SensorOverlayKind.NONE, channels=["ppco2_lab"])
    with pytest.raises(ValidationError):
        s.SensorOverlay(kind=s.SensorOverlayKind.DRIFT, channels=[])
    assert s.SensorOverlay(kind=s.SensorOverlayKind.NONE).channels == []


def test_onset_window_is_ordered() -> None:
    with pytest.raises(ValidationError):
        s.OnsetWindow(start_s=10.0, end_s=5.0)
