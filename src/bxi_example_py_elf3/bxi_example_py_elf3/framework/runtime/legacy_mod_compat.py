"""Compatibility for origin/dev Mods predating explicit orientation guards."""

_ZERO_TORQUE = "com.bxi.basic_actions/zero_torque"

# These API-4 states checked ctx.is_orientation_unsafe() in on_update. Inference
# now pauses on the first unsafe frame, so their old on_update guard cannot run.
LEGACY_ORIENTATION_TARGETS = {
    f"com.bxi.basic_actions/{name}": _ZERO_TORQUE
    for name in (
        "normal",
        "normal_run",
        "amp_run",
        "dance",
        "applause",
        "hello",
    )
}


def orientation_guard_active(state: object) -> bool:
    return bool(getattr(state, "orientation_guard_enabled", False)) or (
        getattr(state, "name", None) in LEGACY_ORIENTATION_TARGETS
    )


def legacy_orientation_target(state: object) -> str | None:
    if getattr(state, "orientation_guard_enabled", False):
        return None
    return LEGACY_ORIENTATION_TARGETS.get(getattr(state, "name", None))
