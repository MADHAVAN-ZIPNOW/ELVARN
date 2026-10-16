"""The operations the dashboard can ask for, in one place: the local server calls them directly, the agent calls them for the hosted site.
Each takes a Monitor (which knows the registry and the control file) and returns plain JSON-able data."""
from backend.pipeline import collect_request, control

ANSWER_WAIT = 12.0


def start(monitor):
    control.start(monitor.control_path)
    return {"ok": True}


def stop(monitor):
    control.request_stop(monitor.control_path)
    return {"ok": True}


def targeted(monitor, body):
    """'collect C. Joseph Vijay ED cases' (or a candidate the owner picked): resolve the person and start only those sources.
    -> the `request` object of the API: {ok, message?, person?, sources?, stages?, needs_choice?, candidates?}"""
    if control.is_running(monitor.control_path):
        return {"ok": False, "message": "A collection is running. Stop it first (or wait), then ask again."}
    try:
        plan = collect_request.plan(monitor.registry, body.get("text"), body.get("person"), body.get("sources"))
    except Exception as exc:  # noqa: BLE001 - Wikidata unreachable and the like
        return {"ok": False, "message": f"Could not look the person up: {str(exc)[:120]}"}
    if not plan["ok"]:
        return plan
    groups = [g for g in plan["sources"] if g != "all"]
    extra = ("--person", plan["person"]["key"]) + (("--sources", ",".join(groups)) if groups else ())
    wording = "all sources" if not groups else ", ".join(groups)
    control.start(monitor.control_path, extra=extra, message=f"Collecting {wording} for {plan['person']['name']} ...")
    return plan


def people(monitor, args):
    return monitor.people_page(args.get("status"), args.get("q"), args.get("limit", 50), args.get("offset", 0))


def person(monitor, args):
    return monitor.person_detail(args.get("key")) or {}                       # {} = unknown person (None would look like "the computer did not answer")


COMMANDS = {"start": start, "stop": stop, "targeted": targeted, "people": people, "person": person}


def run(monitor, action, args):
    """Run one command from the hosted site. Never raises: a failure is an answer ({"error": ...}), so the dashboard can say so."""
    fn = COMMANDS.get(action)
    if fn is None:
        return {"error": f"unknown command {action!r}"}
    try:
        return fn(monitor, args or {}) if action in ("targeted", "people", "person") else fn(monitor)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{exc.__class__.__name__}: {str(exc)[:160]}"}
