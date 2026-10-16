"""The operations the dashboard can ask for, in one place: the local server calls them directly, the agent calls them for the hosted site.
Each takes a Monitor (which knows the registry and the control file) and returns plain JSON-able data."""
from backend.pipeline import collect_request, control

ANSWER_WAIT = 12.0


def judgments_args(args):
    """The settings of a judgments run from the dashboard -> command line words. Everything is checked: this is what the site may ask the computer to run."""
    from backend.pipeline import judgments
    args = args or {}
    scope = args.get("scope") if args.get("scope") in judgments.SCOPES else "all"
    def number(value, default, low, high, cast):
        try:
            return min(high, max(low, cast(value if value not in (None, "") else default)))
        except (TypeError, ValueError):
            return default                                                            # one bad setting falls back alone; the others are kept
    gb = number(args.get("max_gb"), float(judgments.DEFAULT_GB), 1.0, 500.0, float)
    first = number(args.get("from_year"), judgments.FIRST_YEAR, judgments.FIRST_YEAR, 2026, int)
    last = number(args.get("to_year"), 2026, first, 2026, int)
    topics = [t for t in (args.get("topics") or []) if t in judgments.SUBJECT_NAMES][:12] if isinstance(args.get("topics"), list) else []
    words = ["--max-gb", f"{gb:g}", "--scope", scope, "--from-year", str(first), "--to-year", str(last)]
    if topics:
        words += ["--topics", ",".join(topics)]
    then = args.get("then") == "people"                                          # run plan C: the politician collector follows by itself
    if then:
        words += ["--then", "people"]
    return words, ("Step 1 of 2: " if then else "") + f"Collecting court judgments from all of India ({scope}, {first}-{last}, up to {gb:g} GB" + (f", about {', '.join(topics)}" if topics else "") + ") ..."


def start(monitor, args=None):
    """No settings: the politician-by-politician collector. {"mode": "judgments", ...}: the all-India judgments collector."""
    if isinstance(args, dict) and args.get("mode") == "judgments":
        if control.is_running(monitor.control_path):
            return {"ok": False, "message": "A collection is running. Stop it first."}
        words, message = judgments_args(args)
        control.start(monitor.control_path, extra=tuple(words), message=message, module=control.JUDGMENTS_MODULE, check_stack=True)
        return {"ok": True}
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
        return fn(monitor, args or {}) if action in ("targeted", "people", "person", "start") else fn(monitor)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{exc.__class__.__name__}: {str(exc)[:160]}"}
