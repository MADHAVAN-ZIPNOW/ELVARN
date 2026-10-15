"""Copy the mail settings from your local .env to the ELVARN site on Render, then restart the site so it uses them.

    python scripts/push_mail_settings.py            # do it
    python scripts/push_mail_settings.py --dry-run  # only say what would be set

Needs RENDER_API_KEY and SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD (a Gmail "app password") / OWNER_EMAIL in .env.
Values are never printed. Without SMTP_PASSWORD nothing is changed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deploy_render as dr  # noqa: E402

KEYS = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "OWNER_EMAIL")


def push(api, env, dry_run=False, say=print):
    missing = [k for k in KEYS if not env.get(k)]
    if missing:
        raise dr.DeployError("these are empty in .env: " + ", ".join(missing))
    site = next((s for s in api.services() if s["name"] == dr.SITE), None)
    if site is None:
        raise dr.DeployError(f'no service named "{dr.SITE}" on this Render account')
    if dry_run:
        say("(dry run) would set " + ", ".join(KEYS) + f' on "{dr.SITE}" and restart it')
        return
    for k in KEYS:
        api.call("PUT", f"/v1/services/{site['id']}/env-vars/{k}", json={"value": env[k]})
    api.call("POST", f"/v1/services/{site['id']}/deploys", json={})
    say("mail settings set on the site; it is restarting (about 2 minutes)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    env = dr.read_env()
    if not env.get("RENDER_API_KEY"):
        print("No RENDER_API_KEY in .env.", file=sys.stderr)
        return 2
    try:
        push(dr.Render(env["RENDER_API_KEY"]), env, a.dry_run)
    except dr.DeployError as exc:
        secret = env.get("SMTP_PASSWORD", "")
        print("STOPPED:", str(exc).replace(secret, "***") if secret else str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
