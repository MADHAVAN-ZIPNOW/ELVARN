"""Create / update the ELVARN knowledge service on Render through Render's API, wire the site to it, and check that it is up.

    python scripts/deploy_render.py            # do it
    python scripts/deploy_render.py --dry-run  # show what it would do, change nothing

Needs a Render API key: Render dashboard > Account Settings > API Keys. Put it in your local .env as
    RENDER_API_KEY=rnd_...
(never paste it into a chat; .env is git-ignored). Optional: RENDER_OWNER=<workspace name> if you have several workspaces.

What it does (idempotent, safe to run again):
  1. finds your workspace and the existing "elvarn" site service (to copy its repo, branch and region);
  2. makes a long random KNOWLEDGE_TOKEN (kept in .env, never printed) unless one exists;
  3. creates the "elvarn-knowledge" Docker web service on the FREE plan if it is missing (Dockerfile knowledge_service/Dockerfile, repo root as
     build context, health check /healthz), or just re-sets its token if it already exists;
  4. sets KNOWLEDGE_URL and KNOWLEDGE_TOKEN on the site so it uses the service;
  5. waits for https://<service>.onrender.com/healthz and then /readyz (all three stores must be reachable).
The key and the token are only ever sent to Render / to the service over https and are never written to the screen.
"""
import argparse
import os
import secrets
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
API = "https://api.render.com"
SITE, KNOWLEDGE = "elvarn", "elvarn-knowledge"


class DeployError(Exception):
    pass


def read_env(path=ENV_FILE):
    out = {}
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def save_env(key, value, path=ENV_FILE):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n{key}={value}\n")


class Render:
    """A very small Render API client. The key travels only in the Authorization header."""

    def __init__(self, key, transport=None, base_url=API):
        self.http = httpx.Client(base_url=base_url, timeout=60, transport=transport,
                                 headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})

    def call(self, method, path, **kw):
        r = self.http.request(method, path, **kw)
        if r.status_code in (401, 403):
            raise DeployError("Render refused the API key (401/403): check RENDER_API_KEY in .env")
        if r.status_code >= 400:
            raise DeployError(f"Render answered {r.status_code} for {method} {path}: {r.text[:300]}")
        return r.json() if r.content else None

    def owners(self):
        return [o["owner"] for o in self.call("GET", "/v1/owners", params={"limit": 20})]

    def services(self):
        out, cursor = [], None
        while True:
            page = self.call("GET", "/v1/services", params={"limit": 100, **({"cursor": cursor} if cursor else {})})
            out += [x["service"] for x in page]
            if len(page) < 100:
                return out
            cursor = page[-1].get("cursor")


def pick_owner(owners, wanted=None):
    if not owners:
        raise DeployError("this API key has no workspace")
    if wanted:
        for o in owners:
            if o.get("name") == wanted:
                return o
        raise DeployError(f"no workspace named {wanted!r}")
    return owners[0]


def service_url(svc):
    return (svc.get("serviceDetails") or {}).get("url") or f"https://{svc['name']}.onrender.com"


def deploy(api, token, dry_run=False, owner_name=None, say=print, wait=True, sleep=time.sleep, http_check=None):
    owner = pick_owner(api.owners(), owner_name)
    by_name = {s["name"]: s for s in api.services() if s.get("ownerId", owner["id"]) == owner["id"]}
    site = by_name.get(SITE)
    if site is None:
        raise DeployError(f'no service named "{SITE}" in workspace "{owner.get("name")}": is this the right workspace? (set RENDER_OWNER)')
    region = (site.get("serviceDetails") or {}).get("region") or "oregon"
    say(f'workspace "{owner.get("name")}", site "{SITE}" found (region {region})')

    knowledge = by_name.get(KNOWLEDGE)
    if knowledge is None:
        say(f'"{KNOWLEDGE}" does not exist: creating it on the free plan')
        body = {"type": "web_service", "name": KNOWLEDGE, "ownerId": owner["id"], "repo": site.get("repo"), "branch": site.get("branch") or "main",
                "autoDeployTrigger": "commit", "rootDir": "", "envVars": [{"key": "KNOWLEDGE_TOKEN", "value": token}],
                "serviceDetails": {"runtime": "docker", "plan": "free", "region": region, "healthCheckPath": "/healthz", "numInstances": 1,
                                   "envSpecificDetails": {"dockerfilePath": "./knowledge_service/Dockerfile", "dockerContext": "."}}}
        if dry_run:
            say("(dry run) would create the service and set the site's KNOWLEDGE_URL / KNOWLEDGE_TOKEN")
            return None
        created = api.call("POST", "/v1/services", json=body)
        knowledge = created.get("service", created)
    else:
        say(f'"{KNOWLEDGE}" already exists: re-setting its token')
        if dry_run:
            say("(dry run) would set KNOWLEDGE_TOKEN on both services")
            return None
        api.call("PUT", f"/v1/services/{knowledge['id']}/env-vars/KNOWLEDGE_TOKEN", json={"value": token})
    url = service_url(knowledge)
    say(f"knowledge service address: {url}")
    api.call("PUT", f"/v1/services/{site['id']}/env-vars/KNOWLEDGE_URL", json={"value": url})
    api.call("PUT", f"/v1/services/{site['id']}/env-vars/KNOWLEDGE_TOKEN", json={"value": token})
    say("the site now has KNOWLEDGE_URL and KNOWLEDGE_TOKEN (Render redeploys it)")
    if wait:
        wait_until_up(url, token, say=say, sleep=sleep, http_check=http_check)
    return url


def wait_until_up(url, token, say=print, sleep=time.sleep, http_check=None, minutes=30):
    """Poll /healthz (the first build takes 10-15 minutes), then /readyz with the token."""
    check = http_check or (lambda path, auth=None: httpx.get(url + path, timeout=40, headers={"Authorization": f"Bearer {token}"} if auth else {}))
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            if check("/healthz").status_code == 200:
                ready = check("/readyz", auth=True)
                say(f"the service is up; stores reachable: {ready.json() if ready.status_code == 200 else ready.status_code}")
                return
        except httpx.HTTPError:
            pass
        say("waiting for the build / start ...")
        sleep(30)
    raise DeployError(f"{url} did not come up within {minutes} minutes: look at the service's logs in the Render dashboard")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args(argv)
    env = {**read_env(), **{k: v for k, v in os.environ.items() if k in ("RENDER_API_KEY", "RENDER_OWNER", "KNOWLEDGE_TOKEN")}}
    key = env.get("RENDER_API_KEY", "")
    if not key:
        print("No RENDER_API_KEY. Create one (Render dashboard > Account Settings > API Keys) and add this line to your .env:\n  RENDER_API_KEY=rnd_...\n"
              "Do not paste it into a chat.", file=sys.stderr)
        return 2
    token = env.get("KNOWLEDGE_TOKEN", "")
    if len(token) < 24:
        token = secrets.token_urlsafe(32)
        if not a.dry_run:
            save_env("KNOWLEDGE_TOKEN", token)
            print("generated a KNOWLEDGE_TOKEN and saved it in .env (not shown)")
    try:
        deploy(Render(key), token, dry_run=a.dry_run, owner_name=env.get("RENDER_OWNER"), wait=not a.no_wait and not a.dry_run)
    except DeployError as exc:
        print("STOPPED:", str(exc).replace(key, "***").replace(token, "***"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
