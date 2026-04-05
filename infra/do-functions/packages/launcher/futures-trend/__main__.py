import json
import os
import time

import requests

DO_API = "https://api.digitalocean.com/v2"
DO_TAG = "ephemeral-daily-run"
DO_APP_TAG = "app-runners"
JOB_IMAGE = "ghcr.io/mosesmc52/futures-trend:latest"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def optional_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def parse_ssh_key_ref(value: str):
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        return value


def build_cloud_init(env: dict) -> str:
    return f"""#cloud-config
package_update: true
packages:
  - ca-certificates
  - curl

write_files:
  - path: /opt/job/ghcr.env
    permissions: "0600"
    content: |
      GHCR_USERNAME={env.get("GHCR_USERNAME", "")}
      GHCR_TOKEN={env.get("GHCR_TOKEN", "")}
      DO_TOKEN={env.get("DO_TOKEN", "")}
      DO_API={DO_API}

  - path: /opt/job/env
    permissions: "0600"
    content: |
      TZ={env.get("TZ", "")}
      EMAIL_POSITIONS={env.get("EMAIL_POSITIONS", "")}
      TO_ADDRESSES={env.get("TO_ADDRESSES", "")}
      FROM_ADDRESS={env.get("FROM_ADDRESS", "")}
      AWS_SES_REGION_NAME={env.get("AWS_SES_REGION_NAME", "")}
      AWS_SES_ACCESS_KEY_ID={env.get("AWS_SES_ACCESS_KEY_ID", "")}
      AWS_SES_SECRET_ACCESS_KEY={env.get("AWS_SES_SECRET_ACCESS_KEY", "")}
      SPACES_KEY={env.get("SPACES_KEY", "")}
      SPACES_SECRET={env.get("SPACES_SECRET", "")}
      SPACES_BUCKET={env.get("SPACES_BUCKET", "")}
      SPACES_REGION={env.get("SPACES_REGION", "")}
      SPACES_ENDPOINT={env.get("SPACES_ENDPOINT", "")}

  - path: /opt/job/run.sh
    permissions: "0700"
    content: |
      #!/usr/bin/env bash
      set -euo pipefail

      LOG=/var/log/job.log
      mkdir -p /opt/job
      exec > >(tee -a "$LOG") 2>&1

      echo "=== Job start: $(date -Is) ==="

      echo "---- /opt/job/env (redacted) ----"
      sed -E 's/(KEY|TOKEN|SECRET|PASSWORD)=.*/\\1=REDACTED/g' /opt/job/env || true

      curl -fsSL https://get.docker.com | sh
      systemctl enable --now docker

      if [ -f /opt/job/ghcr.env ]; then
        set -a
        . /opt/job/ghcr.env
        set +a
      fi

      if [ -n "${{GHCR_USERNAME:-}}" ] && [ -n "${{GHCR_TOKEN:-}}" ]; then
        echo "${{GHCR_TOKEN}}" | docker login ghcr.io -u "${{GHCR_USERNAME}}" --password-stdin
      else
        echo "GHCR creds not provided; assuming image is public."
      fi

      docker pull "{JOB_IMAGE}"

      set +e
      docker run --rm --env-file /opt/job/env "{JOB_IMAGE}" /app/run.sh
      EXIT_CODE=$?
      set -e

      echo "$EXIT_CODE" > /opt/job/exit_code
      echo "=== Job end: $(date -Is), exit=$EXIT_CODE ==="

      # Optional: upload log to Spaces if configured
      # Self-delete droplet
      META=http://169.254.169.254/metadata/v1
      DROPLET_ID=$(curl -fsS $META/id || true)

      if [ -n "$DROPLET_ID" ] && [ -n "${{DO_TOKEN:-}}" ]; then
        echo "Deleting droplet $DROPLET_ID"
        curl -fsS -X DELETE \
          -H "Authorization: Bearer $DO_TOKEN" \
          "${{DO_API:-https://api.digitalocean.com/v2}}/droplets/$DROPLET_ID" || true
      fi

      sync
      poweroff || true

runcmd:
  - [ bash, -lc, "/opt/job/run.sh" ]
"""


def main(event, context):
    do_token = require_env("DO_TOKEN")
    do_api = DO_API

    droplet_name = f"job-{int(time.time())}"
    tags = [DO_TAG, DO_APP_TAG]

    body = {
        "name": droplet_name,
        "region": require_env("DO_REGION"),
        "size": require_env("DO_SIZE"),
        "image": require_env("DO_IMAGE"),
        "tags": tags,
        "ssh_keys": [parse_ssh_key_ref(require_env("DO_SSH_KEY_ID"))],
        "user_data": build_cloud_init(
            {
                "DO_TOKEN": do_token,
                "TZ": optional_env("TZ"),
                "GHCR_USERNAME": optional_env("GHCR_USERNAME"),
                "GHCR_TOKEN": optional_env("GHCR_TOKEN"),
                "EMAIL_POSITIONS": optional_env("EMAIL_POSITIONS"),
                "TO_ADDRESSES": optional_env("TO_ADDRESSES"),
                "FROM_ADDRESS": optional_env("FROM_ADDRESS"),
                "AWS_SES_REGION_NAME": optional_env("AWS_SES_REGION_NAME"),
                "AWS_SES_ACCESS_KEY_ID": optional_env("AWS_SES_ACCESS_KEY_ID"),
                "AWS_SES_SECRET_ACCESS_KEY": optional_env("AWS_SES_SECRET_ACCESS_KEY"),
                "SPACES_KEY": optional_env("SPACES_KEY"),
                "SPACES_SECRET": optional_env("SPACES_SECRET"),
                "SPACES_BUCKET": optional_env("SPACES_BUCKET"),
                "SPACES_REGION": optional_env("SPACES_REGION"),
                "SPACES_ENDPOINT": optional_env("SPACES_ENDPOINT"),
            }
        ),
        "monitoring": True,
    }

    resp = requests.post(
        f"{do_api}/droplets",
        headers={
            "Authorization": f"Bearer {do_token}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body),
        timeout=30,
    )
    resp.raise_for_status()

    data = resp.json()
    droplet = data.get("droplet", {})

    return {
        "body": {
            "ok": True,
            "droplet_id": droplet.get("id"),
            "droplet_name": droplet.get("name"),
            "status": droplet.get("status"),
        }
    }
