"""Run the audited seed-2 recovery sequentially, then write the final aggregate."""

from __future__ import annotations

from piper_rl.scripts import run_multiseed_campaign as campaign


def main() -> int:
    # --all skips verified completed members and never runs two training jobs together.
    status = campaign.main(["--all"])
    if status:
        return status
    return campaign.main(["--aggregate"])


if __name__ == "__main__":
    raise SystemExit(main())
