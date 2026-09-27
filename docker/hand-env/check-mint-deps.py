"""Permit only decord's known wheel-tag defect, not dependency conflicts."""
import subprocess
import sys

result = subprocess.run([sys.executable, "-m", "pip", "check"],
                        capture_output=True, text=True)
print(result.stdout, end="")
print(result.stderr, end="", file=sys.stderr)
if result.returncode:
    if (result.returncode != 1 or result.stderr.strip()
            or result.stdout.strip() != "decord 0.6.0 is not supported on this platform"):
        raise SystemExit(result.returncode)
    import decord
    assert decord.__version__ == "0.6.0"
    print("Accepted decord's stale cp36 internal tag; doctor must verify decoding.")
