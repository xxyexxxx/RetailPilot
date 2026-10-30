"""Print server-only password-hash configuration; never store a plaintext key."""

import argparse
import getpass
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RetailPilot.auth import password_hash

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    args = parser.parse_args()
    password = getpass.getpass("Web password (12+ characters): ")
    if len(password) < 12 or password != getpass.getpass("Confirm password: "):
        parser.error("passwords must match and contain at least 12 characters")
    print(json.dumps({args.user: password_hash(password)}, ensure_ascii=False))
