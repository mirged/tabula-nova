"""Run the real suite and preserve machine-readable evidence, including skips."""

from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

root = Path(__file__).resolve().parents[1]
suite = unittest.defaultTestLoader.discover(str(root/"tests"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {
    "checked_utc": datetime.now(timezone.utc).isoformat(), "tests": result.testsRun,
    "passed": result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
    "failures": len(result.failures), "errors": len(result.errors),
    "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
    "successful": result.wasSuccessful(),
}
(root/"reports").mkdir(exist_ok=True)
(root/"reports/native-tests.json").write_text(json.dumps(report, indent=2)+"\n")
raise SystemExit(0 if result.wasSuccessful() else 1)
