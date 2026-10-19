"""Run all meaningful physics, geometry, and real Qt interaction checks."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import arcs

if __name__ == '__main__':
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'version': arcs.__version__, 'successful': result.wasSuccessful(),
              'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'skipped': len(result.skipped),
              'elapsed_seconds': time.perf_counter()-started,
              'completed_at_UTC': datetime.now(timezone.utc).isoformat()}
    (ROOT / 'results/unit_tests.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
