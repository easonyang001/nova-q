"""Regenerate the skill-guided evidence plot from actual saved run records."""
from pathlib import Path
import runpy
import shutil
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'figures' / 'skill'
runpy.run_path(str(SOURCE / 'build_evidence.py'), run_name='__main__')
for ext in ['pdf', 'svg', 'png']:
    shutil.copy2(SOURCE / f'evidence.{ext}', HERE / f'evidence.{ext}')
