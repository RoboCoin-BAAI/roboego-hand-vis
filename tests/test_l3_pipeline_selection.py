import json
from pathlib import Path
import subprocess
import sys


def test_l3_csv_qc_gate_without_conversion(tmp_path):
    helper = Path(__file__).resolve().parents[1] / 'scripts/old/L3/stage.py'
    state = tmp_path / 'state'
    state.mkdir()
    source = tmp_path / 'batch_0000001'
    sessions = [source / f'L3-001-2026-10-19-18-10-{s}' for s in ('53', '54')]
    csv = tmp_path / 'batch_0000001.csv'
    csv.write_text('path\n' + '\n'.join('ks3://bucket/' + s.name for s in sessions))
    records = [dict(session=str(s.resolve()), passed=i == 0) for i,s in enumerate(sessions)]
    program = ';'.join('print('+repr(json.dumps(r))+')' for r in records) + ';raise SystemExit(1)'
    result = subprocess.run([sys.executable, str(helper), '--stage', 'qc', '--state', str(state),
                             '--source', str(source), '--input', str(csv), '--',
                             sys.executable, '-c', program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((state / 'qc.sessions.json').read_text()) == [str(sessions[0].resolve())]
    assert (state / 'qc.partial').exists()
    assert json.loads((state / 'qc.input.sessions.json').read_text()) == [str(s.resolve()) for s in sessions]
