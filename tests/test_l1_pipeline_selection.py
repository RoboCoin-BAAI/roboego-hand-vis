"""Exercise isolation using real helper subprocesses and synthetic session results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HELPER = Path(__file__).resolve().parents[1] / 'scripts/old/L1/stage.py'


class SelectionTests(unittest.TestCase):
    def test_partial_conversion_qc_rectification_and_empty_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, output, state = [root / n for n in ('raw', 'output', 'state')]
            state.mkdir()
            sessions = [output / n for n in ('good', 'bad_qc', 'bad_rectify', 'bad_convert')]
            for session in sessions[:3]:
                meta = session / 'meta'
                meta.mkdir(parents=True)
                (meta / 'info.json').write_text('{}')
                (meta / 'episodes.jsonl').write_text('{}\n')
                (meta / 'l1_source.json').write_text(json.dumps({'source': str((raw / session.name).resolve())}))
            csv = root / 'batch.csv'
            csv.write_text('path\n' + '\n'.join('ks3://bucket/' + s.name for s in sessions))

            def run(stage, source_file, records, code=1):
                program = 'import sys; ' + '; '.join('print(' + repr(json.dumps(r)) + ')' for r in records) + f'; sys.exit({code})'
                result = subprocess.run([sys.executable, str(HELPER), '--stage', stage,
                    '--source', str(output), '--state', str(state), '--input', str(source_file),
                    '--raw-source', str(raw), '--', sys.executable, '-c', program], capture_output=True, text=True)
                return result

            r = run('convert', csv, [dict(done=3, failed=1, skipped_existing=0)])
            self.assertEqual(r.returncode, 0, r.stderr)
            selected = json.loads((state / 'convert.sessions.json').read_text())
            self.assertEqual(selected, [str(s.resolve()) for s in sessions[:3]])
            records = [dict(session=str(s), passed=s.name != 'bad_qc') for s in sessions[:3]]
            self.assertEqual(run('qc', state / 'convert.sessions.json', records).returncode, 0)
            selected = json.loads((state / 'qc.sessions.json').read_text())
            self.assertEqual(selected, [str(s.resolve()) for s in (sessions[0], sessions[2])])
            records = [dict(session=str(s), status='done' if s.name == 'good' else 'failed') for s in (sessions[0], sessions[2])]
            self.assertEqual(run('rectify', state / 'qc.sessions.json', records).returncode, 0)
            self.assertEqual(json.loads((state / 'rectify.sessions.json').read_text()), [str(sessions[0].resolve())])
            self.assertTrue((state / 'convert.partial').exists())
            self.assertTrue((state / 'qc.partial').exists())
            # An entry that dies before reporting all selected sessions is fatal.
            self.assertNotEqual(run('qc', state / 'convert.sessions.json', []).returncode, 0)
            self.assertNotEqual(run('convert', csv, []).returncode, 0)
            # Empty input skips the command entirely, even one that would fail.
            empty = state / 'empty.json'
            empty.write_text('[]')
            self.assertEqual(run('depth', empty, [], code=99).returncode, 0)


if __name__ == '__main__':
    unittest.main()
