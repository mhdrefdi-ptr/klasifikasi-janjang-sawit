"""Offline checks; no training or TensorFlow required."""
import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads((ROOT / 'Klasifikasi_TBS_MobileNetV2.ipynb').read_text())
        self.cells = [''.join(c['source']) for c in self.notebook['cells'] if c['cell_type'] == 'code']

    def test_python_syntax(self):
        for cell in self.cells:
            ast.parse('\n'.join(line for line in cell.splitlines() if not line.startswith('%')))
        ast.parse((ROOT / 'app.py').read_text())
        ast.parse((ROOT / 'train.py').read_text())

    def test_local_training_matches_notebook(self):
        expected = ['from IPython.display import display']
        for cell in self.cells:
            if not cell.startswith(('%', 'def predict_image')):
                expected.append(cell.replace('plt.show()', "plt.close('all')"))
        script = ast.parse((ROOT / 'train.py').read_text())
        self.assertEqual(ast.dump(ast.Module(body=script.body[1:], type_ignores=[])),
                         ast.dump(ast.parse('\n\n'.join(expected))))
        for obsolete in ['google.colab', '/content/', 'gradio']:
            self.assertNotIn(obsolete, '\n'.join(self.cells))

    def test_streamlit_entrypoint(self):
        source = (ROOT / 'app.py').read_text()
        self.assertIn('@st.cache_resource', source)
        self.assertIn('st.file_uploader', source)
        self.assertIn('st.stop()', source)
        self.assertIn("ROOT / 'hasil_tbs'", source)
        self.assertNotIn('gradio', source)

    def test_same_gradcam_and_inference_in_notebook_and_app(self):
        app = ast.parse((ROOT / 'app.py').read_text())
        for name in ['gradcam', 'predict_image']:
            expected = next(n for n in app.body if isinstance(n, ast.FunctionDef) and n.name == name)
            notebook_functions = [n for c in self.cells if not c.startswith('%') for n in ast.parse(c).body
                                  if isinstance(n, ast.FunctionDef) and n.name == name]
            self.assertEqual(len(notebook_functions), 1)
            self.assertEqual(ast.dump(expected), ast.dump(notebook_functions[0]))
        self.assertIn('tape.watch(x)', (ROOT / 'app.py').read_text())

    def test_validation_selection_and_split_guards(self):
        source = '\n'.join(self.cells)
        self.assertIn('stratify=df.label', source)
        self.assertIn("drop_duplicates('hash')", source)
        self.assertIn('assert not set(a.hash)&set(b.hash)', source)
        self.assertIn('finetuned_val_loss < frozen_val_loss', source)
        self.assertIn("'validation_target_met':bool(candidates)", source)
        self.assertEqual(self.notebook['nbformat'], 4)

    def test_local_dataset_audit(self):
        path = ROOT / 'data/audit_local.json'
        if not path.exists():
            self.skipTest('Local dataset not downloaded')
        rows = json.loads(path.read_text())
        groups = {}
        for row in rows:
            groups.setdefault(row['hash'], set()).add(row['label'])
        self.assertEqual(len(rows), 857)
        self.assertEqual(len(groups), 851)
        self.assertTrue(all(len(labels) == 1 for labels in groups.values()))


if __name__ == '__main__':
    unittest.main()
