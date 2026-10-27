import unittest
from collections import Counter

import torch

from src.modeling.input_pipeline import CharacterTokenizer, FIELDS, canonical_query
from src.modeling.prepare_synthetic import build_records, make_examples, representations
from src.modeling.train_synthetic import evaluate_loss, select_diagnostics
from src.modeling.diagnose_learning import select_examples
from src.modeling.training_data import answer_example
from src.modeling.transformer import ModelConfig, RecallTransformer


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer(chr(i) for i in range(128))

    def test_reproducible_ids_and_no_real_overlap(self):
        first = build_records(4, {900000000, 900000002})
        self.assertEqual(first, build_records(4, {900000000, 900000002}))
        ids = [r['Player ID'] for r in first]
        self.assertEqual(len(set(ids)), 4)
        self.assertFalse(set(ids) & {900000000, 900000002})

    def test_balanced_modes_and_answer_sources(self):
        records = build_records(2, set())
        examples = make_examples(records, self.tokenizer)
        self.assertEqual(len(examples), 2 * 10 * 4 * 2)
        counts = Counter((e['mode'], e['structure']) for e in examples)
        self.assertEqual(set(counts.values()), {20})
        lookup = {r['Player ID']:r for r in records}
        for e in examples:
            self.assertEqual(e['answer'], str(lookup[e['player_id']][e['target_field']]))
            if e['mode'] == 'recall':
                self.assertEqual(e['prompt'], canonical_query(e))
            else:
                self.assertTrue(e['prompt'].endswith(canonical_query(e)))

    def test_representations_keep_all_values(self):
        import json
        import csv
        import io
        record = build_records(1, set())[0]
        forms = representations(record)
        self.assertEqual(json.loads(forms['key_value']), record)
        def flatten(d):
            out = {}
            for k,v in d.items():
                out.update(flatten(v) if isinstance(v,dict) else {k:v})
            return out
        self.assertEqual(flatten(json.loads(forms['hier_nested'])),record)
        table = next(csv.DictReader(io.StringIO(forms['tabular'])))
        self.assertEqual(table,{k:str(v) for k,v in record.items()})
        self.assertTrue(all(str(v) in forms['flat'] for v in record.values()))

    def test_diagnostics_cover_fields_and_deduplicate_recall(self):
        rows = make_examples(build_records(2,set()), self.tokenizer)
        selected = select_diagnostics(rows,1)
        self.assertEqual(len(selected),50)  # 4 formats * 10 fields + 10 recall
        self.assertEqual({r['target_field'] for r in selected},set(FIELDS))
        self.assertEqual(len({(r['mode'],r['prompt']) for r in selected}),50)
        tiny = select_examples(rows)
        self.assertEqual(len(tiny),4)
        self.assertEqual(len({r['prompt'] for r in tiny}),4)
        self.assertEqual({r['target_field'] for r in tiny},{'Height (cm)','Transfer Fee'})

    def test_validation_does_not_update_weights_or_rng(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = RecallTransformer(ModelConfig(vocab_size=self.tokenizer.vocab_size,
            d_model=16,n_heads=2,n_layers=1,feedforward_dim=32,context_length=32))
        model.train()
        original = {k:v.clone() for k,v in model.state_dict().items()}
        state = torch.get_rng_state().clone()
        examples = [answer_example('a=', '12',self.tokenizer,32), answer_example('b=', '1',self.tokenizer,32)]
        a = evaluate_loss(model,examples,1,'cpu')
        b = evaluate_loss(model,examples,2,'cpu')
        self.assertAlmostEqual(a,b,places=5)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for k,v in model.state_dict().items():
            self.assertTrue(torch.equal(v, original[k]))


if __name__ == '__main__':
    unittest.main()
