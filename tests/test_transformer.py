import unittest

import torch

from src.modeling.input_pipeline import CharacterTokenizer
from src.modeling.training_data import answer_example, collate_examples, corpus_example
from src.modeling.transformer import ModelConfig, RecallTransformer


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.tokenizer = CharacterTokenizer("abc=12")
        self.config = ModelConfig(vocab_size=self.tokenizer.vocab_size, context_length=32,
                                  d_model=16, n_heads=2, n_layers=2, feedforward_dim=32, dropout=0)
        self.model = RecallTransformer(self.config).eval()

    def test_future_tokens_do_not_change_prefix_predictions(self):
        first = torch.tensor([[1, 4, 5, 6, 7]])
        second = first.clone()
        second[0, 3:] = torch.tensor([8, 9])
        with torch.no_grad():
            a = self.model(first)["logits"][:, :3]
            b = self.model(second)["logits"][:, :3]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_padding_does_not_change_predictions_or_loss(self):
        sample = corpus_example("ab", self.tokenizer, 32)
        alone = collate_examples([sample])
        padded = {"input_ids": torch.cat([alone['input_ids'], torch.zeros(1, 3, dtype=torch.long)], 1),
                  "labels": torch.cat([alone['labels'], torch.full((1, 3), -100, dtype=torch.long)], 1)}
        a, b = self.model(**alone), self.model(**padded)
        torch.testing.assert_close(a['loss'], b['loss'])
        torch.testing.assert_close(a['logits'], b['logits'][:, :alone['input_ids'].shape[1]], atol=1e-6, rtol=1e-6)

    def test_answer_mask_and_shift(self):
        example = answer_example("a=", "12", self.tokenizer, 32)
        self.assertEqual(example['labels'][:2], [-100, -100])
        self.assertEqual(example['labels'][2:], self.tokenizer.encode("12", eos=True))
        corpus = corpus_example("ab", self.tokenizer, 32)
        self.assertEqual(corpus['input_ids'], self.tokenizer.encode('ab', bos=True))
        self.assertEqual(corpus['labels'], self.tokenizer.encode('ab', eos=True))

    def test_optimizer_step_reduces_toy_loss(self):
        batch = collate_examples([corpus_example("ab=12", self.tokenizer, 32)])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01)
        initial = float(self.model(**batch)['loss'].detach())
        self.model.train()
        for _ in range(30):
            optimizer.zero_grad()
            loss = self.model(**batch)['loss']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
            optimizer.step()
        final = float(self.model(**batch)['loss'].detach())
        self.assertLess(final, initial * .25)

    def test_eos_and_mode_restoration(self):
        self.model.train()
        with torch.no_grad():
            self.model.output.weight.zero_()
        # All remaining logits tie, so lowest allowed ID (EOS=2) wins.
        result = self.model.generate([1, 4], max_new_tokens=5)
        self.assertEqual(result, {'token_ids': [2], 'stop_reason': 'eos'})
        self.assertTrue(self.model.training)

    def test_context_and_padding_validation(self):
        with self.assertRaises(ValueError):
            corpus_example('a' * 40, self.tokenizer, 32)
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[1, 0, 4]], dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model(torch.ones(1, 33, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
