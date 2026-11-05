"""Ensure every selectable presentation setting matches the committed audit."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PresentationTests(unittest.TestCase):
    def test_all_selection_masks_match_reported_metrics(self):
        html = (ROOT / 'demo/advanced.html').read_text()
        payload, _ = json.JSONDecoder().raw_decode(html.split('const DATA=', 1)[1])
        policies = {'risk': 'risk_only', 'greedy': 'greedy_floor',
                    'optimized': 'optimized_floor', 'calibrated': 'calibrated_floor'}
        rows = payload['rows']
        for target, modes in payload['selection_grid'].items():
            for mode, mask in modes.items():
                with self.subTest(target=target, policy=mode):
                    self.assertEqual(len(mask), len(rows))
                    self.assertLessEqual(set(mask), {'0', '1'})
                    selected = [r for r, bit in zip(rows, mask) if bit == '1']
                    expected = payload['report']['policy_grid'][target][policies[mode]]
                    self.assertEqual(len(selected), expected['accepted'])
                    mae = sum(abs(r[3] - r[4]) for r in selected) / len(selected)
                    # Embedded inspector values are rounded to 0.001 kg.
                    self.assertAlmostEqual(mae, expected['accepted_mae_kg'], delta=.001)
                    fuel = sum(r[4] for r in selected) / sum(r[4] for r in rows)
                    self.assertAlmostEqual(fuel, expected['accepted_reference_fuel_fraction'], delta=1e-6)
