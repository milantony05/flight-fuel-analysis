"""Cross-language check: browser decision totals must equal the Python audit."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js required for JavaScript integration check")
class DashboardTests(unittest.TestCase):
    def test_live_javascript_matches_committed_python_scores(self):
        html = (ROOT / "data/benchmark-source.html").read_text()
        source = html.split("<script>", 1)[1].split("</script>", 1)[0]
        startup = "['target','floor'].forEach(id=>$(id).addEventListener('change',update));update();"
        self.assertEqual(source.count(startup), 1)
        source = source.replace(startup, "")
        source += """
const out={};for(const [name,floor] of [['risk_only',0],['representation_floor',.5]]){
 const selected=choose(DATA.rows,.75,floor), summary=stats(DATA.rows,selected.selected);
 out[name]={accepted:summary.accepted,mae:summary.mae,fuel:summary.fuel,
  long:stats(DATA.rows,selected.selected,r=>group(r[1])==='15–60 min').coverage,
  b744:stats(DATA.rows,selected.selected,r=>r[0]==='B744').accepted};
}console.log(JSON.stringify(out));
"""
        run = subprocess.run(["node", "-"], input=source, text=True, capture_output=True, check=True)
        actual = json.loads(run.stdout)
        expected = json.loads((ROOT / "reports/eligibility-results.json").read_text())["periods"]["validation"]["policies"]
        for name in actual:
            with self.subTest(policy=name):
                self.assertEqual(actual[name]["accepted"], expected[name]["accepted"])
                self.assertAlmostEqual(actual[name]["mae"], expected[name]["accepted_mae_kg"], places=8)
                self.assertAlmostEqual(actual[name]["fuel"], expected[name]["accepted_reference_fuel_fraction"], places=8)
                self.assertAlmostEqual(actual[name]["long"], expected[name]["groups"]["duration"]["15–60 min"]["coverage"])
                self.assertEqual(actual[name]["b744"], expected[name]["groups"]["aircraft_type"]["B744"]["accepted"])


if __name__ == "__main__":
    unittest.main()
