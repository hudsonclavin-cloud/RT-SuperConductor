import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class HPCTemplateTests(unittest.TestCase):
    def test_scheduler_templates_are_explicitly_unconfirmed(self):
        lsf = (ROOT / "scripts/hpc/job.lsf.sh.template").read_text()
        slurm = (ROOT / "scripts/hpc/job.slurm.sh.template").read_text()
        self.assertIn("UNCONFIRMED TEMPLATE", lsf)
        self.assertIn("#BSUB", lsf)
        self.assertIn("UNCONFIRMED TEMPLATE", slurm)
        self.assertIn("#SBATCH", slurm)


if __name__ == "__main__":
    unittest.main()
