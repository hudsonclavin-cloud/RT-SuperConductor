import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class HPCTemplateTests(unittest.TestCase):
    def test_scheduler_templates_are_explicitly_unconfirmed(self):
        lsf = (ROOT / "scripts/hpc/job.lsf.sh.template").read_text()
        slurm = (ROOT / "scripts/hpc/job.slurm.sh.template").read_text()
        self.assertIn("UNCONFIRMED TEMPLATE", lsf)
        self.assertIn("#BSUB", lsf)
        self.assertIn("UNCONFIRMED RESOURCES", slurm)
        self.assertIn("#SBATCH", slurm)

    def test_slurm_template_follows_hazel_rules(self):
        slurm = (ROOT / "scripts/hpc/job.slurm.sh.template").read_text()
        lsf = (ROOT / "scripts/hpc/job.lsf.sh.template").read_text()
        self.assertIn("Hazel uses Slurm", slurm)
        self.assertIn("#SBATCH --partition=", slurm)
        self.assertIn("#SBATCH --qos=", slurm)
        self.assertIn("no internet", slurm)
        self.assertIn("OBSOLETE FOR HAZEL", lsf)


if __name__ == "__main__":
    unittest.main()
