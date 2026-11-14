import unittest
from frozen_pilot import validate_policy, execute


class FrozenPilotPolicyTests(unittest.TestCase):
    def config(self):
        return dict(mode='learning', init_mode='warm_start', optimizer_init='new',
                    scheduler_init='new', collect_learning_probe=False,
                    run_public=False, max_steps=18, gradient_accumulation=4,
                    max_epochs_equivalent=4, max_seconds=600,
                    child_timeout_seconds=900)

    def test_CPU_preparation_needs_no_GPU(self):
        validate_policy(self.config())

    def test_GPU_denied_before_any_trainer(self):
        with self.assertRaises(ValueError):
            execute(self.config(), lambda *args: self.fail('Optimizer reached'))

    def test_no_resume_or_fresh_fallback(self):
        for mode in ('fresh', 'resume'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_policy({**self.config(), 'init_mode': mode})

    def test_no_hidden_recollection_or_public(self):
        for key in ('collect_learning_probe', 'run_public'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_policy({**self.config(), key: True})

    def test_steps_cannot_revert_to_old_default(self):
        with self.assertRaises(ValueError):
            validate_policy({**self.config(), 'max_steps': 9})

    def test_no_unbounded_timeout(self):
        for value in (float('inf'), float('nan'), 0, True, 901):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_policy({**self.config(), 'child_timeout_seconds': value})


if __name__ == '__main__':
    unittest.main()
