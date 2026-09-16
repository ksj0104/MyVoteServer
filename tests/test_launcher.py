"""Launcher contracts; no server, network, model, or installer is invoked."""

import io
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import main as launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.bundle = Path("/tmp/myvote-official-bundle")
        self.model_id = "myvote-qwen3.5-4b"
        self.config = {
            "demo": {"bundle_dir": self.bundle},
            "network": {"expected_ip": "192.168.219.103", "port": 50051},
            "lmstudio": {
                "base_url": "http://127.0.0.1:1234",
                "model_id": self.model_id,
                "context_length": 4096,
            },
        }

    def run_launcher(self, arguments, ready=True):
        with (
            patch.object(launcher, "read_config", return_value=self.config),
            patch.object(launcher, "check", return_value=ready) as check,
            patch.object(launcher.os, "chdir") as chdir,
            patch.object(launcher.os, "execve") as execute,
            patch.object(launcher.sys, "executable", "/opt/native-python/bin/python3.13"),
            patch.dict(launcher.os.environ, {"PATH": "/usr/bin:/bin"}, clear=True),
            patch("builtins.print"),
        ):
            result = launcher.main(arguments)
        check.assert_called_once_with(self.config)
        return result, chdir, execute

    def test_failed_readiness_never_changes_directory_or_executes(self):
        result, chdir, execute = self.run_launcher(["start", "--install"], ready=False)
        self.assertEqual(result, 1)
        chdir.assert_not_called()
        execute.assert_not_called()

    def test_normal_start_has_no_implicit_install_download_or_speaker_flags(self):
        _, _, execute = self.run_launcher(["start"])
        execute.assert_called_once()
        self.assertEqual(
            execute.call_args.args[1],
            [
                "/usr/bin/caffeinate", "-i", "/bin/bash",
                str(self.bundle / "start-demo.command"),
                "--llm-model", self.model_id,
                "--host", "192.168.219.103", "--port", "50051",
            ],
        )

    def test_explicit_options_and_configured_model_are_forwarded(self):
        _, _, execute = self.run_launcher(
            ["start", "--install", "--download-asr", "--experimental-speakers"]
        )
        self.assertEqual(
            execute.call_args.args[1][4:],
            [
                "--install", "--download-asr", "--experimental-speakers",
                "--llm-model", self.model_id,
                "--host", "192.168.219.103", "--port", "50051",
            ],
        )

    def test_official_bundle_runs_under_caffeinate_with_native_python_path(self):
        _, chdir, execute = self.run_launcher(["start"])
        chdir.assert_called_once_with(self.bundle)
        executable, command, environment = execute.call_args.args
        self.assertEqual(executable, "/usr/bin/caffeinate")
        self.assertEqual(command[:4], [
            executable, "-i", "/bin/bash", str(self.bundle / "start-demo.command")
        ])
        self.assertEqual(environment["PATH"], "/opt/native-python/bin:/usr/bin:/bin")

    def configure_speaker_update(self, *, enabled=True, overlap_enabled=True):
        self.config["speaker_update"] = {
            "enabled": enabled,
            "bundle_dir": Path("/tmp/myvote-update"),
            "overlap_enabled": overlap_enabled,
            "overlap_model": Path("/tmp/pytorch_model.bin"),
            "overlap_threads": 1,
        }

    def test_speaker_update_reuses_demo_environment_and_forwards_overlap_model(self):
        self.configure_speaker_update()
        _, chdir, execute = self.run_launcher(["start"])
        chdir.assert_called_once_with(self.bundle)
        execute.assert_called_once()
        executable, command, environment = execute.call_args.args
        self.assertEqual(executable, "/usr/bin/caffeinate")
        self.assertEqual(command, [
            executable, "-i", "/bin/bash",
            "/tmp/myvote-update/start-update.command", str(self.bundle),
            "--llm-model", self.model_id,
            "--host", "192.168.219.103", "--port", "50051",
            "--experimental-overlap-model", "/tmp/pytorch_model.bin",
            "--overlap-threads", "1",
        ])
        self.assertEqual(environment["PATH"], "/opt/native-python/bin:/usr/bin:/bin")

    def test_speaker_update_preserves_explicit_speaker_option(self):
        self.configure_speaker_update()
        _, _, execute = self.run_launcher(["start", "--experimental-speakers"])
        self.assertEqual(execute.call_args.args[1][4:8], [
            str(self.bundle), "--experimental-speakers", "--llm-model", self.model_id,
        ])

    def test_opt_in_json_compat_uses_project_gateway_and_preserves_options(self):
        self.configure_speaker_update()
        self.config["lmstudio"]["context_review_json_schema"] = True
        _, chdir, execute = self.run_launcher(["start"])
        chdir.assert_called_once_with(self.bundle)
        self.assertEqual(execute.call_args.args[1], [
            "/usr/bin/caffeinate", "-i", str(self.bundle / ".venv/bin/python"), "-I",
            str(launcher.ROOT / "scripts/start_update_with_compat.py"),
            "--update-dir", "/tmp/myvote-update", "--demo-dir", str(self.bundle),
            "--llm-model", self.model_id, "--host", "192.168.219.103", "--port", "50051",
            "--experimental-overlap-model", "/tmp/pytorch_model.bin", "--overlap-threads", "1",
        ])

    def test_json_compat_flag_requires_boolean_and_enabled_update(self):
        for value in ('"true"', 'true'):
            with self.subTest(value=value):
                content = f'''[demo]
bundle_dir = "/tmp/myvote-official-bundle"
[network]
expected_ip = "192.168.219.103"
port = 50051
[lmstudio]
base_url = "http://127.0.0.1:1234"
model_id = "google/gemma-4-26b-a4b"
context_length = 262144
context_review_json_schema = {value}
'''.encode()
                with patch.object(Path, "open", return_value=io.BytesIO(content)):
                    with self.assertRaises(ValueError):
                        launcher.read_config(Path("/tmp/server.toml"))

    def test_speaker_update_can_disable_overlap_without_reverting_update(self):
        self.configure_speaker_update(overlap_enabled=False)
        _, _, execute = self.run_launcher(["start"])
        self.assertEqual(execute.call_args.args[1], [
            "/usr/bin/caffeinate", "-i", "/bin/bash",
            "/tmp/myvote-update/start-update.command", str(self.bundle),
            "--llm-model", self.model_id,
            "--host", "192.168.219.103", "--port", "50051",
        ])

    def test_disabled_speaker_update_uses_original_demo_launcher(self):
        self.configure_speaker_update(enabled=False)
        _, _, execute = self.run_launcher(["start"])
        self.assertEqual(execute.call_args.args[1], [
            "/usr/bin/caffeinate", "-i", "/bin/bash",
            str(self.bundle / "start-demo.command"),
            "--llm-model", self.model_id,
            "--host", "192.168.219.103", "--port", "50051",
        ])

    def test_speaker_update_rejects_install_and_download_before_readiness_or_execution(self):
        self.configure_speaker_update()
        for option in ("--install", "--download-asr"):
            with (
                self.subTest(option=option),
                patch.object(launcher, "read_config", return_value=self.config),
                patch.object(launcher, "check") as check,
                patch.object(launcher.os, "chdir") as chdir,
                patch.object(launcher.os, "execve") as execute,
                patch.object(launcher.sys, "stderr", new_callable=io.StringIO),
            ):
                with self.assertRaises(SystemExit) as raised:
                    launcher.main(["start", option])
                self.assertEqual(raised.exception.code, 2)
                check.assert_not_called()
                chdir.assert_not_called()
                execute.assert_not_called()

    def read_config_with_context(self, context_literal):
        content = f'''[demo]
bundle_dir = "/tmp/myvote-official-bundle"
[network]
expected_ip = "192.168.219.103"
port = 50051
[lmstudio]
base_url = "http://127.0.0.1:1234"
model_id = "google/gemma-4-26b-a4b"
context_length = {context_literal}
'''.encode()
        with patch.object(Path, "open", return_value=io.BytesIO(content)):
            return launcher.read_config(Path("/tmp/server.toml"))

    def test_context_length_accepts_32768(self):
        config = self.read_config_with_context("32768")
        self.assertEqual(config["lmstudio"]["context_length"], 32768)

    def test_context_length_rejects_non_positive_and_non_integer_values(self):
        for literal in ("0", "-1", "true", "false", '"32768"', "32768.0"):
            with self.subTest(context_length=literal):
                with self.assertRaisesRegex(ValueError, "context_length"):
                    self.read_config_with_context(literal)

    def test_gemma_model_id_is_forwarded_exactly_to_both_launchers(self):
        self.config = self.read_config_with_context("32768")
        gemma_id = "google/gemma-4-26b-a4b"
        for update_enabled in (False, True):
            with self.subTest(speaker_update=update_enabled):
                self.configure_speaker_update(enabled=update_enabled)
                _, _, execute = self.run_launcher(["start"])
                command = execute.call_args.args[1]
                self.assertEqual(command.count("--llm-model"), 1)
                self.assertEqual(command[command.index("--llm-model") + 1], gemma_id)

    def test_remote_lmstudio_url_is_rejected(self):
        content = b'''[demo]
bundle_dir = "/tmp/myvote-official-bundle"
[network]
expected_ip = "192.168.219.103"
port = 50051
[lmstudio]
base_url = "http://192.168.219.103:1234"
model_id = "myvote-qwen3.5-4b"
context_length = 4096
'''
        with patch.object(Path, "open", return_value=io.BytesIO(content)):
            with self.assertRaisesRegex(ValueError, "http://127.0.0.1:1234"):
                launcher.read_config(Path("/tmp/server.toml"))

    def configure_dedicated_translation(self):
        self.configure_speaker_update(overlap_enabled=False)
        self.config["lmstudio"].update({
            "translation_profile": "translategemma",
            "model_id": "translategemma-12b-it",
            "context_length": 2048,
            "orchestrator_model_id": "google/gemma-4-26b-a4b",
            "orchestrator_context_length": 262144,
            "translation_workers": 4,
            "context_review_json_schema": True,
        })

    def read_config_values(self, path=Path("/tmp/server.toml")):
        with (
            patch.object(Path, "open", return_value=io.BytesIO()),
            patch.object(launcher.tomllib, "load", return_value=deepcopy(self.config)),
        ):
            return launcher.read_config(path)

    def test_generic_profile_preserves_defaults_and_rejects_dedicated_options(self):
        self.assertNotIn("translation_profile", self.read_config_values()["lmstudio"])
        self.config["lmstudio"]["translation_profile"] = "generic"
        baseline = deepcopy(self.config)
        for field, value in (("orchestrator_model_id", "gemma"),
                             ("orchestrator_context_length", 2048),
                             ("translation_workers", 1), ("translation_model_ids", [])):
            with self.subTest(field=field):
                self.config = deepcopy(baseline)
                self.config["lmstudio"][field] = value
                with self.assertRaisesRegex(ValueError, "translategemma"):
                    self.read_config_values()

    def test_unknown_translation_profile_is_rejected(self):
        for profile in ("unknown", True, 1):
            with self.subTest(profile=profile):
                self.config["lmstudio"]["translation_profile"] = profile
                with self.assertRaisesRegex(ValueError, "translation_profile"):
                    self.read_config_values()

    def test_dedicated_translation_requires_enabled_update(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"]["context_review_json_schema"] = False
        self.config["speaker_update"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "speaker_update.enabled"):
            self.read_config_values()

    def test_dedicated_translation_validates_worker_count(self):
        self.configure_dedicated_translation()
        for value in (0, -1, 5, True, "4", 4.0):
            with self.subTest(workers=value):
                self.config["lmstudio"]["translation_workers"] = value
                with self.assertRaisesRegex(ValueError, "translation_workers"):
                    self.read_config_values()

    def test_dedicated_translation_validates_exact_distinct_model_ids(self):
        self.configure_dedicated_translation()
        for identifiers in ([], "translation", ["same", "same"], ["has space"], [True],
                            ["one", "two", "three", "four", "five"]):
            with self.subTest(identifiers=identifiers):
                self.config["lmstudio"]["translation_model_ids"] = identifiers
                with self.assertRaisesRegex(ValueError, "translation_model_ids"):
                    self.read_config_values()
        self.config["lmstudio"]["translation_model_ids"] = ["one", "two"]
        self.config["lmstudio"]["translation_workers"] = 1
        with self.assertRaisesRegex(ValueError, "translation_model_ids"):
            self.read_config_values()

    def test_dedicated_translation_requires_separate_general_orchestrator(self):
        self.configure_dedicated_translation()
        for identifier in (None, "", "has space", "translategemma-12b-it", "other/Translate-Gemma"):
            with self.subTest(identifier=identifier):
                self.config["lmstudio"]["orchestrator_model_id"] = identifier
                with self.assertRaisesRegex(ValueError, "orchestrator_model_id"):
                    self.read_config_values()
        self.config["lmstudio"]["orchestrator_model_id"] = "shared-instance"
        self.config["lmstudio"]["translation_model_ids"] = ["shared-instance"]
        with self.assertRaisesRegex(ValueError, "orchestrator_model_id"):
            self.read_config_values()

    def test_orchestrator_context_requires_positive_integer(self):
        self.configure_dedicated_translation()
        for value in (None, 0, -1, True, "262144", 262144.0):
            with self.subTest(context=value):
                self.config["lmstudio"]["orchestrator_context_length"] = value
                with self.assertRaisesRegex(ValueError, "orchestrator_context_length"):
                    self.read_config_values()

    def test_dedicated_start_defaults_to_four_workers_without_repeated_model_option(self):
        self.configure_dedicated_translation()
        del self.config["lmstudio"]["translation_workers"]
        self.config = self.read_config_values()
        _, _, execute = self.run_launcher(["start"])
        command = execute.call_args.args[1]
        self.assertIn(str(launcher.ROOT / "scripts/start_update_with_compat.py"), command)
        self.assertEqual(command[command.index("--llm-model") + 1], "translategemma-12b-it")
        self.assertEqual(command[command.index("--translation-profile"):], [
            "--translation-profile", "translategemma",
            "--orchestrator-model", "google/gemma-4-26b-a4b", "--translation-workers", "4",
        ])
        self.assertNotIn("--translation-model-id", command)
        self.assertNotIn("--semantic-boundary-prompt", command)

    def test_dedicated_start_forwards_explicit_translation_instances_in_order(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"]["translation_workers"] = 3
        self.config["lmstudio"]["translation_model_ids"] = ["tg/instance-a", "tg/instance-b"]
        self.config = self.read_config_values()
        _, _, execute = self.run_launcher(["start"])
        command = execute.call_args.args[1]
        self.assertEqual(command[command.index("--translation-profile"):], [
            "--translation-profile", "translategemma",
            "--orchestrator-model", "google/gemma-4-26b-a4b", "--translation-workers", "3",
            "--translation-model-id", "tg/instance-a", "--translation-model-id", "tg/instance-b",
        ])

    def model_payloads(self, *instances):
        listing = {"data": [{"id": identifier} for identifier, _, _ in instances]}
        native = {"models": [
            {"key": identifier, "loaded_instances": [{
                "id": identifier,
                "config": {"context_length": context, **({"parallel": parallel} if parallel is not None else {})},
            }]}
            for identifier, context, parallel in instances
        ]}
        return listing, native

    def run_readiness(self, listing, native, *, missing_files=(), prompt_symlink=False,
                      prompt_size=1024, prompt_stat_error=None):
        with (
            patch.object(launcher.platform, "system", return_value="Darwin"),
            patch.object(launcher.platform, "machine", return_value="arm64"),
            patch.object(launcher.platform, "mac_ver", return_value=("26.0", (), "")),
            patch.object(launcher.sys, "version_info", (3, 12, 0)),
            patch.object(launcher.socket, "socket"),
            patch.object(Path, "is_file", autospec=True, side_effect=lambda path: path not in missing_files),
            patch.object(Path, "is_dir", return_value=True),
            patch.object(Path, "is_symlink", return_value=prompt_symlink),
            patch.object(Path, "stat", return_value=SimpleNamespace(st_size=prompt_size),
                         side_effect=prompt_stat_error),
            patch.object(Path, "iterdir", return_value=[Path("asset")]),
            patch.object(Path, "rglob", return_value=[]),
            patch.object(launcher, "build_opener") as opener,
            patch.object(launcher.sys, "stdout", new_callable=io.StringIO) as output,
        ):
            opener.return_value.open.side_effect = [
                io.BytesIO(json.dumps(value).encode()) for value in (listing, native)
            ]
            ready = launcher.check(self.config)
        self.assertEqual(opener.return_value.open.call_count, 2)
        return ready, output.getvalue()

    def test_generic_readiness_does_not_require_runtime_parallel_metadata(self):
        self.assertTrue(self.run_readiness(*self.model_payloads((self.model_id, 4096, None)))[0])

    def test_dedicated_readiness_accepts_loaded_models_with_individual_contexts(self):
        self.configure_dedicated_translation()
        self.assertTrue(self.run_readiness(*self.model_payloads(
            ("translategemma-12b-it", 2048, 4), ("google/gemma-4-26b-a4b", 262144, None)
        ))[0])

    def test_dedicated_readiness_requires_both_api_and_exact_loaded_instance_ids(self):
        self.configure_dedicated_translation()
        for index in (0, 1):
            for missing_from in ("api", "native"):
                with self.subTest(model=index, missing_from=missing_from):
                    listing, native = self.model_payloads(
                        ("translategemma-12b-it", 2048, 4), ("google/gemma-4-26b-a4b", 262144, None))
                    if missing_from == "api":
                        listing["data"].pop(index)
                    else:
                        native["models"][index]["loaded_instances"][0]["id"] += "-other-instance"
                    self.assertFalse(self.run_readiness(listing, native)[0])

    def test_dedicated_readiness_rejects_wrong_context_for_either_role(self):
        self.configure_dedicated_translation()
        for index in (0, 1):
            with self.subTest(model=index):
                listing, native = self.model_payloads(
                    ("translategemma-12b-it", 2048, 4), ("google/gemma-4-26b-a4b", 262144, None))
                native["models"][index]["loaded_instances"][0]["config"]["context_length"] = 4096
                ready, output = self.run_readiness(listing, native)
                self.assertFalse(ready)
                self.assertIn("[BLOCK] 문맥 길이", output)

    def test_dedicated_readiness_blocks_unverified_or_insufficient_parallel_capacity(self):
        self.configure_dedicated_translation()
        for capacity in (None, 1, 3, True, "4"):
            with self.subTest(capacity=capacity):
                ready, output = self.run_readiness(*self.model_payloads(
                    ("translategemma-12b-it", 2048, capacity), ("google/gemma-4-26b-a4b", 262144, None)))
                self.assertFalse(ready)
                self.assertIn("[BLOCK] 번역 런타임 병렬 처리", output)

    def test_dedicated_readiness_accepts_known_parallel_capacity_alias(self):
        self.configure_dedicated_translation()
        listing, native = self.model_payloads(
            ("translategemma-12b-it", 2048, 4), ("google/gemma-4-26b-a4b", 262144, None))
        runtime = native["models"][0]["loaded_instances"][0]["config"]
        runtime["max_parallel_predictions"] = runtime.pop("parallel")
        self.assertTrue(self.run_readiness(listing, native)[0])

    def test_translation_parallel_capacity_follows_round_robin_slot_allocation(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"]["translation_model_ids"] = ["tg/a", "tg/b", "tg/c"]
        listing, native = self.model_payloads(
            ("tg/a", 2048, 2), ("tg/b", 2048, 1), ("tg/c", 2048, 1),
            ("google/gemma-4-26b-a4b", 262144, None))
        listing["data"].append({"id": "translategemma-12b-it"})
        self.assertTrue(self.run_readiness(listing, native)[0])
        native["models"][0]["loaded_instances"][0]["config"]["parallel"] = 1
        self.assertFalse(self.run_readiness(listing, native)[0])

    def test_semantic_boundary_prompt_requires_nonempty_string_path(self):
        self.configure_dedicated_translation()
        for value in (None, 1, True, "", "   "):
            with self.subTest(path=value):
                self.config["lmstudio"]["semantic_boundary_prompt"] = value
                with self.assertRaisesRegex(ValueError, "semantic_boundary_prompt"):
                    self.read_config_values()

    def test_semantic_boundary_prompt_requires_dedicated_gemma4_and_json_compat(self):
        self.config["lmstudio"]["semantic_boundary_prompt"] = "boundary.md"
        with self.assertRaisesRegex(ValueError, "semantic_boundary_prompt"):
            self.read_config_values()
        self.configure_dedicated_translation()
        for field, value in (("orchestrator_model_id", "google/gemma-4-other"),
                             ("context_review_json_schema", False)):
            with self.subTest(field=field):
                baseline = deepcopy(self.config)
                self.config["lmstudio"][field] = value
                with self.assertRaisesRegex(ValueError, "semantic_boundary_prompt"):
                    self.read_config_values()
                self.config = baseline

    def test_semantic_boundary_prompt_path_uses_config_directory_without_resolving(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"]["semantic_boundary_prompt"] = "prompts/../boundary.md"
        config = self.read_config_values(Path("/tmp/config/server.toml"))
        self.assertEqual(config["lmstudio"]["semantic_boundary_prompt"],
                         Path("/tmp/config/prompts/../boundary.md"))
        self.config["lmstudio"]["semantic_boundary_prompt"] = "/tmp/custom prompts/boundary.md"
        config = self.read_config_values(Path("/tmp/config/server.toml"))
        self.assertEqual(config["lmstudio"]["semantic_boundary_prompt"],
                         Path("/tmp/custom prompts/boundary.md"))

    def test_semantic_boundary_prompt_is_forwarded_as_bootstrap_option(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"]["semantic_boundary_prompt"] = "prompts/boundary.md"
        self.config = self.read_config_values()
        bundle = self.config["demo"]["bundle_dir"]
        _, _, execute = self.run_launcher(["start", "--experimental-speakers"])
        command = execute.call_args.args[1]
        self.assertEqual(command[:11], [
            "/usr/bin/caffeinate", "-i", str(bundle / ".venv/bin/python"), "-I",
            str(launcher.ROOT / "scripts/start_update_with_compat.py"),
            "--update-dir", "/tmp/myvote-update", "--demo-dir", str(bundle),
            "--semantic-boundary-prompt", "/tmp/prompts/boundary.md",
        ])
        self.assertEqual(command[11], "--experimental-speakers")
        self.assertLess(command.index("--semantic-boundary-prompt"), command.index("--llm-model"))
        self.assertEqual(command.count("--semantic-boundary-prompt"), 1)

    def test_semantic_boundary_readiness_accepts_regular_file_at_size_limit(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"]["semantic_boundary_prompt"] = Path("/tmp/boundary.md")
        ready, output = self.run_readiness(*self.model_payloads(
            ("translategemma-12b-it", 2048, 4), ("google/gemma-4-26b-a4b", 262144, None)),
            prompt_size=65536)
        self.assertTrue(ready)
        self.assertIn("[OK] 구간 선택 프롬프트", output)

    def test_semantic_boundary_readiness_blocks_missing_invalid_or_unreadable_assets(self):
        self.configure_dedicated_translation()
        prompt = Path("/tmp/boundary.md")
        self.config["lmstudio"]["semantic_boundary_prompt"] = prompt
        cases = (
            {"missing_files": (prompt,)},
            {"missing_files": (launcher.ROOT / "scripts/gemma_boundary_prompt.py",)},
            {"prompt_symlink": True},
            {"prompt_size": 65537},
            {"prompt_stat_error": PermissionError("test denied")},
        )
        for options in cases:
            with self.subTest(options=options):
                ready, output = self.run_readiness(*self.model_payloads(
                    ("translategemma-12b-it", 2048, 4), ("google/gemma-4-26b-a4b", 262144, None)),
                    **options)
                self.assertFalse(ready)
                self.assertIn("[BLOCK] 구간 선택 프롬프트", output)

    def test_semantic_boundary_refinement_requires_boolean(self):
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value):
                self.config["lmstudio"]["semantic_boundary_refinement"] = value
                with self.assertRaisesRegex(ValueError, "semantic_boundary_refinement"):
                    self.read_config_values()

    def test_semantic_boundary_refinement_requires_prompt(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"]["semantic_boundary_refinement"] = True
        with self.assertRaisesRegex(ValueError, "semantic_boundary_prompt"):
            self.read_config_values()

    def test_semantic_boundary_refinement_inherits_prompt_profile_requirements(self):
        self.config["lmstudio"].update({
            "semantic_boundary_refinement": True, "semantic_boundary_prompt": "boundary.md",
        })
        with self.assertRaisesRegex(ValueError, "semantic_boundary_prompt"):
            self.read_config_values()
        self.configure_dedicated_translation()
        for field, value in (("orchestrator_model_id", "google/gemma-4-other"),
                             ("context_review_json_schema", False)):
            with self.subTest(field=field):
                baseline = deepcopy(self.config)
                self.config["lmstudio"][field] = value
                with self.assertRaisesRegex(ValueError, "semantic_boundary_prompt"):
                    self.read_config_values()
                self.config = baseline

    def test_semantic_boundary_refinement_forwards_single_bootstrap_flag(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"].update({
            "semantic_boundary_refinement": True, "semantic_boundary_prompt": "boundary.md",
        })
        self.config = self.read_config_values()
        _, _, execute = self.run_launcher(["start", "--experimental-speakers"])
        command = execute.call_args.args[1]
        self.assertEqual(command[9:13], [
            "--semantic-boundary-prompt", "/tmp/boundary.md",
            "--semantic-boundary-refinement", "--experimental-speakers",
        ])
        self.assertEqual(command.count("--semantic-boundary-refinement"), 1)
        self.assertLess(command.index("--semantic-boundary-refinement"), command.index("--llm-model"))

    def test_disabled_semantic_boundary_refinement_preserves_generic_and_prompt_launchers(self):
        self.config["lmstudio"]["semantic_boundary_refinement"] = False
        self.read_config_values()  # False is valid without a prompt or dedicated profile.
        for with_prompt in (False, True):
            with self.subTest(with_prompt=with_prompt):
                if with_prompt:
                    self.configure_dedicated_translation()
                    self.config["lmstudio"]["semantic_boundary_prompt"] = Path("/tmp/boundary.md")
                self.config["lmstudio"].pop("semantic_boundary_refinement", None)
                _, _, baseline = self.run_launcher(["start"])
                self.config["lmstudio"]["semantic_boundary_refinement"] = False
                _, _, disabled = self.run_launcher(["start"])
                self.assertEqual(baseline.call_args.args, disabled.call_args.args)
                self.assertNotIn("--semantic-boundary-refinement", disabled.call_args.args[1])

    def test_semantic_boundary_refinement_checks_both_runtime_modules_only_when_enabled(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"].update({
            "semantic_boundary_refinement": True, "semantic_boundary_prompt": Path("/tmp/boundary.md"),
        })
        payloads = self.model_payloads(
            ("translategemma-12b-it", 2048, 4), ("google/gemma-4-26b-a4b", 262144, None))
        ready, output = self.run_readiness(*payloads)
        self.assertTrue(ready)
        self.assertIn("[OK] 구간 선택 보완", output)
        for name in ("semantic_boundary_routing.py", "semantic_boundary_pipeline.py"):
            with self.subTest(module=name):
                missing = (launcher.ROOT / "scripts" / name,)
                ready, output = self.run_readiness(*payloads, missing_files=missing)
                self.assertFalse(ready)
                self.assertIn("[BLOCK] 구간 선택 보완", output)
        self.config["lmstudio"]["semantic_boundary_refinement"] = False
        ready, output = self.run_readiness(*payloads, missing_files=missing)
        self.assertTrue(ready)
        self.assertNotIn("구간 선택 보완", output)

    def test_semantic_quality_first_requires_boolean(self):
        for value in (None, 0, 1, "true", "false", [], {}):
            with self.subTest(value=value):
                self.config["lmstudio"]["semantic_quality_first"] = value
                with self.assertRaisesRegex(ValueError, "semantic_quality_first"):
                    self.read_config_values()

    def test_semantic_quality_first_requires_enabled_refinement(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"].update({
            "semantic_quality_first": True, "semantic_boundary_prompt": "boundary.md",
        })
        for refinement in (None, False):
            with self.subTest(refinement=refinement):
                if refinement is None:
                    self.config["lmstudio"].pop("semantic_boundary_refinement", None)
                else:
                    self.config["lmstudio"]["semantic_boundary_refinement"] = refinement
                with self.assertRaisesRegex(ValueError, "semantic_boundary_refinement = true"):
                    self.read_config_values()

    def test_semantic_quality_first_inherits_prompt_requirement(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"].update({
            "semantic_quality_first": True, "semantic_boundary_refinement": True,
        })
        with self.assertRaisesRegex(ValueError, "semantic_boundary_prompt"):
            self.read_config_values()

    def test_semantic_quality_first_inherits_prompt_profile_requirements(self):
        self.config["lmstudio"].update({
            "semantic_quality_first": True, "semantic_boundary_refinement": True,
            "semantic_boundary_prompt": "boundary.md",
        })
        with self.assertRaisesRegex(ValueError, "semantic_boundary_prompt"):
            self.read_config_values()
        self.configure_dedicated_translation()
        for field, value in (("orchestrator_model_id", "google/gemma-4-other"),
                             ("context_review_json_schema", False)):
            with self.subTest(field=field):
                baseline = deepcopy(self.config)
                self.config["lmstudio"][field] = value
                with self.assertRaisesRegex(ValueError, "semantic_boundary_prompt"):
                    self.read_config_values()
                self.config = baseline

    def test_semantic_quality_first_forwards_single_bootstrap_flag(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"].update({
            "semantic_quality_first": True, "semantic_boundary_refinement": True,
            "semantic_boundary_prompt": "boundary.md",
        })
        self.config = self.read_config_values()
        self.assertIs(self.config["lmstudio"]["semantic_quality_first"], True)
        _, _, execute = self.run_launcher(["start", "--experimental-speakers"])
        command = execute.call_args.args[1]
        self.assertEqual(command[9:14], [
            "--semantic-boundary-prompt", "/tmp/boundary.md",
            "--semantic-boundary-refinement", "--semantic-quality-first",
            "--experimental-speakers",
        ])
        self.assertEqual(command.count("--semantic-quality-first"), 1)
        self.assertLess(command.index("--semantic-quality-first"), command.index("--llm-model"))

    def test_disabled_semantic_quality_first_preserves_existing_launchers(self):
        self.config["lmstudio"]["semantic_quality_first"] = False
        self.read_config_values()  # False needs no refinement, prompt, or dedicated profile.
        for with_refinement in (False, True):
            with self.subTest(with_refinement=with_refinement):
                if with_refinement:
                    self.configure_dedicated_translation()
                    self.config["lmstudio"].update({
                        "semantic_boundary_refinement": True,
                        "semantic_boundary_prompt": Path("/tmp/boundary.md"),
                    })
                self.config["lmstudio"].pop("semantic_quality_first", None)
                _, _, baseline = self.run_launcher(["start"])
                self.config["lmstudio"]["semantic_quality_first"] = False
                _, _, disabled = self.run_launcher(["start"])
                self.assertEqual(baseline.call_args.args, disabled.call_args.args)
                self.assertNotIn("--semantic-quality-first", disabled.call_args.args[1])

    def test_quality_timing_values_validate_and_reach_compat_launcher(self):
        self.configure_dedicated_translation()
        values = {"semantic_selection_timeout_s": 4.5, "semantic_translation_timeout_s": 12,
                  "semantic_max_hold_s": .75, "semantic_total_age_s": 35,
                  "semantic_min_request_interval_s": .25, "semantic_latency_target_s": 2,
                  "semantic_inactivity_flush_s": 2}
        self.config["lmstudio"].update(semantic_quality_first=True, semantic_boundary_refinement=True,
                                      semantic_boundary_prompt="boundary.md", semantic_low_latency=True, **values)
        self.config = self.read_config_values()
        _, _, execute = self.run_launcher(["start"])
        command = execute.call_args.args[1]
        self.assertIn("--semantic-low-latency", command)
        for key, value in values.items():
            option = "--" + key.replace("_", "-")
            self.assertEqual(command[command.index(option) + 1], str(value))

    def test_invalid_quality_timing_values_are_rejected(self):
        for key in ("semantic_selection_timeout_s", "semantic_translation_timeout_s",
                    "semantic_max_hold_s", "semantic_total_age_s",
                    "semantic_min_request_interval_s", "semantic_latency_target_s"):
            for value in (True, 0, -1, float("inf"), float("nan"), 121, "5"):
                with self.subTest(key=key, value=value):
                    self.config["lmstudio"][key] = value
                    with self.assertRaises(ValueError):
                        self.read_config_values()
            self.config["lmstudio"].pop(key)
        self.config["lmstudio"].update(semantic_selection_timeout_s=10,
                                      semantic_translation_timeout_s=15, semantic_total_age_s=24)
        with self.assertRaises(ValueError):
            self.read_config_values()

    def test_low_latency_requires_boolean_and_quality_safeguards(self):
        for value in (None, 1, "true", True):
            self.config["lmstudio"]["semantic_low_latency"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read_config_values()

    def test_inactivity_requires_quality_and_a_bounded_finite_timeout(self):
        self.config["lmstudio"]["semantic_inactivity_flush_s"] = 0
        self.read_config_values()
        for value in (True, -1, float("inf"), float("nan"), 45, "2", 2):
            self.config["lmstudio"]["semantic_inactivity_flush_s"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read_config_values()
        self.configure_dedicated_translation()
        self.config["lmstudio"].update(semantic_quality_first=True,
            semantic_boundary_refinement=True, semantic_boundary_prompt="boundary.md",
            semantic_inactivity_flush_s=2)
        self.assertEqual(self.read_config_values()["lmstudio"]["semantic_inactivity_flush_s"], 2)

    def test_semantic_quality_first_checks_policy_module_only_when_enabled(self):
        self.configure_dedicated_translation()
        self.config["lmstudio"].update({
            "semantic_quality_first": True, "semantic_boundary_refinement": True,
            "semantic_boundary_prompt": Path("/tmp/boundary.md"),
        })
        payloads = self.model_payloads(
            ("translategemma-12b-it", 2048, 4), ("google/gemma-4-26b-a4b", 262144, None))
        ready, output = self.run_readiness(*payloads)
        self.assertTrue(ready)
        self.assertIn("[OK] 의미 구간 품질 우선", output)
        missing = (launcher.ROOT / "scripts/semantic_quality_policy.py",)
        ready, output = self.run_readiness(*payloads, missing_files=missing)
        self.assertFalse(ready)
        self.assertIn("[BLOCK] 의미 구간 품질 우선", output)
        for enabled in (None, False):
            with self.subTest(enabled=enabled):
                if enabled is None:
                    self.config["lmstudio"].pop("semantic_quality_first", None)
                else:
                    self.config["lmstudio"]["semantic_quality_first"] = enabled
                ready, output = self.run_readiness(*payloads, missing_files=missing)
                self.assertTrue(ready)
                self.assertNotIn("의미 구간 품질 우선", output)


if __name__ == "__main__":
    unittest.main()
