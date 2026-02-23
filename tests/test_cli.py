"""Tests for the CLI interface."""

from __future__ import annotations

from click.testing import CliRunner

import pytest

from netops_7b.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "NetOps-7B" in result.output
        assert "data" in result.output
        assert "train" in result.output
        assert "eval" in result.output
        assert "demo" in result.output

    def test_data_help(self, runner):
        result = runner.invoke(cli, ["data", "--help"])
        assert result.exit_code == 0
        assert "curate" in result.output
        assert "convert" in result.output

    def test_train_help(self, runner):
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "export" in result.output

    def test_eval_help(self, runner):
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "benchmark" in result.output
        assert "test-set" in result.output

    def test_data_curate(self, runner, tmp_path):
        output = str(tmp_path / "train.jsonl")
        result = runner.invoke(cli, ["data", "curate", "-o", output, "-n", "50"])
        assert result.exit_code == 0
        assert "Curating dataset" in result.output

    def test_eval_test_set(self, runner, tmp_path):
        output = str(tmp_path / "test.jsonl")
        result = runner.invoke(cli, ["eval", "test-set", "-o", output, "-n", "20"])
        assert result.exit_code == 0
        assert "Generating test set" in result.output
