import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
AGENTS = ROOT / ".github" / "agents"
COORDINATOR = AGENTS / "ckh-kernel-harness.agent.md"
LEGACY_AGENTS = ROOT / ".claude" / "agents"
WORKFLOW_YAML = ROOT / "workflows" / "performance-optimization.workflow.yaml"
POLICY_YAML = ROOT / "workflows" / "performance-optimization.policy.yaml"


def _frontmatter_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    assert match, f"missing {key!r} frontmatter"
    return match.group(1).strip()


def test_coordinator_delegates_are_model_routed():
    coordinator = COORDINATOR.read_text(encoding="utf-8")
    raw_agents = _frontmatter_value(coordinator, "agents")
    assert raw_agents.startswith("[") and raw_agents.endswith("]")

    delegates = [name.strip() for name in raw_agents[1:-1].split(",")]
    assert delegates
    for delegate in delegates:
        definition = AGENTS / f"{delegate}.agent.md"
        assert definition.is_file(), f"missing delegate definition: {definition}"
        text = definition.read_text(encoding="utf-8")
        assert _frontmatter_value(text, "name") == delegate
        assert _frontmatter_value(text, "model").strip('"\'')
        assert "## Stop Conditions" in text


def test_pipeline_workflow_is_unattended_and_doctor_first():
    coordinator = COORDINATOR.read_text(encoding="utf-8")
    doctor = coordinator.index("1. `ckh-doctor`")
    pre_profile = coordinator.index("2. `ckh-pre-profiler`")
    profile_run = coordinator.index("3. `ckh-profile-runner`")
    profile_report = coordinator.index("4. `ckh-profile-reporter`")

    assert doctor < pre_profile < profile_run < profile_report
    assert "Do not stop after announcing a plan" in coordinator
    assert "Never allow a CLI stdin prompt" in coordinator
    assert "performance-optimization.workflow.yaml" in coordinator
    assert "performance-optimization.policy.yaml" in coordinator
    assert "docs/OVERVIEW.md" not in coordinator
    assert "docs/WORKFLOW.md" not in coordinator
    assert "docs/ARCHITECTURE.md" not in coordinator
    assert "ask for one unless they already know the target" in coordinator
    assert "kernel. If they already know the target kernel" in coordinator
    assert "write that argv back into `[profile].pipeline`" in coordinator
    assert "load the required `ckh/*` MCP surface with\n   `tool_search`" in coordinator
    assert "CKH MCP tools are unavailable in this session" in coordinator
    assert "python deploy/setup_local_mcp.py" in coordinator
    assert "Do not continue\n   to subagents" in coordinator
    assert "exactly one workflow step at a time" in coordinator
    assert "Never dispatch doctor with preflight" in coordinator
    assert "`ckh-profile-runner`'s `dump_dir`, `runs`, and app metrics" in coordinator
    assert "must not search for or regenerate" in coordinator
    assert "an older profile" in coordinator


def test_user_selects_hot_kernel_before_kernel_generation():
    coordinator = COORDINATOR.read_text(encoding="utf-8")
    profile_report = coordinator.index("4. `ckh-profile-reporter`")
    selection = coordinator.index("5. Kernel selection checkpoint")
    onboarder = coordinator.index("6. Delegate `ckh-kernel-onboarder`")
    assert profile_report < selection < onboarder

    assert "ranked top-k kernels" in coordinator
    assert "vscode_askQuestions` to choose exactly one kernel" in coordinator
    assert "never auto-select" in coordinator
    assert "dumped OpenVINO source directory" in coordinator

    reporter = (AGENTS / "ckh-profile-reporter.agent.md").read_text(encoding="utf-8")
    assert "ranked top-k kernel list" in reporter
    assert "dumped OpenVINO kernel source directory" in reporter

    onboarder_text = (AGENTS / "ckh-kernel-onboarder.agent.md").read_text(encoding="utf-8")
    assert "ckh.kernel_prepare" in onboarder_text
    assert "dump_sources" in onboarder_text

    policy = POLICY_YAML.read_text(encoding="utf-8")
    assert "  kernel_selection:" in policy
    assert "Ask the user to choose exactly one kernel; never auto-select when the choice is offered." in policy

    workflow = WORKFLOW_YAML.read_text(encoding="utf-8")
    assert "- gate: kernel_selection" in workflow
    assert "interaction: user_choice" in workflow
    assert '"kernel_selection.output.selected_kernel"' in workflow


def test_kernel_generation_is_never_bypassed():
    coordinator = COORDINATOR.read_text(encoding="utf-8")
    assert "Always generate the kernel for the selected target" in coordinator
    assert "never bypass generation" in coordinator

    onboarder_text = (AGENTS / "ckh-kernel-onboarder.agent.md").read_text(encoding="utf-8")
    assert "Always generate the port for the selected kernel" in onboarder_text
    assert "Stop if an existing sandbox spec already covers" not in onboarder_text

    policy = POLICY_YAML.read_text(encoding="utf-8")
    assert "ledger entries and existing sandbox specs are advisory only" in policy
    assert "Existing sandbox spec already covers the selected kernel." not in policy

    workflow = WORKFLOW_YAML.read_text(encoding="utf-8")
    assert "sandbox_spec_missing" not in workflow
    assert "this never bypasses kernel generation" in workflow


def test_no_legacy_claude_agents_layer_remains():
    assert not LEGACY_AGENTS.exists()

    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert ".claude/agents/" not in text, path


def test_every_runtime_agent_declares_stop_conditions():
    for definition in AGENTS.glob("*.agent.md"):
        text = definition.read_text(encoding="utf-8")
        assert "## Stop Conditions" in text, definition


def test_declarative_workflow_role_bindings_resolve_to_runtime_agents():
    text = WORKFLOW_YAML.read_text(encoding="utf-8")
    assert "role_bindings:" in text
    assert "graph:" in text
    assert "finalizers:" in text
    assert "mode: sequential" in text
    assert "max_active_steps: 1" in text
    assert "never dispatch adjacent workflow steps concurrently" in text
    assert '"collector.output.dump_dir"' in text
    assert '"collector.output.runs"' in text
    assert '"collector.output.ok == true"' in text

    bound_agents = re.findall(r":\s*(ckh-[A-Za-z0-9-]+)\s*$", text, re.MULTILINE)
    assert bound_agents
    for agent_name in bound_agents:
        if agent_name == "ckh-ledger":
            continue
        assert (AGENTS / f"{agent_name}.agent.md").is_file(), agent_name

    for role in (
        "gatekeeper", "preflight", "collector", "reporter", "onboarder", "roofline",
        "critic", "budgeter", "classifier", "optimizer", "prover", "benchmarker",
        "tuner", "integrator",
    ):
        assert f"  {role}:" in text

    assert "intake_policy:" not in text
    assert "hard_rules:" not in text
    assert "step_contracts:" not in text
    assert "adoption_condition:" not in text


def test_policy_requires_tool_search_before_mcp_blocker():
    text = POLICY_YAML.read_text(encoding="utf-8")
    assert "after one `tool_search` attempt" in text
    assert "Do not treat deferred-tool absence as final before attempting the session-local `tool_search` load." in text


def test_policy_yaml_carries_runtime_semantics():
    text = POLICY_YAML.read_text(encoding="utf-8")
    assert "intake_policy:" in text
    assert "hard_rules:" in text
    assert "backend_blockers:" in text
    assert "step_contracts:" in text
    assert "loop_policy:" in text

    for role in (
        "gatekeeper", "preflight", "collector", "reporter", "onboarder", "roofline",
        "critic", "budgeter", "classifier", "optimizer", "prover", "benchmarker",
        "tuner", "integrator",
    ):
        assert f"  {role}:" in text

    assert "adoption_condition:" in text
    assert "mcp_unavailable:" in text
    assert "Do not continue to subagents after this blocker." in text
