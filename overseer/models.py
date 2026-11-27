"""Pydantic v2 schemas and status literals for Overseer MCP tools, workers, and manifests."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WORKER_ID_PATTERN = r"^[A-Za-z0-9_-]{1,32}$"
RUN_ID_PATTERN = r"^run_[0-9]{8}_[0-9]{6}_[a-f0-9]{6}$"
SHA256_PREFIX_PATTERN = r"^sha256:[a-f0-9]{64}$"

WorkerStatus = Literal[
    "queued",
    "running",
    "completed",
    "escalated",
    "interrupted",
    "cancelled",
    "halted_budget",
    "failed",
]

RunStatus = Literal[
    "running",
    "completed",
    "completed_with_escalations",
    "halted_budget",
    "interrupted",
    "cancelled",
    "failed",
    "rolled_back",
]

StaticCheckStatus = Literal["passed", "failed", "not_run"]
TierName = Literal["free", "cheap", "heavy"]
ProviderSortPreset = Literal["price", "throughput", "latency"]


def _clean_str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Expected a list of strings.")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("List items must be strings.")
        val = item.strip()
        if val and val not in seen:
            seen.add(val)
            cleaned.append(val)
    return cleaned


class ProviderRoutingRule(BaseModel):
    """Optional OpenRouter provider routing preferences for a tier or model."""

    model_config = ConfigDict(extra="forbid")

    sort: ProviderSortPreset | None = None
    order: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    only: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)
    allow_fallbacks: bool | None = None
    zdr: bool | None = None
    quantizations: Annotated[list[str], Field(max_length=20)] = Field(default_factory=list)

    @field_validator("order", "only", "quantizations", mode="before")
    @classmethod
    def _normalize_provider_lists(cls, val: Any) -> list[str]:
        return _clean_str_list(val)

    def to_openrouter_dict(self) -> dict[str, Any] | None:
        payload: dict[str, Any] = {}
        if self.sort is not None:
            payload["sort"] = self.sort
        if self.order:
            payload["order"] = list(self.order)
        if self.only:
            payload["only"] = list(self.only)
        if self.allow_fallbacks is not None:
            payload["allow_fallbacks"] = self.allow_fallbacks
        if self.zdr is not None:
            payload["zdr"] = self.zdr
        if self.quantizations:
            payload["quantizations"] = list(self.quantizations)
        return payload or None


class TierConfig(BaseModel):
    """Configured primary model, fallback chain, and per-run USD cap for one tier."""

    primary: str | None = None
    fallbacks: Annotated[list[str], Field(max_length=10)] | None = None
    max_run_usd: Annotated[float, Field(gt=0, le=50.0)] | None = None

    @field_validator("primary", mode="before")
    @classmethod
    def _normalize_primary(cls, val: Any) -> str | None:
        if val is None:
            return None
        if not isinstance(val, str):
            raise ValueError("primary model must be a string.")
        cleaned = val.strip()
        return cleaned or None

    @field_validator("fallbacks", mode="before")
    @classmethod
    def _normalize_fallbacks(cls, val: Any) -> list[str] | None:
        if val is None:
            return None
        return _clean_str_list(val)


class OverseerConfigFile(BaseModel):
    """Structured non-secret settings persisted in <data_dir>/config.json."""

    version: int = 1
    max_concurrent_workers: int | None = Field(default=None, ge=1, le=10)
    tiers: dict[TierName, TierConfig] = Field(default_factory=dict)
    provider_tiers: dict[TierName, ProviderRoutingRule] = Field(default_factory=dict)
    provider_models: Annotated[dict[str, ProviderRoutingRule], Field(max_length=100)] = Field(default_factory=dict)


class WorkerTaskInput(BaseModel):
    """Specification for a single folder-scoped worker task."""

    worker_id: str = Field(..., pattern=WORKER_ID_PATTERN)
    folder: str = Field(..., min_length=1)
    goal: str = Field(..., min_length=1, max_length=20_000)
    target_files: list[str] = Field(default_factory=list, max_length=50)
    model: str | None = None


ProjectRootArg = Annotated[str, Field(min_length=1)]
TasksArg = Annotated[list[WorkerTaskInput], Field(min_length=1, max_length=20)]
SharedContractArg = Annotated[str | None, Field(max_length=50_000)]
MaxRunUsdArg = Annotated[float, Field(gt=0, le=50.0)]
MaxStepsPerWorkerArg = Annotated[int, Field(ge=1, le=200)]
Max429WaitSecondsArg = Annotated[int, Field(ge=0, le=300)]
RequestTimeoutSecArg = Annotated[float, Field(gt=0, le=600.0)]


class RunLimits(BaseModel):
    """Execution caps and HTTP timeouts enforced for a worker run."""

    max_run_usd: MaxRunUsdArg = 0.25
    max_steps_per_worker: MaxStepsPerWorkerArg = 25
    max_429_wait_seconds: Max429WaitSecondsArg = 30
    request_timeout_sec: RequestTimeoutSecArg = 60.0


class StartRunRequest(BaseModel):
    """Validated input arguments for the start_run MCP tool."""

    project_root: ProjectRootArg
    tasks: TasksArg
    shared_contract: SharedContractArg = None
    max_run_usd: MaxRunUsdArg | None = None
    max_steps_per_worker: MaxStepsPerWorkerArg = 25
    max_429_wait_seconds: Max429WaitSecondsArg = 30
    request_timeout_sec: RequestTimeoutSecArg = 60.0

    @model_validator(mode="after")
    def _ensure_unique_worker_ids(self) -> "StartRunRequest":
        seen: set[str] = set()
        for task in self.tasks:
            norm_id = task.worker_id.lower()
            if norm_id in seen:
                raise ValueError(f"Duplicate worker_id in tasks (case-insensitive): {task.worker_id}")
            seen.add(norm_id)
        return self


class StartRunResponse(BaseModel):
    """Immediate acknowledgment returned by start_run."""

    run_id: str
    status: Literal["running"] = "running"
    project_root: str
    manifest_path: str
    worker_count: int
    warning: str | None = None


class FileEditRecord(BaseModel):
    """Backup and hash tracking for a pre-existing modified file."""

    backup_path: str
    initial_hash: str = Field(..., pattern=SHA256_PREFIX_PATTERN)
    worker_last_hash: str = Field(..., pattern=SHA256_PREFIX_PATTERN)
    added_lines: int = 0
    removed_lines: int = 0


class CreatedFileRecord(BaseModel):
    """Hash and size tracking for a newly created file."""

    worker_last_hash: str = Field(..., pattern=SHA256_PREFIX_PATTERN)
    line_count: int = 0


class StaticCheckResult(BaseModel):
    """Outcome of static syntax checks run during finish() and on any escalation when files were touched."""

    status: StaticCheckStatus = "not_run"
    errors: list[str] = Field(default_factory=list)


class WorkerManifestState(BaseModel):
    """Persistent state for one worker inside RunManifest."""

    worker_id: str
    folder: str
    goal: str
    target_files: list[str] = Field(default_factory=list)
    requested_model: str
    models_used: list[str] = Field(default_factory=list)
    status: WorkerStatus = "queued"
    steps_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    spend_usd: float = 0.0
    modified_files: dict[str, FileEditRecord] = Field(default_factory=dict)
    created_files: dict[str, CreatedFileRecord] = Field(default_factory=dict)
    created_dirs: list[str] | None = None
    static_check: StaticCheckResult = Field(default_factory=StaticCheckResult)
    summary: str = ""
    notes: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None
    stop_reason: str | None = None


class RunManifest(BaseModel):
    """Complete persisted manifest for an Overseer run stored under <data_dir>."""

    version: str
    run_id: str = Field(..., pattern=RUN_ID_PATTERN)
    server_pid: int
    server_pid_create_time: int | None = None
    created_at_iso: str
    completed_at_iso: str | None = None
    project_root: str
    status: RunStatus = "running"
    shared_contract: str | None = None
    limits: RunLimits
    spend_usd: float = 0.0
    cached_tokens: int = 0
    patch_dir: str
    workers: dict[str, WorkerManifestState] = Field(default_factory=dict)


class WorkerReportItem(BaseModel):
    """Bounded worker summary returned by get_report."""

    worker_id: str
    status: WorkerStatus
    steps_used: int
    models_used: list[str]
    spend_usd: float
    cached_tokens: int = 0
    diffstat: dict[str, str]
    static_check: StaticCheckResult
    summary: str
    notes: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None
    stop_reason: str | None = None
    patch_path: str | None = None


class GetReportResponse(BaseModel):
    """Compact bounded report returned by get_report (<= 8192 UTF-8 bytes)."""

    run_id: str
    project_root: str
    status: RunStatus
    spend_usd: float
    cached_tokens: int = 0
    patch_dir: str
    full_report_path: str
    truncated: bool = False
    workers: list[WorkerReportItem] = Field(default_factory=list)


class RollbackResponse(BaseModel):
    """Result payload returned by rollback."""

    run_id: str
    status: Literal["rolled_back"] = "rolled_back"
    restored_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)
    skipped_conflicts: list[str] = Field(default_factory=list)
    already_rolled_back: bool = False


class CancelRunResponse(BaseModel):
    """Result payload returned by cancel_run."""

    run_id: str
    project_root: str
    status: RunStatus
    cancelled_workers: list[str] = Field(default_factory=list)
    already_finished: bool = False


class ViewFileArgs(BaseModel):
    """Arguments for worker tool view_file."""

    path: str = Field(..., min_length=1)
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_line_range(self) -> "ViewFileArgs":
        if self.end_line is None:
            self.end_line = self.start_line + 299
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be greater than or equal to start_line ({self.start_line})."
            )
        return self


class FileOutlineArgs(BaseModel):
    """Arguments for worker tool file_outline."""

    path: str = Field(..., min_length=1)


class ListDirArgs(BaseModel):
    """Arguments for worker tool list_dir."""

    path: str = "."


class SearchDirArgs(BaseModel):
    """Arguments for worker tool search_dir."""

    query: str = Field(..., min_length=1)
    path: str = "."
    case_insensitive: bool = False
    file_glob: str | None = None


class ListModelsArgs(BaseModel):
    """Arguments for listing and filtering OpenRouter models."""

    free_only: bool = False
    tools_only: bool = True
    search: str | None = None
    limit: int = Field(default=50, ge=1, le=200)


class ModelSummary(BaseModel):
    """Compact summary of an OpenRouter model."""

    id: str
    name: str = ""
    context_length: int | None = None
    prompt_price_per_m: float = 0.0
    completion_price_per_m: float = 0.0
    is_free: bool = False
    supports_tools: bool = False


class FindFileArgs(BaseModel):
    """Arguments for worker tool find_file."""

    pattern: str = Field(..., min_length=1)
    path: str = "."


class ReplaceBlockArgs(BaseModel):
    """Arguments for worker tool replace_block."""

    path: str = Field(..., min_length=1)
    target: str
    replacement: str


class WriteNewFileArgs(BaseModel):
    """Arguments for worker tool write_new_file."""

    path: str = Field(..., min_length=1)
    content: str


class FinishArgs(BaseModel):
    """Arguments for worker tool finish."""

    summary: str = Field(..., min_length=1)
    notes: list[str] = Field(default_factory=list)
