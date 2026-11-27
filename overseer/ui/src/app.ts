import {
  App,
  applyDocumentTheme,
  applyHostStyleVariables,
  type McpUiHostContext,
  type McpUiHostContextChangedNotification,
} from "@modelcontextprotocol/ext-apps";
import { z } from "zod";
import {
  buildConfigUpdatePayload,
  configToDraft,
  emptyRoutingDraft,
  type ConfigFormDraft,
  type RoutingRuleDraft,
} from "./config_helpers.ts";
import { buildRenderSignature } from "./signature.ts";
import type {
  BannerState,
  CustomInstructionsPayload,
  ListModelsPayload,
  ListRunsPayload,
  ModelCatalogEntry,
  OpenMonitorPayload,
  OverseerConfigSnapshot,
  ProviderSortPreset,
  RollbackRunPayload,
  RunSnapshot,
  RunSummary,
  TabId,
  TierName,
  WorkerPatchPayload,
  WorkerSnapshot,
} from "./types.ts";

const MAX_CONCURRENT_TOOL_CALLS = 4;
const POLL_INTERVAL_MS = 2500;
const IDLE_POLL_INTERVAL_MS = 12000;
const MAX_CUSTOM_INSTRUCTIONS_CHARS = 20000;
const SVG_NS = "http://www.w3.org/2000/svg";
const AuxPaneResultSchema = z.object({ isError: z.boolean().optional() }).passthrough();

type IconName =
  | "layers"
  | "refresh"
  | "check"
  | "minus"
  | "file"
  | "plus"
  | "trash"
  | "alert"
  | "info"
  | "stop"
  | "chat"
  | "copy"
  | "search"
  | "undo"
  | "diff"
  | "lock";

const ICON_PATHS: Record<IconName, string[]> = {
  layers: ["M8 2 14 5.5 8 9 2 5.5 8 2Z", "M2 8.5 8 12 14 8.5", "M2 11.5 8 15 14 11.5"],
  refresh: ["M13.5 8a5.5 5.5 0 1 1-1.4-3.7", "M13.5 2.5v3h-3"],
  check: ["M3.5 8.5 6.5 11.5 12.5 4.5"],
  minus: ["M3.5 8h9"],
  file: ["M4 2.5h5.5L12.5 5.5v8H4v-11Z", "M9.5 2.5v3h3"],
  plus: ["M8 3.5v9", "M3.5 8h9"],
  trash: ["M3.5 4.5h9", "M6 4.5V3h4v1.5", "M5 4.5v8h6v-8"],
  alert: ["M8 2.5 14 13H2L8 2.5Z", "M8 6.5v3", "M8 11.2h.01"],
  info: ["M8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12Z", "M8 7.2v3.8", "M8 5h.01"],
  stop: ["M4.5 4.5h7v7h-7z"],
  chat: ["M2.5 3.5h11v7.5H6.5L3.5 13.5v-2.5h-1v-7.5Z"],
  copy: ["M5.5 5.5h7v8h-7z", "M3.5 10.5H3a.5.5 0 0 1-.5-.5V3a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 .5.5v.5"],
  search: ["M7 11.5a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9Z", "M10.5 10.5 13.5 13.5"],
  undo: ["M4 7h6.5a3.5 3.5 0 1 1 0 7H8", "M6.5 4.5 4 7l2.5 2.5"],
  diff: ["M5.5 3.5h5", "M8 1v5", "M5.5 12.5h5"],
  lock: ["M4.5 7.5h7v6h-7z", "M6 7.5V5.5a2 2 0 1 1 4 0v2"],
};

const state: {
  connected: boolean;
  loading: boolean;
  refreshing: boolean;
  cancelling: boolean;
  rollingBack: boolean;
  confirmingRollback: boolean;
  sendingMessage: boolean;
  receivedInitialResult: boolean;
  auxPaneSupported: boolean;
  canSendMessage: boolean;
  activeTab: TabId;
  projectFilter: string | null;
  runs: RunSummary[];
  activeCount: number;
  runsTotalCount: number | null;
  runsHasMore: boolean;
  loadingMoreRuns: boolean;
  selectedKey: { project_root: string; run_id: string } | null;
  selectedRun: RunSnapshot | null;
  banner: BannerState | null;
} = {
  connected: false,
  loading: true,
  refreshing: false,
  cancelling: false,
  rollingBack: false,
  confirmingRollback: false,
  sendingMessage: false,
  receivedInitialResult: false,
  auxPaneSupported: true,
  canSendMessage: false,
  activeTab: "runs",
  projectFilter: null,
  runs: [],
  activeCount: 0,
  runsTotalCount: null,
  runsHasMore: false,
  loadingMoreRuns: false,
  selectedKey: null,
  selectedRun: null,
  banner: null,
};

let standaloneMode = false;
const previewSnapshots = new Map<string, RunSnapshot>();
const workerDiffs = new Map<
  string,
  { open: boolean; loading: boolean; patch: string | null; error: string | null }
>();
let runSearchQuery = "";
let inFlightToolCalls = 0;
let pollTimer: number | null = null;
let fallbackTimer: number | null = null;
let pollInProgress = false;
let refreshToken = 0;
let pendingPostConnectUpgrade: "snapshot" | "refresh" | null = null;
let lastRunsSignature = "";
let bannerClearTimer: number | null = null;
let fieldIdCounter = 0;

// Custom instructions state
let savedInstructions = "";
let instructionsLoaded = false;
let instructionsDirty = false;
let savingInstructions = false;
let confirmingClearInstructions = false;
let instructionsLoadError: string | null = null;

// Config & provider routing state
let savedConfig: OverseerConfigSnapshot | null = null;
let configDraft: ConfigFormDraft | null = null;
let configLoaded = false;
let configDirty = false;
let savingConfig = false;
const routingDetailsOpenState: Record<TierName, boolean> = {
  free: false,
  cheap: false,
  heavy: false,
};
let modelsCatalog: ModelCatalogEntry[] = [];
let modelsCatalogLoaded = false;
let modelsCatalogLoading = false;

// Persistent DOM anchors (never torn down on 2.5s polling)
let shellMounted = false;
let topbarBrandEl: HTMLElement | null = null;
let topbarActionsEl: HTMLElement | null = null;
let tabRunsBtn: HTMLButtonElement | null = null;
let tabModelsBtn: HTMLButtonElement | null = null;
let tabInstrBtn: HTMLButtonElement | null = null;
let bannerSlotEl: HTMLElement | null = null;
let runsPanelEl: HTMLElement | null = null;
let runsLayoutMounted = false;
let runsCountTitleEl: HTMLHeadingElement | null = null;
let runsProjectSelectEl: HTMLSelectElement | null = null;
let runsSearchInputEl: HTMLInputElement | null = null;
let runsListEl: HTMLElement | null = null;
let runsDetailSlotEl: HTMLElement | null = null;
let lastProjectOptionsKey = "";
let modelsPanelEl: HTMLElement | null = null;
let instructionsPanelEl: HTMLElement | null = null;

let configStatusPill: HTMLSpanElement | null = null;
let configSaveBtn: HTMLButtonElement | null = null;
let configResetBtn: HTMLButtonElement | null = null;

let instructionsTextarea: HTMLTextAreaElement | null = null;
let instructionsStatusPill: HTMLSpanElement | null = null;
let instructionsCountEl: HTMLSpanElement | null = null;
let instructionsBarFillEl: HTMLDivElement | null = null;
let instructionsErrorBannerEl: HTMLDivElement | null = null;
let instructionsSaveBtn: HTMLButtonElement | null = null;
let instructionsClearBtn: HTMLButtonElement | null = null;
let instructionsConfirmClearBtn: HTMLButtonElement | null = null;
let instructionsCancelClearBtn: HTMLButtonElement | null = null;

const waitQueue: Array<() => void> = [];

async function withToolSlot<T>(fn: () => Promise<T>): Promise<T> {
  while (inFlightToolCalls >= MAX_CONCURRENT_TOOL_CALLS) {
    await new Promise<void>((resolve) => {
      waitQueue.push(resolve);
    });
  }
  inFlightToolCalls += 1;
  try {
    return await fn();
  } finally {
    inFlightToolCalls = Math.max(0, inFlightToolCalls - 1);
    const next = waitQueue.shift();
    if (next) next();
  }
}

function nextFieldId(prefix: string): string {
  fieldIdCounter += 1;
  return `ov-${prefix}-${fieldIdCounter}`;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function svgIcon(name: IconName, small = false): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.5");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", small ? "ov-icon ov-icon-sm" : "ov-icon");
  for (const d of ICON_PATHS[name]) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    svg.appendChild(path);
  }
  return svg;
}

const STATUS_LABELS: Record<string, string> = {
  running: "Running",
  completed: "Completed",
  completed_with_escalations: "Escalations",
  rolled_back: "Rolled Back",
  halted_budget: "Budget Halted",
  failed: "Failed",
  escalated: "Escalated",
  passed: "Check Passed",
  not_run: "No Check",
  skipped: "Skipped",
  idle: "Idle",
};

function formatStatusLabel(status: string): string {
  if (status in STATUS_LABELS) return STATUS_LABELS[status];
  return status
    .split("_")
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function formatStaticCheckLabel(status: string): string {
  if (status === "passed") return "Check Passed";
  if (status === "failed") return "Check Failed";
  if (status === "skipped") return "Check Skipped";
  if (status === "not_run") return "No Check";
  return `Check: ${formatStatusLabel(status)}`;
}

function statusPill(status: string, label?: string, pulse = false): HTMLSpanElement {
  const pill = el("span", statusPillClass(status));
  const dot = el("span", pulse ? "ov-status-dot ov-dot-pulse" : "ov-status-dot");
  dot.setAttribute("aria-hidden", "true");
  pill.appendChild(dot);
  pill.appendChild(el("span", undefined, label ?? formatStatusLabel(status)));
  return pill;
}

function setBanner(kind: BannerState["kind"], text: string, autoClearMs = 0): void {
  if (bannerClearTimer !== null) {
    window.clearTimeout(bannerClearTimer);
    bannerClearTimer = null;
  }
  state.banner = { kind, text };
  renderBanner();
  if (autoClearMs > 0) {
    bannerClearTimer = window.setTimeout(() => {
      state.banner = null;
      bannerClearTimer = null;
      renderBanner();
    }, autoClearMs);
  }
}

function applyThemeFromContext(
  ctx?: McpUiHostContext | McpUiHostContextChangedNotification["params"]
): void {
  if (!ctx) return;
  if (ctx.theme) {
    applyDocumentTheme(ctx.theme);
  }
  if (ctx.styles?.variables) {
    const fontVars: Record<string, string | undefined> = {};
    for (const [key, val] of Object.entries(ctx.styles.variables)) {
      if (key.startsWith("--font-") && typeof val === "string") {
        fontVars[key] = val;
      }
    }
    if (Object.keys(fontVars).length > 0) {
      applyHostStyleVariables(fontVars as Parameters<typeof applyHostStyleVariables>[0]);
    }
  }
}

function isStandaloneWindow(): boolean {
  try {
    return window.self === window.top;
  } catch {
    return false;
  }
}

function extractResultText(content: unknown): string {
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const item of content) {
    if (item && typeof item === "object" && "text" in item && typeof item.text === "string") {
      parts.push(item.text);
    }
  }
  return parts.join("\n").trim();
}

function extractToolPayload<T>(result: {
  isError?: boolean;
  structuredContent?: unknown;
  content?: unknown;
}): T | null {
  if (result.isError) {
    const errText = extractResultText(result.content) || "Tool execution failed.";
    throw new Error(errText);
  }
  if (result.structuredContent && typeof result.structuredContent === "object") {
    return result.structuredContent as T;
  }
  const text = extractResultText(result.content);
  if (text) {
    try {
      return JSON.parse(text) as T;
    } catch {
      return null;
    }
  }
  return null;
}

function hasRunningRun(): boolean {
  if (state.selectedRun?.status === "running") return true;
  return state.runs.some((r) => r.status === "running");
}

function clearPolling(): void {
  if (pollTimer !== null) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function workerDiffKey(projectRoot: string, runId: string, workerId: string): string {
  return `${projectRoot}::${runId}::${workerId}`;
}

function getWorkerDiffsSignature(): string {
  const entries: Array<[string, boolean, boolean, number, string | null]> = [];
  for (const [k, v] of workerDiffs.entries()) {
    if (v.open || v.loading || v.error) {
      entries.push([k, v.open, v.loading, v.patch?.length ?? -1, v.error]);
    }
  }
  entries.sort((a, b) => a[0].localeCompare(b[0]));
  return JSON.stringify(entries);
}

function schedulePollingIfNeeded(): void {
  clearPolling();
  if (!state.connected || standaloneMode) {
    return;
  }
  if (typeof document !== "undefined" && document.visibilityState !== "visible") {
    return;
  }
  const interval = hasRunningRun() ? POLL_INTERVAL_MS : IDLE_POLL_INTERVAL_MS;
  pollTimer = window.setTimeout(async () => {
    pollTimer = null;
    await refreshAll(true);
  }, interval);
}

function formatUsd(amount: number): string {
  const safe = Number.isFinite(amount) ? amount : 0;
  return `$${safe.toFixed(4)}`;
}

function formatShortTime(iso: string | null): string {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function pathBasename(fullPath: string): string {
  const normalized = fullPath.replace(/\\/g, "/").replace(/\/+$/, "");
  const idx = normalized.lastIndexOf("/");
  return idx >= 0 ? normalized.slice(idx + 1) || fullPath : fullPath;
}

function formatDuration(startIso: string, endIso: string | null, status: string): string {
  if (!endIso) {
    return status === "running" ? "In progress" : "-";
  }
  const startMs = new Date(startIso).getTime();
  const endMs = new Date(endIso).getTime();
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs < startMs) {
    return "-";
  }
  const totalSec = Math.round((endMs - startMs) / 1000);
  if (totalSec < 60) return `${totalSec}s`;
  const mins = Math.floor(totalSec / 60);
  const secs = totalSec % 60;
  if (mins < 60) return `${mins}m ${String(secs).padStart(2, "0")}s`;
  const hrs = Math.floor(mins / 60);
  const remMins = mins % 60;
  return `${hrs}h ${String(remMins).padStart(2, "0")}m`;
}

function shortRunHash(runId: string): string {
  const parts = runId.split("_");
  if (parts.length >= 2) {
    const tail = parts[parts.length - 1];
    if (tail) return tail.slice(0, 8);
  }
  return runId.slice(-6);
}

function statusPillClass(status: string): string {
  const known = new Set([
    "running",
    "completed",
    "passed",
    "escalated",
    "completed_with_escalations",
    "rolled_back",
    "halted_budget",
    "failed",
  ]);
  return known.has(status) ? `ov-pill ov-pill-${status}` : "ov-pill";
}

function pickDefaultRunKey(runs: RunSummary[]): { project_root: string; run_id: string } | null {
  if (runs.length === 0) return null;
  const running = runs.find((r) => r.status === "running");
  const chosen = running ?? runs[0];
  return { project_root: chosen.project_root, run_id: chosen.run_id };
}

const app = new App({ name: "Overseer Monitor", version: "0.3.0" }, {});

async function callServerJson<T>(name: string, args: Record<string, unknown>): Promise<T> {
  return await withToolSlot(async () => {
    const result = await app.callServerTool({ name, arguments: args });
    const parsed = extractToolPayload<T>(result);
    if (parsed === null) {
      throw new Error(`Empty response from ${name}`);
    }
    return parsed;
  });
}

async function fetchSnapshot(projectRoot: string, runId: string): Promise<void> {
  const snapshot = await callServerJson<RunSnapshot>("app_get_run_snapshot", {
    project_root: projectRoot,
    run_id: runId,
  });
  if (
    state.selectedKey &&
    (state.selectedKey.run_id !== runId || state.selectedKey.project_root !== projectRoot)
  ) {
    return;
  }
  state.selectedKey = { project_root: snapshot.project_root, run_id: snapshot.run_id };
  state.selectedRun = snapshot;
}

async function refreshAll(silent = false): Promise<void> {
  if (!state.connected) return;
  if (standaloneMode) {
    state.loading = false;
    state.refreshing = false;
    if (!silent) {
      setBanner("info", "Standalone preview mode — showing loaded snapshot.", 2500);
    }
    render();
    return;
  }
  if (silent && (pollInProgress || inFlightToolCalls >= 2)) {
    return;
  }
  const token = ++refreshToken;
  pollInProgress = true;
  if (!silent) {
    state.refreshing = true;
    renderTopbar();
    renderRunsPanel();
  }

  try {
    const listArgs: Record<string, unknown> = {
      limit: Math.max(20, state.runs.length),
      offset: 0,
    };
    if (state.projectFilter) {
      listArgs.project_root = state.projectFilter;
    }
    const listData = await callServerJson<ListRunsPayload>("app_list_runs", listArgs);
    if (token !== refreshToken) return;
    state.runs = Array.isArray(listData.runs) ? listData.runs : [];
    state.activeCount = typeof listData.active_count === "number" ? listData.active_count : 0;
    state.runsTotalCount = typeof listData.total_count === "number" ? listData.total_count : null;
    state.runsHasMore = Boolean(listData.has_more);

    let targetKey = state.selectedKey;
    if (
      targetKey &&
      state.projectFilter &&
      targetKey.project_root !== state.projectFilter
    ) {
      targetKey = null;
    }
    if (!targetKey) {
      targetKey = pickDefaultRunKey(state.runs);
    }

    if (targetKey) {
      try {
        await fetchSnapshot(targetKey.project_root, targetKey.run_id);
      } catch (err) {
        if (token !== refreshToken) return;
        const fallback = pickDefaultRunKey(state.runs);
        if (
          fallback &&
          (fallback.project_root !== targetKey.project_root || fallback.run_id !== targetKey.run_id)
        ) {
          state.selectedKey = fallback;
          await fetchSnapshot(fallback.project_root, fallback.run_id);
        } else {
          throw err;
        }
      }
    } else {
      state.selectedKey = null;
      state.selectedRun = null;
    }
    if (!silent && state.banner?.kind === "error") {
      state.banner = null;
    }
  } catch (err) {
    if (token !== refreshToken) return;
    const msg = err instanceof Error ? err.message : String(err);
    setBanner("error", `Refresh failed: ${msg}`);
  } finally {
    if (token === refreshToken) {
      pollInProgress = false;
      state.loading = false;
      state.refreshing = false;
      renderTopbar();
      renderBanner();
      renderRunsPanel();
      schedulePollingIfNeeded();
    }
  }
}

async function handleLoadMoreRuns(): Promise<void> {
  if (!state.connected || state.loadingMoreRuns || !state.runsHasMore || standaloneMode) return;
  state.loadingMoreRuns = true;
  renderRunsPanel();
  try {
    const listArgs: Record<string, unknown> = {
      limit: 20,
      offset: state.runs.length,
    };
    if (state.projectFilter) {
      listArgs.project_root = state.projectFilter;
    }
    const listData = await callServerJson<ListRunsPayload>("app_list_runs", listArgs);
    const incoming = Array.isArray(listData.runs) ? listData.runs : [];
    const seen = new Set(state.runs.map((r) => `${r.project_root}::${r.run_id}`));
    for (const r of incoming) {
      const key = `${r.project_root}::${r.run_id}`;
      if (!seen.has(key)) {
        seen.add(key);
        state.runs.push(r);
      }
    }
    if (typeof listData.active_count === "number") {
      state.activeCount = listData.active_count;
    }
    state.runsTotalCount = typeof listData.total_count === "number" ? listData.total_count : null;
    state.runsHasMore = Boolean(listData.has_more);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setBanner("error", `Failed to load more runs: ${msg}`);
  } finally {
    state.loadingMoreRuns = false;
    renderRunsPanel();
  }
}

async function handleSelectRun(run: RunSummary): Promise<void> {
  refreshToken += 1;
  pollInProgress = false;
  state.confirmingRollback = false;
  state.selectedKey = { project_root: run.project_root, run_id: run.run_id };
  if (standaloneMode) {
    const key = `${run.project_root}::${run.run_id}`;
    state.selectedRun = previewSnapshots.get(key) ?? {
      run_id: run.run_id,
      project_root: run.project_root,
      status: run.status,
      created_at_iso: run.created_at_iso,
      completed_at_iso: run.completed_at_iso,
      spend_usd: run.spend_usd,
      limits: {
        max_run_usd: run.max_run_usd,
        max_steps_per_worker: 25,
        max_429_wait_seconds: 30,
        request_timeout_sec: 60,
      },
      workers: [],
    };
    renderTopbar();
    renderRunsPanel();
    return;
  }
  state.refreshing = true;
  renderTopbar();
  renderRunsPanel();
  try {
    await fetchSnapshot(run.project_root, run.run_id);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setBanner("error", `Could not load run ${run.run_id}: ${msg}`);
  } finally {
    state.refreshing = false;
    renderTopbar();
    renderRunsPanel();
    schedulePollingIfNeeded();
  }
}

async function handleCancelRun(run: RunSnapshot): Promise<void> {
  if (state.cancelling || run.status !== "running") return;
  if (standaloneMode) {
    setBanner("info", "Standalone preview mode — changes are not written to disk.", 3000);
    return;
  }
  state.cancelling = true;
  renderRunsPanel();
  try {
    await callServerJson<Record<string, unknown>>("app_cancel_run", {
      project_root: run.project_root,
      run_id: run.run_id,
    });
    setBanner("info", `Cancellation requested for ${run.run_id}.`, 4000);
    await refreshAll(false);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setBanner("error", `Cancel failed: ${msg}`);
  } finally {
    state.cancelling = false;
    renderRunsPanel();
    schedulePollingIfNeeded();
  }
}

async function handleRollbackRun(run: RunSnapshot): Promise<void> {
  if (state.rollingBack || run.status === "running" || run.status === "rolled_back") return;
  if (standaloneMode) {
    state.confirmingRollback = false;
    setBanner("info", "Standalone preview mode — rollback is not available.", 3000);
    renderRunsPanel();
    return;
  }
  state.rollingBack = true;
  state.confirmingRollback = false;
  renderRunsPanel();
  try {
    const res = await callServerJson<RollbackRunPayload>("app_rollback_run", {
      project_root: run.project_root,
      run_id: run.run_id,
      confirm: true,
    });
    if (res.already_rolled_back) {
      setBanner("info", `Run ${run.run_id} was already rolled back.`, 3500);
    } else {
      const restored = Array.isArray(res.restored_files) ? res.restored_files.length : 0;
      const deleted = Array.isArray(res.deleted_files) ? res.deleted_files.length : 0;
      const conflicts = Array.isArray(res.skipped_conflicts) ? res.skipped_conflicts.length : 0;
      const conflictNote = conflicts > 0 ? `, ${conflicts} conflict(s) skipped` : "";
      setBanner(
        conflicts > 0 ? "info" : "success",
        `Rolled back ${run.run_id} (${restored} restored, ${deleted} deleted${conflictNote}).`,
        4500
      );
    }
    await refreshAll(false);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setBanner("error", `Rollback failed: ${msg}`);
  } finally {
    state.rollingBack = false;
    renderRunsPanel();
    schedulePollingIfNeeded();
  }
}

async function handleToggleWorkerDiff(run: RunSnapshot, worker: WorkerSnapshot): Promise<void> {
  const key = workerDiffKey(run.project_root, run.run_id, worker.worker_id);
  const existing = workerDiffs.get(key);
  if (existing?.open) {
    existing.open = false;
    renderRunsPanel();
    return;
  }
  if (standaloneMode) {
    workerDiffs.set(key, {
      open: true,
      loading: false,
      patch:
        "--- a/src/math_utils.py\n+++ b/src/math_utils.py\n@@ -1,3 +1,9 @@\n def add(a: float, b: float) -> float:\n     return a + b\n+\n+def subtract(a: float, b: float) -> float:\n+    return a - b\n+\n+def multiply(a: float, b: float) -> float:\n+    return a * b",
      error: null,
    });
    renderRunsPanel();
    return;
  }
  if (existing && existing.patch !== null && !existing.error && run.status !== "running") {
    existing.open = true;
    renderRunsPanel();
    return;
  }
  workerDiffs.set(key, {
    open: true,
    loading: true,
    patch: existing?.patch ?? null,
    error: null,
  });
  renderRunsPanel();
  try {
    const res = await callServerJson<WorkerPatchPayload>("app_get_worker_patch", {
      project_root: run.project_root,
      run_id: run.run_id,
      worker_id: worker.worker_id,
    });
    workerDiffs.set(key, {
      open: true,
      loading: false,
      patch: typeof res.patch === "string" ? res.patch : "",
      error: null,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    workerDiffs.set(key, {
      open: true,
      loading: false,
      patch: null,
      error: msg,
    });
  } finally {
    renderRunsPanel();
  }
}

async function handleCollectResultsInChat(run: RunSnapshot): Promise<void> {
  if (state.sendingMessage || !state.canSendMessage) return;
  state.sendingMessage = true;
  renderRunsPanel();
  try {
    const res = await app.sendMessage({
      role: "user",
      content: [
        {
          type: "text",
          text: `Collect results for run ${run.run_id} in ${run.project_root} using get_report.`,
        },
      ],
    });
    if (res?.isError) {
      setBanner("error", "Host declined sending message to chat.");
    } else {
      setBanner("success", "Collect-results prompt sent to chat.", 3000);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setBanner("error", `Failed to send chat message: ${msg}`);
  } finally {
    state.sendingMessage = false;
    renderRunsPanel();
  }
}

async function handleOpenFile(uri: string): Promise<void> {
  if (!uri || !uri.startsWith("file://")) return;
  if (standaloneMode) {
    setBanner("info", "Standalone preview mode — file viewer is only available inside Antigravity.", 3000);
    return;
  }
  try {
    const result = await app.request(
      {
        method: "com.google.antigravity/toggle-aux-pane",
        params: { tabType: "fileView", uri },
      } as unknown as Parameters<typeof app.request>[0],
      AuxPaneResultSchema
    );
    if (result?.isError) {
      setBanner("info", "File viewer unavailable in this surface or file missing.", 3000);
    }
  } catch (err) {
    const code = (err as { code?: unknown })?.code;
    const msg = err instanceof Error ? err.message : String(err);
    if (
      code === -32601 ||
      msg.includes("-32601") ||
      msg.toLowerCase().includes("method not found")
    ) {
      if (state.auxPaneSupported) {
        state.auxPaneSupported = false;
        renderRunsPanel();
      }
    }
  }
}

// ============================================================================
// Tab 1: Runs View
// ============================================================================

function renderBudgetBar(spendUsd: number, maxRunUsd: number, labelPrefix: string): HTMLElement {
  const wrap = el("div", "ov-progress-wrap");
  const safeMax = maxRunUsd > 0 ? maxRunUsd : 0;
  const pct = safeMax > 0 ? Math.min(100, Math.max(0, (spendUsd / safeMax) * 100)) : 0;

  const row = el("div", "ov-progress-row");
  row.appendChild(
    el("span", "ov-num", `${labelPrefix}: ${formatUsd(spendUsd)} / ${formatUsd(safeMax)}`)
  );
  row.appendChild(el("span", "ov-num", `${pct.toFixed(0)}%`));

  const track = el("div", "ov-bar");
  track.setAttribute("role", "progressbar");
  track.setAttribute("aria-label", labelPrefix);
  track.setAttribute("aria-valuemin", "0");
  track.setAttribute("aria-valuemax", "100");
  track.setAttribute("aria-valuenow", String(Math.round(pct)));

  const fillClass =
    pct >= 95 ? "ov-bar-fill ov-bar-fill-danger" : pct >= 75 ? "ov-bar-fill ov-bar-fill-warn" : "ov-bar-fill";
  const fill = el("div", fillClass);
  fill.style.width = `${pct}%`;
  track.appendChild(fill);

  wrap.appendChild(row);
  wrap.appendChild(track);
  return wrap;
}

function renderWorkerCard(
  worker: WorkerSnapshot,
  maxRunUsd: number,
  showWorkerBudgetBar = true,
  run?: RunSnapshot
): HTMLElement {
  const card = el("article", "ov-worker-card");

  const head = el("div", "ov-worker-head");
  const titleWrap = el("div", "ov-worker-title");
  titleWrap.appendChild(el("h3", "ov-worker-id", worker.worker_id));
  const folderPill = el("span", "ov-pill ov-pill-truncate ov-mono", worker.folder);
  folderPill.title = worker.folder;
  titleWrap.appendChild(folderPill);
  head.appendChild(titleWrap);

  const badgesWrap = el("div", "ov-worker-title");
  badgesWrap.appendChild(statusPill(worker.status, undefined, worker.status === "running"));
  const checkStatus = worker.static_check?.status ?? "not_run";
  badgesWrap.appendChild(statusPill(checkStatus, formatStaticCheckLabel(checkStatus)));

  const diffState = run
    ? workerDiffs.get(workerDiffKey(run.project_root, run.run_id, worker.worker_id))
    : undefined;
  if (run) {
    const diffBtn = el("button", "ov-copy-btn");
    diffBtn.type = "button";
    diffBtn.setAttribute("aria-expanded", diffState?.open ? "true" : "false");
    diffBtn.appendChild(svgIcon("diff", true));
    diffBtn.appendChild(el("span", undefined, diffState?.open ? "Hide Diff" : "View Diff"));
    diffBtn.addEventListener("click", () => {
      void handleToggleWorkerDiff(run, worker);
    });
    badgesWrap.appendChild(diffBtn);
  }
  head.appendChild(badgesWrap);
  card.appendChild(head);

  const kvGrid = el("div", "ov-kv-grid");

  const stepsBox = el("div", "ov-kv-box");
  stepsBox.appendChild(el("div", "ov-kv-label", "Steps"));
  stepsBox.appendChild(el("div", "ov-kv-value ov-num", `${worker.steps_used} / ${worker.max_steps}`));
  const stepPct =
    worker.max_steps > 0 ? Math.min(100, Math.max(0, (worker.steps_used / worker.max_steps) * 100)) : 0;
  const stepMiniBar = el("div", "ov-mini-bar");
  const stepMiniFill = el(
    "div",
    stepPct >= 95 ? "ov-mini-bar-fill ov-mini-bar-danger" : stepPct >= 75 ? "ov-mini-bar-fill ov-mini-bar-warn" : "ov-mini-bar-fill"
  );
  stepMiniFill.style.width = `${stepPct}%`;
  stepMiniBar.appendChild(stepMiniFill);
  stepsBox.appendChild(stepMiniBar);
  kvGrid.appendChild(stepsBox);

  const modelBox = el("div", "ov-kv-box ov-kv-box-model");
  modelBox.appendChild(el("div", "ov-kv-label", "Model"));
  const modelText =
    worker.models_used.length > 0
      ? worker.models_used.join(", ")
      : worker.requested_model
        ? `${worker.requested_model} (pending)`
        : "-";
  modelBox.appendChild(el("div", "ov-kv-value ov-mono", modelText));
  if (
    worker.requested_model &&
    worker.models_used.length > 0 &&
    !worker.models_used.includes(worker.requested_model)
  ) {
    modelBox.appendChild(
      el("div", "ov-failover-note ov-mono", `Failover from ${worker.requested_model}`)
    );
  }
  kvGrid.appendChild(modelBox);

  const spendBox = el("div", "ov-kv-box");
  spendBox.appendChild(el("div", "ov-kv-label", "Worker Spend"));
  spendBox.appendChild(
    el("div", "ov-kv-value ov-num", `${formatUsd(worker.spend_usd)} / ${formatUsd(maxRunUsd)}`)
  );
  kvGrid.appendChild(spendBox);

  card.appendChild(kvGrid);
  if (showWorkerBudgetBar) {
    card.appendChild(renderBudgetBar(worker.spend_usd, maxRunUsd, "Spend vs run cap"));
  }

  if (worker.escalation_reason || worker.stop_reason) {
    const reasonBox = el("div", "ov-block-box ov-block-warn");
    reasonBox.textContent = [
      worker.escalation_reason ? `Escalation: ${worker.escalation_reason}` : "",
      worker.stop_reason ? `Stop reason: ${worker.stop_reason}` : "",
    ]
      .filter(Boolean)
      .join(" | ");
    card.appendChild(reasonBox);
  }

  if (worker.static_check?.errors && worker.static_check.errors.length > 0) {
    const errBox = el("div", "ov-block-box ov-block-danger");
    errBox.appendChild(el("div", "ov-kv-label", "Static Check Errors"));
    const ul = el("ul", "ov-list ov-mono");
    for (const errLine of worker.static_check.errors) {
      ul.appendChild(el("li", undefined, errLine));
    }
    errBox.appendChild(ul);
    card.appendChild(errBox);
  }

  if (worker.summary) {
    const sumBox = el("div", "ov-block-section");
    sumBox.appendChild(el("div", "ov-kv-label", "Summary"));
    sumBox.appendChild(el("div", "ov-summary-text", worker.summary));
    card.appendChild(sumBox);
  }

  if (worker.notes && worker.notes.length > 0) {
    const notesBox = el("div", "ov-block-section");
    notesBox.appendChild(el("div", "ov-kv-label", "Out-of-Folder Notes"));
    const ul = el("ul", "ov-list");
    for (const note of worker.notes) {
      ul.appendChild(el("li", undefined, note));
    }
    notesBox.appendChild(ul);
    card.appendChild(notesBox);
  }

  if (worker.touched_files && worker.touched_files.length > 0) {
    const filesBox = el("div", "ov-block-section");
    filesBox.appendChild(el("div", "ov-kv-label", "Touched Files"));
    const list = el("div", "ov-files-list");
    for (const tf of worker.touched_files) {
      const chip = el("div", "ov-file-chip");
      const kindClass =
        tf.kind === "created"
          ? "ov-pill ov-pill-completed"
          : tf.kind === "modified"
            ? "ov-pill ov-pill-running"
            : tf.kind === "deleted"
              ? "ov-pill ov-pill-failed"
              : "ov-pill";
      chip.appendChild(el("span", kindClass, tf.kind));
      const pathSpan = el("span", "ov-mono ov-file-path", tf.rel_path);
      pathSpan.title = tf.rel_path;
      chip.appendChild(pathSpan);
      chip.appendChild(el("span", "ov-muted ov-num ov-file-stat", tf.stat));
      if (state.auxPaneSupported && tf.abs_uri && tf.abs_uri.startsWith("file://")) {
        const openBtn = el("button", "ov-file-open-btn");
        openBtn.type = "button";
        openBtn.setAttribute("aria-label", `Open ${tf.rel_path}`);
        openBtn.appendChild(svgIcon("file", true));
        openBtn.appendChild(el("span", undefined, "Open"));
        openBtn.addEventListener("click", () => {
          void handleOpenFile(tf.abs_uri);
        });
        chip.appendChild(openBtn);
      }
      list.appendChild(chip);
    }
    filesBox.appendChild(list);
    card.appendChild(filesBox);
  }

  if (diffState?.open) {
    const diffSection = el("div", "ov-block-section");
    diffSection.appendChild(el("div", "ov-kv-label", "Worker Patch Diff"));
    const drawer = el("div", "ov-diff-drawer");
    const pre = el("pre", "ov-diff-pre");
    if (diffState.loading) {
      pre.textContent = "Loading patch...";
    } else if (diffState.error) {
      pre.textContent = `Failed to load patch: ${diffState.error}`;
    } else {
      pre.textContent = diffState.patch && diffState.patch.trim().length > 0
        ? diffState.patch
        : "(No diff recorded for this worker)";
    }
    drawer.appendChild(pre);
    diffSection.appendChild(drawer);
    card.appendChild(diffSection);
  }

  return card;
}

function renderRunDetail(run: RunSnapshot): HTMLElement {
  const panel = el("section", "ov-panel");

  const header = el("div", "ov-detail-header");
  const left = el("div", "ov-detail-heading");
  const titleRow = el("div", "ov-worker-title");
  titleRow.appendChild(el("h2", "ov-run-title", run.run_id));
  const copyBtn = el("button", "ov-copy-btn");
  copyBtn.type = "button";
  copyBtn.setAttribute("aria-label", `Copy run ID ${run.run_id}`);
  copyBtn.appendChild(svgIcon("copy", true));
  copyBtn.appendChild(el("span", undefined, "Copy ID"));
  copyBtn.addEventListener("click", () => {
    void (async () => {
      try {
        if (!navigator.clipboard?.writeText) {
          throw new Error("Clipboard unavailable");
        }
        await navigator.clipboard.writeText(run.run_id);
        copyBtn.replaceChildren(svgIcon("check", true), el("span", undefined, "Copied"));
        window.setTimeout(() => {
          copyBtn.replaceChildren(svgIcon("copy", true), el("span", undefined, "Copy ID"));
        }, 1500);
      } catch {
        setBanner("info", `Run ID: ${run.run_id}`, 2500);
      }
    })();
  });
  titleRow.appendChild(copyBtn);
  titleRow.appendChild(statusPill(run.status, undefined, run.status === "running"));
  left.appendChild(titleRow);

  const metaGrid = el("div", "ov-run-meta-grid");
  metaGrid.appendChild(el("span", "ov-mono", run.project_root));
  metaGrid.appendChild(el("span", "ov-num", `Started: ${formatShortTime(run.created_at_iso)}`));
  if (run.completed_at_iso) {
    metaGrid.appendChild(el("span", "ov-num", `Ended: ${formatShortTime(run.completed_at_iso)}`));
  }
  metaGrid.appendChild(
    el("span", "ov-num", `Duration: ${formatDuration(run.created_at_iso, run.completed_at_iso, run.status)}`)
  );
  metaGrid.appendChild(el("span", "ov-num", `Max steps/worker: ${run.limits.max_steps_per_worker}`));
  left.appendChild(metaGrid);
  header.appendChild(left);

  const actions = el("div", "ov-detail-actions");
  if (run.status === "running") {
    const cancelBtn = el("button", "ov-btn-danger");
    cancelBtn.type = "button";
    cancelBtn.disabled = state.cancelling;
    cancelBtn.appendChild(svgIcon("stop", true));
    cancelBtn.appendChild(el("span", undefined, state.cancelling ? "Cancelling..." : "Cancel Run"));
    cancelBtn.addEventListener("click", () => {
      void handleCancelRun(run);
    });
    actions.appendChild(cancelBtn);
  }

  const rollbackDisabled =
    state.rollingBack || run.status === "running" || run.status === "rolled_back";
  if (state.confirmingRollback && !rollbackDisabled) {
    const confirmWrap = el("div", "ov-inline-confirm");
    const confirmBtn = el("button", "ov-btn-danger");
    confirmBtn.type = "button";
    confirmBtn.appendChild(svgIcon("undo", true));
    confirmBtn.appendChild(el("span", undefined, "Confirm Rollback"));
    confirmBtn.addEventListener("click", () => {
      void handleRollbackRun(run);
    });
    confirmWrap.appendChild(confirmBtn);

    const cancelRbBtn = el("button", "ov-btn-ghost", "Cancel");
    cancelRbBtn.type = "button";
    cancelRbBtn.addEventListener("click", () => {
      state.confirmingRollback = false;
      renderRunsPanel();
    });
    confirmWrap.appendChild(cancelRbBtn);
    actions.appendChild(confirmWrap);
  } else {
    const rollbackBtn = el("button", "ov-btn-danger");
    rollbackBtn.type = "button";
    rollbackBtn.disabled = rollbackDisabled;
    rollbackBtn.title =
      run.status === "running"
        ? "Cannot rollback while run is running"
        : run.status === "rolled_back"
          ? "Run is already rolled back"
          : "Revert files touched by this run";
    rollbackBtn.appendChild(svgIcon("undo", true));
    rollbackBtn.appendChild(
      el(
        "span",
        undefined,
        state.rollingBack
          ? "Rolling back..."
          : run.status === "rolled_back"
            ? "Rolled Back"
            : "Rollback"
      )
    );
    rollbackBtn.addEventListener("click", () => {
      state.confirmingRollback = true;
      renderRunsPanel();
    });
    actions.appendChild(rollbackBtn);
  }

  if (state.canSendMessage) {
    const collectBtn = el("button", "ov-btn-primary");
    collectBtn.type = "button";
    collectBtn.disabled = state.sendingMessage;
    collectBtn.appendChild(svgIcon("chat", true));
    collectBtn.appendChild(
      el("span", undefined, state.sendingMessage ? "Sending..." : "Collect Results in Chat")
    );
    collectBtn.addEventListener("click", () => {
      void handleCollectResultsInChat(run);
    });
    actions.appendChild(collectBtn);
  }

  header.appendChild(actions);
  panel.appendChild(header);

  const workerTokensSum = run.workers.reduce((acc, w) => {
    const t =
      typeof w.total_tokens === "number"
        ? w.total_tokens
        : (w.prompt_tokens ?? 0) + (w.completion_tokens ?? 0);
    return acc + t;
  }, 0);
  const runTokens =
    typeof run.total_tokens === "number"
      ? run.total_tokens
      : typeof run.prompt_tokens === "number" || typeof run.completion_tokens === "number"
        ? (run.prompt_tokens ?? 0) + (run.completion_tokens ?? 0)
        : workerTokensSum;
  const completedWorkersCount = run.workers.filter((w) => w.status === "completed").length;

  const summaryStrip = el("div", "ov-run-summary-strip");
  summaryStrip.appendChild(el("span", "ov-num", `Total Spend: ${formatUsd(run.spend_usd)}`));
  summaryStrip.appendChild(el("span", "ov-num", `Total Tokens: ${runTokens.toLocaleString()}`));
  summaryStrip.appendChild(
    el("span", "ov-num", `Workers: ${completedWorkersCount}/${run.workers.length} completed`)
  );
  panel.appendChild(summaryStrip);

  panel.appendChild(renderBudgetBar(run.spend_usd, run.limits.max_run_usd, "Total run spend"));

  const workersWrap = el("div", "ov-workers-grid");
  if (run.workers.length === 0) {
    workersWrap.appendChild(el("div", "ov-empty", "No workers recorded for this run."));
  } else {
    for (const worker of run.workers) {
      workersWrap.appendChild(
        renderWorkerCard(worker, run.limits.max_run_usd, run.workers.length > 1, run)
      );
    }
  }
  panel.appendChild(workersWrap);
  return panel;
}

function ensureRunsLayout(): void {
  if (!runsPanelEl || runsLayoutMounted) return;

  const layout = el("div", "ov-layout");
  const sidebar = el("aside", "ov-panel ov-sidebar-panel");

  const sideHead = el("div", "ov-sidebar-head");
  runsCountTitleEl = el("h2", "ov-section-title ov-num", "Runs (0)");
  sideHead.appendChild(runsCountTitleEl);

  runsProjectSelectEl = el("select", "ov-select ov-project-select");
  runsProjectSelectEl.setAttribute("aria-label", "Filter runs by project");
  runsProjectSelectEl.addEventListener("change", () => {
    state.projectFilter = runsProjectSelectEl?.value || null;
    void refreshAll(false);
  });
  sideHead.appendChild(runsProjectSelectEl);
  sidebar.appendChild(sideHead);

  const searchWrap = el("div", "ov-search-wrap");
  const searchIcon = el("span", "ov-search-icon");
  searchIcon.appendChild(svgIcon("search", true));
  searchWrap.appendChild(searchIcon);

  runsSearchInputEl = el("input", "ov-input ov-search-input");
  runsSearchInputEl.type = "search";
  runsSearchInputEl.placeholder = "Filter runs by project, ID, or status...";
  runsSearchInputEl.setAttribute("aria-label", "Filter runs");
  runsSearchInputEl.value = runSearchQuery;
  runsSearchInputEl.addEventListener("input", () => {
    runSearchQuery = runsSearchInputEl?.value ?? "";
    renderRunsList();
  });
  searchWrap.appendChild(runsSearchInputEl);
  sidebar.appendChild(searchWrap);

  runsListEl = el("div", "ov-run-list");
  sidebar.appendChild(runsListEl);
  layout.appendChild(sidebar);

  runsDetailSlotEl = el("div");
  layout.appendChild(runsDetailSlotEl);

  runsPanelEl.replaceChildren(layout);
  runsLayoutMounted = true;
}

function renderRunsList(): void {
  const prevListScroll = runsListEl?.scrollTop ?? 0;
  const q = runSearchQuery.trim().toLowerCase();
  const filteredRuns = q
    ? state.runs.filter(
        (r) =>
          r.run_id.toLowerCase().includes(q) ||
          r.project_root.toLowerCase().includes(q) ||
          pathBasename(r.project_root).toLowerCase().includes(q) ||
          r.status.toLowerCase().includes(q) ||
          formatStatusLabel(r.status).toLowerCase().includes(q)
      )
    : state.runs;

  if (runsCountTitleEl) {
    if (q) {
      runsCountTitleEl.textContent = `Runs (${filteredRuns.length}/${state.runs.length})`;
    } else if (state.runsTotalCount !== null && state.runsTotalCount > state.runs.length) {
      runsCountTitleEl.textContent = `Runs (${state.runs.length}/${state.runsTotalCount})`;
    } else {
      runsCountTitleEl.textContent = `Runs (${state.runs.length})`;
    }
  }

  if (!runsListEl) return;
  if (filteredRuns.length === 0) {
    runsListEl.replaceChildren(
      el(
        "div",
        "ov-empty",
        state.runs.length === 0 ? "No Overseer runs recorded yet." : "No runs match your filter."
      )
    );
    return;
  }

  const items: HTMLElement[] = [];
  for (const run of filteredRuns) {
    const isSelected =
      state.selectedKey?.run_id === run.run_id &&
      state.selectedKey?.project_root === run.project_root;
    const item = el(
      "button",
      isSelected ? "ov-run-item ov-run-item-selected" : "ov-run-item"
    );
    item.type = "button";
    item.setAttribute("aria-pressed", isSelected ? "true" : "false");
    item.title = `${run.run_id}\n${run.project_root}`;

    const row1 = el("div", "ov-run-item-head");
    row1.appendChild(el("span", "ov-run-item-id", pathBasename(run.project_root)));
    row1.appendChild(
      el("span", "ov-muted ov-num ov-run-item-time", formatShortTime(run.created_at_iso))
    );
    item.appendChild(row1);

    const row2 = el("div", "ov-run-item-sub");
    row2.appendChild(statusPill(run.status, undefined, run.status === "running"));
    row2.appendChild(
      el(
        "span",
        "ov-pill-truncate ov-num ov-mono",
        `#${shortRunHash(run.run_id)} · ${run.completed_workers}/${run.worker_count} · ${formatUsd(run.spend_usd)}`
      )
    );
    item.appendChild(row2);

    item.addEventListener("click", () => {
      void handleSelectRun(run);
    });
    items.push(item);
  }
  if (state.runsHasMore) {
    const loadMoreBtn = el(
      "button",
      "ov-load-more-btn",
      state.loadingMoreRuns ? "Loading..." : "Load More"
    );
    loadMoreBtn.type = "button";
    loadMoreBtn.disabled = state.loadingMoreRuns;
    loadMoreBtn.addEventListener("click", () => {
      void handleLoadMoreRuns();
    });
    items.push(loadMoreBtn);
  }
  runsListEl.replaceChildren(...items);
  if (prevListScroll > 0) {
    runsListEl.scrollTop = prevListScroll;
  }
}

function renderRunsPanel(): void {
  if (!runsPanelEl) return;
  const sig = buildRenderSignature({
    connected: state.connected,
    loading: state.loading,
    refreshing: state.refreshing,
    cancelling: state.cancelling,
    rollingBack: state.rollingBack,
    confirmingRollback: state.confirmingRollback,
    sendingMessage: state.sendingMessage,
    auxPaneSupported: state.auxPaneSupported,
    canSendMessage: state.canSendMessage,
    projectFilter: state.projectFilter,
    activeCount: state.activeCount,
    runsHasMore: state.runsHasMore,
    loadingMoreRuns: state.loadingMoreRuns,
    workerDiffsKey: getWorkerDiffsSignature(),
    banner: state.banner,
    selectedKey: state.selectedKey,
    runs: state.runs,
    selectedRun: state.selectedRun,
  });
  if (sig === lastRunsSignature && runsPanelEl.childElementCount > 0) {
    return;
  }
  lastRunsSignature = sig;

  if (state.loading && state.runs.length === 0 && !state.selectedRun) {
    runsLayoutMounted = false;
    runsCountTitleEl = null;
    runsProjectSelectEl = null;
    runsSearchInputEl = null;
    runsListEl = null;
    runsDetailSlotEl = null;
    lastProjectOptionsKey = "";
    runsPanelEl.replaceChildren(el("div", "ov-panel ov-empty", "Loading Overseer runs..."));
    return;
  }

  ensureRunsLayout();

  const projectSet = Array.from(
    new Set([
      ...state.runs.map((r) => r.project_root),
      ...(state.projectFilter ? [state.projectFilter] : []),
    ])
  );
  if (runsProjectSelectEl) {
    const showSelect = projectSet.length > 1 || Boolean(state.projectFilter);
    runsProjectSelectEl.hidden = !showSelect;
    if (showSelect) {
      const optsKey = projectSet.join("\0");
      if (optsKey !== lastProjectOptionsKey) {
        lastProjectOptionsKey = optsKey;
        const opts: HTMLOptionElement[] = [];
        const allOpt = el("option", undefined, "All Projects");
        allOpt.value = "";
        opts.push(allOpt);
        for (const proj of projectSet) {
          const opt = el("option", undefined, pathBasename(proj));
          opt.value = proj;
          opt.title = proj;
          opts.push(opt);
        }
        runsProjectSelectEl.replaceChildren(...opts);
      }
      runsProjectSelectEl.value = state.projectFilter ?? "";
    }
  }

  renderRunsList();

  if (runsDetailSlotEl) {
    if (state.selectedRun) {
      runsDetailSlotEl.replaceChildren(renderRunDetail(state.selectedRun));
    } else {
      runsDetailSlotEl.replaceChildren(
        el("section", "ov-panel ov-empty", "Select a run to inspect worker progress and touched files.")
      );
    }
  }
}

// ============================================================================
// Tab 2: Models & Routing View
// ============================================================================

function syncConfigStatusUi(): void {
  if (configStatusPill) {
    let pillClass = "ov-pill";
    let label = "Loading...";
    let pulse = false;
    if (savingConfig) {
      pillClass = "ov-pill ov-pill-running";
      label = "Saving...";
      pulse = true;
    } else if (configDirty) {
      pillClass = "ov-pill ov-pill-escalated";
      label = "Unsaved";
    } else if (configLoaded) {
      pillClass = "ov-pill ov-pill-completed";
      label = "Synced";
    }
    configStatusPill.className = pillClass;
    const dot = el("span", pulse ? "ov-status-dot ov-dot-pulse" : "ov-status-dot");
    dot.setAttribute("aria-hidden", "true");
    configStatusPill.replaceChildren(dot, el("span", undefined, label));
  }
  if (configSaveBtn) {
    configSaveBtn.disabled = !state.connected || savingConfig || !configDraft;
    configSaveBtn.textContent = savingConfig ? "Saving..." : "Save Settings";
  }
  if (configResetBtn) {
    configResetBtn.disabled = savingConfig || !savedConfig;
  }
  syncTabButtons();
}

function syncRoutingOpenStateFromDraft(draft: ConfigFormDraft): void {
  for (const tier of ["free", "cheap", "heavy"] as const) {
    routingDetailsOpenState[tier] = hasCustomRouting(draft.tiers[tier].routing);
  }
}

function applyLoadedConfig(cfg: OverseerConfigSnapshot): void {
  savedConfig = cfg;
  configLoaded = true;
  if (!configDirty || !configDraft) {
    configDraft = configToDraft(cfg);
    syncRoutingOpenStateFromDraft(configDraft);
    configDirty = false;
    renderModelsPanel();
  } else {
    syncConfigStatusUi();
  }
}

function syncModelDatalists(): void {
  const allList = document.getElementById("ov-models-all");
  const freeList = document.getElementById("ov-models-free");
  if (!allList && !freeList) return;
  const allOpts: HTMLOptionElement[] = [];
  const freeOpts: HTMLOptionElement[] = [];
  for (const m of modelsCatalog) {
    const label = m.name ? `${m.name} (${m.id})` : m.id;
    if (allList) {
      const opt = el("option");
      opt.value = m.id;
      opt.label = label;
      allOpts.push(opt);
    }
    if (freeList && (m.is_free || m.id.endsWith(":free"))) {
      const opt = el("option");
      opt.value = m.id;
      opt.label = label;
      freeOpts.push(opt);
    }
  }
  if (allList) allList.replaceChildren(...allOpts);
  if (freeList) freeList.replaceChildren(...freeOpts);
}

async function loadModelsCatalog(): Promise<void> {
  if (!state.connected || modelsCatalogLoading || modelsCatalogLoaded || standaloneMode) return;
  modelsCatalogLoading = true;
  try {
    const res = await callServerJson<ListModelsPayload>("app_list_models", {
      free_only: false,
      tools_only: true,
      limit: 200,
    });
    modelsCatalog = Array.isArray(res.models) ? res.models : [];
    modelsCatalogLoaded = true;
    syncModelDatalists();
  } catch {
    // Autocomplete is best-effort; keep Models & Routing usable if catalog fetch fails
  } finally {
    modelsCatalogLoading = false;
  }
}

async function loadConfig(): Promise<void> {
  if (!state.connected) return;
  try {
    const cfg = await callServerJson<OverseerConfigSnapshot>("app_get_config", {});
    applyLoadedConfig(cfg);
  } catch (err) {
    if (!configLoaded) {
      const msg = err instanceof Error ? err.message : String(err);
      setBanner("error", `Could not load config: ${msg}`);
    }
  }
}

function syncFieldAria(control: HTMLInputElement | HTMLSelectElement, forceInvalidCheck = false): void {
  const valid = control.checkValidity();
  const showInvalid = !valid && (forceInvalidCheck || control.matches(":user-invalid") || control.classList.contains("ov-user-invalid"));
  control.classList.toggle("ov-user-invalid", showInvalid);
  control.setAttribute("aria-invalid", showInvalid ? "true" : "false");

  const errorId = control.dataset.error;
  if (!errorId) return;
  const ids = (control.getAttribute("aria-describedby") || "")
    .split(/\s+/)
    .filter((id) => id && id !== errorId);
  if (showInvalid) ids.push(errorId);
  if (ids.length > 0) {
    control.setAttribute("aria-describedby", ids.join(" "));
  } else {
    control.removeAttribute("aria-describedby");
  }
}

async function handleSaveConfig(): Promise<void> {
  if (!state.connected || savingConfig || !configDraft) return;
  if (modelsPanelEl) {
    const controls = Array.from(
      modelsPanelEl.querySelectorAll<HTMLInputElement | HTMLSelectElement>("input[data-error], select[data-error]")
    );
    let firstInvalid: HTMLInputElement | HTMLSelectElement | null = null;
    for (const ctrl of controls) {
      if (!ctrl.checkValidity()) {
        syncFieldAria(ctrl, true);
        if (!firstInvalid) firstInvalid = ctrl;
      }
    }
    if (firstInvalid) {
      firstInvalid.focus();
      const builtCheck = buildConfigUpdatePayload(configDraft, savedConfig);
      setBanner(
        "error",
        !builtCheck.ok
          ? builtCheck.error
          : firstInvalid.validationMessage || "Fix invalid fields before saving."
      );
      return;
    }
  }
  const built = buildConfigUpdatePayload(configDraft, savedConfig);
  if (!built.ok) {
    setBanner("error", built.error);
    return;
  }
  if (standaloneMode) {
    setBanner("info", "Standalone preview mode — changes are not written to disk.", 3000);
    return;
  }

  savingConfig = true;
  syncConfigStatusUi();
  try {
    const updated = await callServerJson<OverseerConfigSnapshot>("app_update_config", built.payload);
    savedConfig = updated;
    configLoaded = true;
    configDirty = false;
    configDraft = configToDraft(updated);
    renderModelsPanel();
    setBanner("success", "Model tiers, limits, and provider routing saved.", 3500);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setBanner("error", `Config update rejected: ${msg}`);
  } finally {
    savingConfig = false;
    syncConfigStatusUi();
  }
}

function buildLabeledField(
  labelText: string,
  control: HTMLInputElement | HTMLSelectElement,
  options?: { className?: string; hint?: string; errorMessage?: string; envOverride?: string }
): HTMLElement {
  const wrap = el("div", options?.className ? `ov-field ${options.className}` : "ov-field");
  const id = nextFieldId("field");
  control.id = id;
  const lbl = el("label", "ov-field-label", labelText);
  lbl.htmlFor = id;
  if (options?.envOverride) {
    const labelRow = el("div", "ov-field-label-row");
    labelRow.appendChild(lbl);
    const lockBadge = el("span", "ov-env-lock");
    lockBadge.title = `Overridden by environment variable ${options.envOverride}`;
    lockBadge.appendChild(svgIcon("lock", true));
    lockBadge.appendChild(el("span", undefined, "ENV"));
    labelRow.appendChild(lockBadge);
    wrap.appendChild(labelRow);
    control.title = `Overridden by environment variable ${options.envOverride}`;
  } else {
    wrap.appendChild(lbl);
  }
  if (options?.hint) {
    const hintId = `${id}-hint`;
    const hintEl = el("span", "ov-hint", options.hint);
    hintEl.id = hintId;
    control.setAttribute("aria-describedby", hintId);
    wrap.appendChild(hintEl);
  }
  wrap.appendChild(control);
  if (options?.errorMessage) {
    const errorId = `${id}-error`;
    control.dataset.error = errorId;
    control.setAttribute("aria-invalid", "false");
    const errEl = el("div", "ov-field-error");
    errEl.id = errorId;
    errEl.appendChild(svgIcon("alert", true));
    errEl.appendChild(el("span", undefined, options.errorMessage));
    wrap.appendChild(errEl);

    control.addEventListener("blur", () => {
      syncFieldAria(control, !control.checkValidity());
    });
    control.addEventListener("input", () => {
      if (control.getAttribute("aria-invalid") === "true" || control.classList.contains("ov-user-invalid")) {
        syncFieldAria(control, false);
      }
    });
  }
  return wrap;
}

function hasCustomRouting(rule: RoutingRuleDraft): boolean {
  return Boolean(
    rule.sort ||
      rule.allow_fallbacks !== "default" ||
      rule.zdr !== "default" ||
      rule.order.trim() ||
      rule.only.trim() ||
      rule.quantizations.trim()
  );
}

function buildRoutingFields(
  rule: RoutingRuleDraft,
  onDirty: () => void
): HTMLElement {
  const grid = el("div", "ov-routing-grid");

  const sortSel = el("select", "ov-select");
  for (const [val, label] of [
    ["", "Default"],
    ["price", "Lowest Price"],
    ["throughput", "Highest Throughput"],
    ["latency", "Lowest Latency"],
  ] as const) {
    const opt = el("option", undefined, label);
    opt.value = val;
    sortSel.appendChild(opt);
  }
  sortSel.value = rule.sort;
  sortSel.addEventListener("change", () => {
    rule.sort = sortSel.value as "" | ProviderSortPreset;
    onDirty();
  });
  grid.appendChild(buildLabeledField("Provider Sort", sortSel));

  const fbSel = el("select", "ov-select");
  for (const [val, label] of [
    ["default", "Default"],
    ["true", "Allow Fallbacks"],
    ["false", "Strict (No Fallbacks)"],
  ] as const) {
    const opt = el("option", undefined, label);
    opt.value = val;
    fbSel.appendChild(opt);
  }
  fbSel.value = rule.allow_fallbacks;
  fbSel.addEventListener("change", () => {
    rule.allow_fallbacks = fbSel.value as RoutingRuleDraft["allow_fallbacks"];
    onDirty();
  });
  grid.appendChild(buildLabeledField("Provider Fallbacks", fbSel));

  const orderInput = el("input", "ov-input ov-mono");
  orderInput.type = "text";
  orderInput.placeholder = "e.g. Anthropic, Google";
  orderInput.value = rule.order;
  orderInput.addEventListener("input", () => {
    rule.order = orderInput.value;
    onDirty();
  });
  grid.appendChild(buildLabeledField("Order (comma-separated)", orderInput));

  const onlyInput = el("input", "ov-input ov-mono");
  onlyInput.type = "text";
  onlyInput.placeholder = "e.g. Anthropic";
  onlyInput.value = rule.only;
  onlyInput.addEventListener("input", () => {
    rule.only = onlyInput.value;
    onDirty();
  });
  grid.appendChild(buildLabeledField("Only (comma-separated)", onlyInput));

  const zdrSel = el("select", "ov-select");
  for (const [val, label] of [
    ["default", "Default"],
    ["true", "Require ZDR"],
    ["false", "Allow Non-ZDR"],
  ] as const) {
    const opt = el("option", undefined, label);
    opt.value = val;
    zdrSel.appendChild(opt);
  }
  zdrSel.value = rule.zdr;
  zdrSel.addEventListener("change", () => {
    rule.zdr = zdrSel.value as RoutingRuleDraft["zdr"];
    onDirty();
  });
  grid.appendChild(buildLabeledField("Zero Data Retention (ZDR)", zdrSel));

  const quantInput = el("input", "ov-input ov-mono");
  quantInput.type = "text";
  quantInput.placeholder = "e.g. fp8, bf16";
  quantInput.value = rule.quantizations;
  quantInput.addEventListener("input", () => {
    rule.quantizations = quantInput.value;
    onDirty();
  });
  grid.appendChild(buildLabeledField("Quantizations (comma-separated)", quantInput));

  return grid;
}

function renderModelsPanel(): void {
  if (!modelsPanelEl) return;
  if (!configDraft) {
    modelsPanelEl.replaceChildren(el("div", "ov-panel ov-empty", "Loading model configuration..."));
    return;
  }

  const draft = configDraft;
  const markDirty = (): void => {
    configDirty = true;
    syncConfigStatusUi();
  };

  const wrap = el("div", "ov-config-stack");

  // Header & Actions
  const headerCard = el("section", "ov-panel");
  const headRow = el("div", "ov-section-head");
  const titleGroup = el("div", "ov-brand");
  titleGroup.appendChild(el("h2", "ov-section-title", "Model Tiers & Provider Routing"));
  configStatusPill = el("span", "ov-pill");
  titleGroup.appendChild(configStatusPill);
  headRow.appendChild(titleGroup);

  const actionGroup = el("div", "ov-detail-actions");
  configResetBtn = el("button", undefined, "Reset");
  configResetBtn.type = "button";
  configResetBtn.addEventListener("click", () => {
    if (!savedConfig) return;
    configDraft = configToDraft(savedConfig);
    syncRoutingOpenStateFromDraft(configDraft);
    configDirty = false;
    renderModelsPanel();
  });
  actionGroup.appendChild(configResetBtn);

  configSaveBtn = el("button", "ov-btn-primary", "Save Settings");
  configSaveBtn.type = "button";
  configSaveBtn.addEventListener("click", () => {
    void handleSaveConfig();
  });
  actionGroup.appendChild(configSaveBtn);
  headRow.appendChild(actionGroup);
  headerCard.appendChild(headRow);

  if (savedConfig) {
    const keysBar = el("div", "ov-keys-bar");
    keysBar.appendChild(el("span", "ov-muted", "API Keys (.env):"));
    const keys: Array<[string, boolean]> = [
      ["Free Key", savedConfig.api_keys_configured.free],
      ["Cheap Key", savedConfig.api_keys_configured.cheap],
      ["Heavy Key", savedConfig.api_keys_configured.heavy],
      ["Shared Key", savedConfig.api_keys_configured.shared],
    ];
    for (const [label, ok] of keys) {
      const pill = el("span", ok ? "ov-pill ov-pill-completed" : "ov-pill");
      pill.appendChild(svgIcon(ok ? "check" : "minus", true));
      pill.appendChild(el("span", undefined, `${label}: ${ok ? "Configured" : "Unset"}`));
      keysBar.appendChild(pill);
    }
    headerCard.appendChild(keysBar);
  }

  const settingsRow = el("div", "ov-config-settings-row");
  const workersInput = el("input", "ov-input ov-mono ov-num");
  workersInput.type = "number";
  workersInput.min = "1";
  workersInput.max = "10";
  workersInput.step = "1";
  workersInput.required = true;
  workersInput.value = draft.max_concurrent_workers;
  const syncWorkersValidity = (): void => {
    const raw = workersInput.value.trim();
    const n = Number(raw);
    workersInput.setCustomValidity(
      raw && Number.isInteger(n) && n >= 1 && n <= 10
        ? ""
        : "Enter an integer between 1 and 10."
    );
  };
  syncWorkersValidity();
  workersInput.addEventListener("input", () => {
    draft.max_concurrent_workers = workersInput.value;
    syncWorkersValidity();
    markDirty();
  });
  settingsRow.appendChild(
    buildLabeledField("Max Concurrent Workers (1–10)", workersInput, {
      className: "ov-workers-limit-field",
      hint: "Parallel workers per run (default 3).",
      errorMessage: "Enter an integer between 1 and 10.",
      envOverride: savedConfig?.env_overrides?.max_concurrent_workers,
    })
  );
  headerCard.appendChild(settingsRow);
  wrap.appendChild(headerCard);

  // 3 Tier Cards
  const tierMeta: Array<{
    tier: TierName;
    title: string;
    badge: string;
    primaryPlaceholder: string;
    hasLimit: boolean;
  }> = [
    {
      tier: "free",
      title: "Free Tier",
      badge: "Must end with :free",
      primaryPlaceholder: "particle-ai/particle-1:free",
      hasLimit: false,
    },
    {
      tier: "cheap",
      title: "Cheap Tier",
      badge: "Default cap $0.25 / run",
      primaryPlaceholder: "deepseek/deepseek-v4.1-flash-1017",
      hasLimit: true,
    },
    {
      tier: "heavy",
      title: "Heavy Tier",
      badge: "Optional · Default cap $2.00 / run",
      primaryPlaceholder: "Unconfigured (e.g. anthropic/claude-sonnet-4.6)",
      hasLimit: true,
    },
  ];

  const tiersGrid = el("div", "ov-tiers-grid");
  for (const meta of tierMeta) {
    const td = draft.tiers[meta.tier];
    const cardClass = `ov-panel ov-tier-card ov-tier-card-${meta.tier}`;
    const card = el("section", cardClass);

    const cardHead = el("div", "ov-section-head");
    cardHead.appendChild(el("h3", "ov-tier-title", meta.title));
    cardHead.appendChild(el("span", "ov-pill ov-num", meta.badge));
    card.appendChild(cardHead);

    const fields = el(
      "div",
      meta.hasLimit ? "ov-tier-fields ov-tier-fields-with-limit" : "ov-tier-fields"
    );

    const primInput = el("input", "ov-input ov-mono");
    primInput.type = "text";
    primInput.placeholder = meta.primaryPlaceholder;
    primInput.value = td.primary;
    primInput.setAttribute("list", meta.tier === "free" ? "ov-models-free" : "ov-models-all");
    if (meta.tier === "free") {
      primInput.required = true;
      primInput.pattern = "^\\s*\\S+:free\\s*$";
    } else if (meta.tier === "cheap") {
      primInput.required = true;
      primInput.pattern = ".*\\S.*";
    }
    primInput.addEventListener("input", () => {
      td.primary = primInput.value;
      markDirty();
    });
    const primError =
      meta.tier === "free"
        ? "Must end with :free (e.g. particle-ai/particle-1:free)."
        : meta.tier === "cheap"
          ? "Enter a primary model slug (e.g. deepseek/deepseek-v4.1-flash-1017)."
          : undefined;
    fields.appendChild(
      buildLabeledField("Primary Model", primInput, {
        errorMessage: primError,
        envOverride: savedConfig?.env_overrides?.[`tiers.${meta.tier}.primary`],
      })
    );

    if (meta.hasLimit) {
      const capInput = el("input", "ov-input ov-mono ov-num");
      capInput.type = "number";
      capInput.step = "any";
      capInput.min = "0";
      capInput.max = "50";
      capInput.value = td.max_run_usd;
      const syncCapValidity = (): void => {
        const trimmed = capInput.value.trim();
        if (!trimmed) {
          capInput.setCustomValidity("");
          return;
        }
        const n = Number(trimmed);
        capInput.setCustomValidity(
          Number.isFinite(n) && n > 0 && n <= 50 ? "" : "Enter a limit above $0 and up to $50.00."
        );
      };
      syncCapValidity();
      capInput.addEventListener("input", () => {
        td.max_run_usd = capInput.value;
        syncCapValidity();
        markDirty();
      });
      fields.appendChild(
        buildLabeledField("Per-Run Limit (USD)", capInput, {
          errorMessage: "Enter a limit above $0 and up to $50.00.",
        })
      );
    }

    const fbInput = el("input", "ov-input ov-mono");
    fbInput.type = "text";
    fbInput.placeholder =
      meta.tier === "free"
        ? "nvidia/nemotron-3-ultra-550b-a55b:free, google/gemma-4-31b-it:free"
        : "Optional fallback model slugs...";
    fbInput.value = td.fallbacks;
    fbInput.setAttribute("list", meta.tier === "free" ? "ov-models-free" : "ov-models-all");
    fbInput.addEventListener("input", () => {
      td.fallbacks = fbInput.value;
      markDirty();
    });
    fields.appendChild(
      buildLabeledField("Fallback Models (comma-separated)", fbInput, {
        className: "ov-field-span",
        envOverride: savedConfig?.env_overrides?.[`tiers.${meta.tier}.fallbacks`],
      })
    );

    card.appendChild(fields);

    const routingDetails = el("details", "ov-routing-details");
    routingDetails.open = routingDetailsOpenState[meta.tier];
    routingDetails.addEventListener("toggle", () => {
      routingDetailsOpenState[meta.tier] = routingDetails.open;
    });

    const routingSummary = el("summary", "ov-routing-summary");
    routingSummary.appendChild(el("span", "ov-routing-summary-title", "Provider Routing"));
    const routingBadge = el("span");
    const syncRoutingBadge = (): void => {
      const custom = hasCustomRouting(td.routing);
      routingBadge.className = custom ? "ov-pill ov-pill-running" : "ov-pill";
      routingBadge.textContent = custom ? "Custom" : "Default";
    };
    syncRoutingBadge();
    routingSummary.appendChild(routingBadge);
    routingDetails.appendChild(routingSummary);

    const routingBody = el("div", "ov-routing-body");
    routingBody.appendChild(
      buildRoutingFields(td.routing, () => {
        syncRoutingBadge();
        markDirty();
      })
    );
    routingDetails.appendChild(routingBody);
    card.appendChild(routingDetails);

    tiersGrid.appendChild(card);
  }
  wrap.appendChild(tiersGrid);

  // Per-Model Provider Routing Overrides
  const overridesCard = el("section", "ov-panel ov-tier-card ov-overrides-card");
  const ovHead = el("div", "ov-section-head");
  ovHead.appendChild(el("h3", "ov-tier-title", "Per-Model Provider Overrides"));
  const addOvBtn = el("button");
  addOvBtn.type = "button";
  addOvBtn.appendChild(svgIcon("plus", true));
  addOvBtn.appendChild(el("span", undefined, "Add Override"));
  addOvBtn.addEventListener("click", () => {
    draft.modelOverrides.push({ model: "", routing: emptyRoutingDraft() });
    markDirty();
    renderModelsPanel();
  });
  ovHead.appendChild(addOvBtn);
  overridesCard.appendChild(ovHead);

  if (draft.modelOverrides.length === 0) {
    overridesCard.appendChild(
      el(
        "div",
        "ov-muted",
        "No model-specific provider overrides configured. Overrides win over tier routing."
      )
    );
  } else {
    const rowsWrap = el("div", "ov-overrides-list");
    draft.modelOverrides.forEach((row, index) => {
      const rowCard = el("div", "ov-subpanel");
      const topRow = el("div", "ov-override-head");

      const slugInput = el("input", "ov-input ov-mono");
      slugInput.type = "text";
      slugInput.placeholder = "e.g. anthropic/claude-sonnet-4.6";
      slugInput.value = row.model;
      slugInput.setAttribute("list", "ov-models-all");
      slugInput.addEventListener("input", () => {
        row.model = slugInput.value;
        markDirty();
      });
      topRow.appendChild(buildLabeledField("Model Slug", slugInput, { className: "ov-field-grow" }));

      const rmBtn = el("button", "ov-btn-danger");
      rmBtn.type = "button";
      rmBtn.setAttribute("aria-label", `Remove override ${row.model || `#${index + 1}`}`);
      rmBtn.appendChild(svgIcon("trash", true));
      rmBtn.appendChild(el("span", undefined, "Remove"));
      rmBtn.addEventListener("click", () => {
        draft.modelOverrides.splice(index, 1);
        markDirty();
        renderModelsPanel();
      });
      topRow.appendChild(rmBtn);
      rowCard.appendChild(topRow);

      rowCard.appendChild(buildRoutingFields(row.routing, markDirty));
      rowsWrap.appendChild(rowCard);
    });
    overridesCard.appendChild(rowsWrap);
  }

  if (savedConfig?.config_path) {
    const foot = el("div", "ov-config-footer ov-mono");
    foot.textContent = `Config file: ${savedConfig.config_path} (env vars override config.json; API keys stay in .env)`;
    overridesCard.appendChild(foot);
  }
  wrap.appendChild(overridesCard);

  const allModelsList = el("datalist");
  allModelsList.id = "ov-models-all";
  const freeModelsList = el("datalist");
  freeModelsList.id = "ov-models-free";
  wrap.appendChild(allModelsList);
  wrap.appendChild(freeModelsList);

  modelsPanelEl.replaceChildren(wrap);
  syncModelDatalists();
  syncConfigStatusUi();
}

// ============================================================================
// Tab 3: Custom Instructions View
// ============================================================================

function syncInstructionsUi(): void {
  const currentLen = instructionsTextarea ? instructionsTextarea.value.length : savedInstructions.length;
  if (instructionsCountEl) {
    instructionsCountEl.textContent = `${currentLen.toLocaleString()} / ${MAX_CUSTOM_INSTRUCTIONS_CHARS.toLocaleString()} chars`;
  }
  if (instructionsBarFillEl) {
    const pct = Math.min(100, Math.max(0, (currentLen / MAX_CUSTOM_INSTRUCTIONS_CHARS) * 100));
    instructionsBarFillEl.style.width = `${pct}%`;
    instructionsBarFillEl.className =
      pct >= 95
        ? "ov-mini-bar-fill ov-mini-bar-danger"
        : pct >= 80
          ? "ov-mini-bar-fill ov-mini-bar-warn"
          : "ov-mini-bar-fill";
  }
  if (instructionsStatusPill) {
    let pillClass = "ov-pill";
    let label = "Empty";
    let pulse = false;
    if (instructionsLoadError !== null) {
      pillClass = "ov-pill ov-pill-failed";
      label = "Load Failed";
    } else if (savingInstructions) {
      pillClass = "ov-pill ov-pill-running";
      label = "Saving...";
      pulse = true;
    } else if (instructionsDirty) {
      pillClass = "ov-pill ov-pill-escalated";
      label = "Unsaved";
    } else if (savedInstructions) {
      pillClass = "ov-pill ov-pill-completed";
      label = "Saved";
    }
    instructionsStatusPill.className = pillClass;
    const dot = el("span", pulse ? "ov-status-dot ov-dot-pulse" : "ov-status-dot");
    dot.setAttribute("aria-hidden", "true");
    instructionsStatusPill.replaceChildren(dot, el("span", undefined, label));
  }
  if (instructionsErrorBannerEl) {
    if (instructionsLoadError !== null) {
      instructionsErrorBannerEl.hidden = false;
      instructionsErrorBannerEl.replaceChildren(
        svgIcon("alert", true),
        el("span", undefined, `Failed to load custom instructions: ${instructionsLoadError}`)
      );
    } else {
      instructionsErrorBannerEl.hidden = true;
      instructionsErrorBannerEl.replaceChildren();
    }
  }
  if (instructionsTextarea) {
    instructionsTextarea.disabled = instructionsLoadError !== null;
  }
  if (instructionsSaveBtn) {
    instructionsSaveBtn.disabled =
      !state.connected || savingInstructions || instructionsLoadError !== null;
    instructionsSaveBtn.textContent = savingInstructions ? "Saving..." : "Save Instructions";
  }
  const canClear =
    state.connected &&
    !savingInstructions &&
    instructionsLoadError === null &&
    Boolean(savedInstructions || currentLen);
  if (instructionsClearBtn) {
    instructionsClearBtn.hidden = confirmingClearInstructions;
    instructionsClearBtn.disabled = !canClear;
  }
  if (instructionsConfirmClearBtn) {
    instructionsConfirmClearBtn.hidden = !confirmingClearInstructions;
    instructionsConfirmClearBtn.disabled = !canClear;
  }
  if (instructionsCancelClearBtn) {
    instructionsCancelClearBtn.hidden = !confirmingClearInstructions;
    instructionsCancelClearBtn.disabled = savingInstructions;
  }
  syncTabButtons();
}

function applyLoadedInstructions(text: string): void {
  instructionsLoadError = null;
  savedInstructions = text;
  instructionsLoaded = true;
  confirmingClearInstructions = false;
  if (instructionsTextarea && !instructionsDirty) {
    instructionsTextarea.value = text;
  }
  syncInstructionsUi();
}

async function loadCustomInstructions(): Promise<void> {
  if (!state.connected) return;
  try {
    const data = await callServerJson<CustomInstructionsPayload>("app_get_custom_instructions", {});
    applyLoadedInstructions(typeof data.custom_instructions === "string" ? data.custom_instructions : "");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    instructionsLoadError = msg;
    instructionsLoaded = false;
    confirmingClearInstructions = false;
    setBanner("error", `Could not load custom instructions: ${msg}`);
    syncInstructionsUi();
  }
}

async function saveInstructionsValue(raw: string): Promise<void> {
  if (!state.connected || savingInstructions || instructionsLoadError !== null) return;
  if (standaloneMode) {
    confirmingClearInstructions = false;
    setBanner("info", "Standalone preview mode — changes are not written to disk.", 3000);
    syncInstructionsUi();
    return;
  }
  savingInstructions = true;
  confirmingClearInstructions = false;
  syncInstructionsUi();
  try {
    const data = await callServerJson<CustomInstructionsPayload>("app_set_custom_instructions", {
      instructions: raw,
    });
    const saved = typeof data.custom_instructions === "string" ? data.custom_instructions : "";
    savedInstructions = saved;
    instructionsLoaded = true;
    instructionsDirty = false;
    if (instructionsTextarea) {
      instructionsTextarea.value = saved;
    }
    setBanner(
      "success",
      saved ? "Custom instructions saved for Overseer workers." : "Custom instructions cleared.",
      3000
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setBanner("error", `Failed to save custom instructions: ${msg}`);
  } finally {
    savingInstructions = false;
    syncInstructionsUi();
  }
}

// ============================================================================
// Persistent Shell, Tabs & Topbar
// ============================================================================

function activateTab(tab: TabId, focusBtn = false): void {
  state.activeTab = tab;
  syncTabButtons();
  const activeBtn =
    tab === "runs" ? tabRunsBtn : tab === "models" ? tabModelsBtn : tabInstrBtn;
  activeBtn?.scrollIntoView({ block: "nearest", inline: "nearest" });
  if (tab === "runs") {
    renderRunsPanel();
    if (focusBtn) tabRunsBtn?.focus();
  } else if (tab === "models") {
    if (!configLoaded && state.connected) {
      void loadConfig();
    }
    if (!modelsCatalogLoaded && state.connected) {
      void loadModelsCatalog();
    }
    if (focusBtn) tabModelsBtn?.focus();
  } else if (tab === "instructions") {
    if (!instructionsLoaded && state.connected) {
      void loadCustomInstructions();
    }
    if (focusBtn) tabInstrBtn?.focus();
  }
}

function syncTabButtons(): void {
  if (!tabRunsBtn || !tabModelsBtn || !tabInstrBtn || !runsPanelEl || !modelsPanelEl || !instructionsPanelEl) {
    return;
  }
  const setTabState = (
    btn: HTMLButtonElement,
    panel: HTMLElement,
    active: boolean,
    label: string,
    dirty = false
  ): void => {
    btn.className = "ov-tab";
    btn.setAttribute("aria-selected", active ? "true" : "false");
    btn.tabIndex = active ? 0 : -1;
    const nodes: Node[] = [el("span", undefined, label)];
    if (dirty) {
      const dot = el("span", "ov-dirty-dot");
      dot.setAttribute("aria-hidden", "true");
      nodes.push(dot, el("span", "ov-visually-hidden", " (unsaved changes)"));
    }
    btn.replaceChildren(...nodes);
    if (active) {
      panel.removeAttribute("hidden");
    } else {
      panel.setAttribute("hidden", "until-found");
    }
  };
  setTabState(tabRunsBtn, runsPanelEl, state.activeTab === "runs", "Runs");
  setTabState(
    tabModelsBtn,
    modelsPanelEl,
    state.activeTab === "models",
    "Models & Routing",
    configDirty
  );
  setTabState(
    tabInstrBtn,
    instructionsPanelEl,
    state.activeTab === "instructions",
    "Instructions",
    instructionsDirty
  );
}

function renderTopbar(): void {
  if (!topbarBrandEl || !topbarActionsEl) return;

  const brandNodes: HTMLElement[] = [];
  const mark = el("span", "ov-brand-mark");
  mark.appendChild(svgIcon("layers"));
  brandNodes.push(mark);
  brandNodes.push(el("h1", "ov-title", "Overseer Monitor"));

  const activeText = state.activeCount > 0 ? `${state.activeCount} active` : "Idle";
  const activePill = statusPill(
    state.activeCount > 0 ? "running" : "idle",
    activeText,
    state.activeCount > 0
  );
  activePill.classList.add("ov-num");
  brandNodes.push(activePill);

  if (state.projectFilter) {
    const scopePill = el("span", "ov-pill ov-pill-truncate ov-mono", `Scope: ${pathBasename(state.projectFilter)}`);
    scopePill.title = state.projectFilter;
    brandNodes.push(scopePill);
    const clearBtn = el("button", "ov-btn-ghost", "All Projects");
    clearBtn.type = "button";
    clearBtn.addEventListener("click", () => {
      state.projectFilter = null;
      void refreshAll(false);
    });
    brandNodes.push(clearBtn);
  }
  topbarBrandEl.replaceChildren(...brandNodes);

  const refreshBtn = el("button");
  refreshBtn.type = "button";
  refreshBtn.disabled = !state.connected || state.refreshing;
  refreshBtn.appendChild(svgIcon("refresh", true));
  refreshBtn.appendChild(el("span", undefined, state.refreshing ? "Refreshing..." : "Refresh"));
  refreshBtn.addEventListener("click", () => {
    void refreshAll(false);
    if (state.activeTab === "models" && !configDirty) {
      void loadConfig();
    } else if (state.activeTab === "instructions" && !instructionsDirty) {
      void loadCustomInstructions();
    }
  });
  topbarActionsEl.replaceChildren(refreshBtn);
}

function renderBanner(): void {
  if (!bannerSlotEl) return;
  if (!state.banner) {
    bannerSlotEl.removeAttribute("role");
    bannerSlotEl.removeAttribute("aria-live");
    bannerSlotEl.replaceChildren();
    return;
  }
  if (state.banner.kind === "error") {
    bannerSlotEl.setAttribute("role", "alert");
    bannerSlotEl.removeAttribute("aria-live");
  } else {
    bannerSlotEl.setAttribute("role", "status");
    bannerSlotEl.setAttribute("aria-live", "polite");
  }
  bannerSlotEl.setAttribute("aria-atomic", "true");

  const banner = el("div", `ov-banner ov-banner-${state.banner.kind}`);
  const body = el("div", "ov-banner-body");
  const iconName: IconName =
    state.banner.kind === "error" ? "alert" : state.banner.kind === "success" ? "check" : "info";
  body.appendChild(svgIcon(iconName, true));
  body.appendChild(el("span", undefined, state.banner.text));
  banner.appendChild(body);

  const dismissBtn = el("button", "ov-btn-ghost", "Dismiss");
  dismissBtn.type = "button";
  dismissBtn.addEventListener("click", () => {
    state.banner = null;
    renderBanner();
  });
  banner.appendChild(dismissBtn);
  bannerSlotEl.replaceChildren(banner);
}

function ensureShell(): boolean {
  const root = document.getElementById("app");
  if (!root) return false;
  if (shellMounted) return true;

  const shell = el("div", "ov-shell");
  const chrome = el("div", "ov-chrome");

  const topbar = el("header", "ov-topbar");
  topbarBrandEl = el("div", "ov-brand");
  topbarActionsEl = el("div", "ov-topbar-actions");
  topbar.appendChild(topbarBrandEl);
  topbar.appendChild(topbarActionsEl);
  chrome.appendChild(topbar);

  const tabsNav = el("nav", "ov-tabs-nav");
  tabsNav.setAttribute("aria-label", "Monitor views");
  const tablist = el("div", "ov-tablist");
  tablist.setAttribute("role", "tablist");

  const tabOrder: TabId[] = ["runs", "models", "instructions"];
  tablist.addEventListener("keydown", (ev) => {
    const idx = tabOrder.indexOf(state.activeTab);
    if (ev.key === "ArrowRight") {
      ev.preventDefault();
      activateTab(tabOrder[(idx + 1) % tabOrder.length], true);
    } else if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      activateTab(tabOrder[(idx - 1 + tabOrder.length) % tabOrder.length], true);
    } else if (ev.key === "Home") {
      ev.preventDefault();
      activateTab(tabOrder[0], true);
    } else if (ev.key === "End") {
      ev.preventDefault();
      activateTab(tabOrder[tabOrder.length - 1], true);
    }
  });

  tabRunsBtn = el("button", "ov-tab", "Runs");
  tabRunsBtn.type = "button";
  tabRunsBtn.id = "ov-tab-runs";
  tabRunsBtn.setAttribute("role", "tab");
  tabRunsBtn.addEventListener("click", () => activateTab("runs"));

  tabModelsBtn = el("button", "ov-tab", "Models & Routing");
  tabModelsBtn.type = "button";
  tabModelsBtn.id = "ov-tab-models";
  tabModelsBtn.setAttribute("role", "tab");
  tabModelsBtn.addEventListener("click", () => activateTab("models"));

  tabInstrBtn = el("button", "ov-tab", "Instructions");
  tabInstrBtn.type = "button";
  tabInstrBtn.id = "ov-tab-instructions";
  tabInstrBtn.setAttribute("role", "tab");
  tabInstrBtn.addEventListener("click", () => activateTab("instructions"));

  tablist.appendChild(tabRunsBtn);
  tablist.appendChild(tabModelsBtn);
  tablist.appendChild(tabInstrBtn);
  tabsNav.appendChild(tablist);
  chrome.appendChild(tabsNav);
  shell.appendChild(chrome);

  const main = el("main", "ov-main");

  bannerSlotEl = el("div", "ov-banner-slot");
  main.appendChild(bannerSlotEl);

  const panelsStage = el("div", "ov-panels-stage");

  runsPanelEl = el("div");
  runsPanelEl.id = "ov-panel-runs";
  runsPanelEl.setAttribute("role", "tabpanel");
  runsPanelEl.setAttribute("aria-labelledby", "ov-tab-runs");
  runsPanelEl.addEventListener("beforematch", () => activateTab("runs"));
  panelsStage.appendChild(runsPanelEl);

  modelsPanelEl = el("div");
  modelsPanelEl.id = "ov-panel-models";
  modelsPanelEl.setAttribute("role", "tabpanel");
  modelsPanelEl.setAttribute("aria-labelledby", "ov-tab-models");
  modelsPanelEl.setAttribute("hidden", "until-found");
  modelsPanelEl.addEventListener("beforematch", () => activateTab("models"));
  panelsStage.appendChild(modelsPanelEl);

  instructionsPanelEl = el("div");
  instructionsPanelEl.id = "ov-panel-instructions";
  instructionsPanelEl.setAttribute("role", "tabpanel");
  instructionsPanelEl.setAttribute("aria-labelledby", "ov-tab-instructions");
  instructionsPanelEl.setAttribute("hidden", "until-found");
  instructionsPanelEl.addEventListener("beforematch", () => activateTab("instructions"));

  const instrPanel = el("section", "ov-panel ov-instructions-panel");
  const instrHead = el("div", "ov-section-head");
  const instrTitleWrap = el("div", "ov-brand");
  instrTitleWrap.appendChild(el("h2", "ov-section-title", "Custom Instructions"));
  instructionsStatusPill = el("span", "ov-pill");
  instrTitleWrap.appendChild(instructionsStatusPill);
  instrHead.appendChild(instrTitleWrap);
  instructionsCountEl = el("span", "ov-muted ov-num", `0 / ${MAX_CUSTOM_INSTRUCTIONS_CHARS.toLocaleString()} chars`);
  instrHead.appendChild(instructionsCountEl);
  instrPanel.appendChild(instrHead);

  const instrBar = el("div", "ov-mini-bar");
  instructionsBarFillEl = el("div", "ov-mini-bar-fill");
  instructionsBarFillEl.style.width = "0%";
  instrBar.appendChild(instructionsBarFillEl);
  instrPanel.appendChild(instrBar);

  instructionsErrorBannerEl = el("div", "ov-block-box ov-block-danger");
  instructionsErrorBannerEl.hidden = true;
  instrPanel.appendChild(instructionsErrorBannerEl);

  const instrLabel = el("label", "ov-field-label", "Worker Prompt Rules");
  instrLabel.htmlFor = "ov-instructions-input";
  instrPanel.appendChild(instrLabel);

  const instrHint = el(
    "span",
    "ov-hint",
    "Optional rules appended to every Overseer worker prompt. Clear the text and save to disable."
  );
  instrHint.id = "ov-instructions-hint";
  instrPanel.appendChild(instrHint);

  instructionsTextarea = el("textarea", "ov-instructions-textarea");
  instructionsTextarea.id = "ov-instructions-input";
  instructionsTextarea.setAttribute("aria-describedby", "ov-instructions-hint");
  instructionsTextarea.maxLength = MAX_CUSTOM_INSTRUCTIONS_CHARS;
  instructionsTextarea.placeholder =
    "Add optional project or coding rules here, then click Save Instructions...";
  instructionsTextarea.value = savedInstructions;
  instructionsTextarea.addEventListener("input", () => {
    instructionsDirty = (instructionsTextarea?.value.trim() ?? "") !== savedInstructions;
    confirmingClearInstructions = false;
    syncInstructionsUi();
  });
  instrPanel.appendChild(instructionsTextarea);

  const instrActions = el("div", "ov-instructions-actions");
  instrActions.appendChild(el("span", "ov-hint ov-mono", "Stored locally in <data_dir>/custom_instructions.txt"));
  const instrBtnGroup = el("div", "ov-detail-actions");
  instructionsClearBtn = el("button", undefined, "Clear");
  instructionsClearBtn.type = "button";
  instructionsClearBtn.addEventListener("click", () => {
    confirmingClearInstructions = true;
    syncInstructionsUi();
  });
  instrBtnGroup.appendChild(instructionsClearBtn);

  instructionsConfirmClearBtn = el("button", "ov-btn-danger", "Confirm Clear");
  instructionsConfirmClearBtn.type = "button";
  instructionsConfirmClearBtn.hidden = true;
  instructionsConfirmClearBtn.addEventListener("click", () => {
    confirmingClearInstructions = false;
    void saveInstructionsValue("");
  });
  instrBtnGroup.appendChild(instructionsConfirmClearBtn);

  instructionsCancelClearBtn = el("button", "ov-btn-ghost", "Cancel");
  instructionsCancelClearBtn.type = "button";
  instructionsCancelClearBtn.hidden = true;
  instructionsCancelClearBtn.addEventListener("click", () => {
    confirmingClearInstructions = false;
    syncInstructionsUi();
  });
  instrBtnGroup.appendChild(instructionsCancelClearBtn);

  instructionsSaveBtn = el("button", "ov-btn-primary", "Save Instructions");
  instructionsSaveBtn.type = "button";
  instructionsSaveBtn.addEventListener("click", () => {
    if (!instructionsTextarea) return;
    void saveInstructionsValue(instructionsTextarea.value);
  });
  instrBtnGroup.appendChild(instructionsSaveBtn);
  instrBtnGroup.appendChild(el("kbd", "ov-kbd", "Ctrl+S"));
  instrActions.appendChild(instrBtnGroup);
  instrPanel.appendChild(instrActions);

  instructionsPanelEl.appendChild(instrPanel);
  panelsStage.appendChild(instructionsPanelEl);
  main.appendChild(panelsStage);

  document.addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "s") {
      if (state.activeTab === "models") {
        ev.preventDefault();
        void handleSaveConfig();
      } else if (state.activeTab === "instructions") {
        ev.preventDefault();
        if (instructionsTextarea) {
          void saveInstructionsValue(instructionsTextarea.value);
        }
      }
    }
  });

  shell.appendChild(main);
  root.replaceChildren(shell);
  shellMounted = true;
  syncInstructionsUi();
  renderModelsPanel();
  syncTabButtons();
  return true;
}

function render(): void {
  if (!ensureShell()) return;
  renderTopbar();
  renderBanner();
  syncTabButtons();
  renderRunsPanel();
  syncInstructionsUi();
  syncConfigStatusUi();
}

function applyMonitorPayload(monitor: OpenMonitorPayload): void {
  state.loading = false;
  if (typeof monitor.project_root === "string" && monitor.project_root) {
    state.projectFilter = monitor.project_root;
  } else if (monitor.project_root === null) {
    state.projectFilter = null;
  }
  if (Array.isArray(monitor.runs)) {
    state.runs = monitor.runs;
  }
  if (typeof monitor.active_count === "number") {
    state.activeCount = monitor.active_count;
  }
  if (typeof monitor.custom_instructions === "string") {
    applyLoadedInstructions(monitor.custom_instructions);
  }
  if (monitor.config) {
    applyLoadedConfig(monitor.config);
  }
  if (monitor.selected_run) {
    state.selectedRun = monitor.selected_run;
    state.selectedKey = {
      project_root: monitor.selected_run.project_root,
      run_id: monitor.selected_run.run_id,
    };
    previewSnapshots.set(
      `${monitor.selected_run.project_root}::${monitor.selected_run.run_id}`,
      monitor.selected_run
    );
    if (!standaloneMode) {
      if (state.connected) {
        void fetchSnapshot(monitor.selected_run.project_root, monitor.selected_run.run_id).then(
          () => render(),
          () => {}
        );
      } else {
        pendingPostConnectUpgrade = "snapshot";
      }
    }
  } else if (state.runs.length > 0 && !state.selectedRun) {
    const key = pickDefaultRunKey(state.runs);
    if (key && !standaloneMode) {
      state.selectedKey = key;
      if (state.connected) {
        void refreshAll(true);
      } else {
        pendingPostConnectUpgrade = "refresh";
      }
    }
  }
}

function buildDefaultPreviewPayload(): OpenMonitorPayload {
  const statuses = ["rolled_back", "completed_with_escalations", "completed", "completed"];
  const runs: RunSummary[] = Array.from({ length: 18 }, (_, i) => {
    const idx = String(18 - i).padStart(2, "0");
    return {
      run_id: `run_20261124_09${idx}03_635e${idx}`,
      project_root:
        i % 3 === 0
          ? "/workspace/sample-app"
          : "/workspace/api-service",
      status: i === 0 ? "rolled_back" : statuses[i % statuses.length],
      created_at_iso: `2026-11-24T09:${idx}:03Z`,
      completed_at_iso: `2026-11-24T09:${idx}:21Z`,
      spend_usd: i % 4 === 0 ? 0.0 : 0.0125,
      max_run_usd: 0.25,
      worker_count: i === 1 ? 2 : 1,
      completed_workers: i === 1 ? 1 : 1,
    };
  });
  return {
    project_root: null,
    runs,
    active_count: 0,
    selected_run: {
      run_id: runs[0].run_id,
      project_root: runs[0].project_root,
      status: "rolled_back",
      created_at_iso: runs[0].created_at_iso,
      completed_at_iso: runs[0].completed_at_iso,
      spend_usd: 0.0,
      limits: {
        max_run_usd: 0.05,
        max_steps_per_worker: 12,
        max_429_wait_seconds: 30,
        request_timeout_sec: 60,
      },
      workers: [
        {
          worker_id: "e2e_math",
          folder: "src",
          status: "completed",
          steps_used: 5,
          max_steps: 12,
          models_used: ["nvidia/nemotron-3-ultra-550b-a55b:free"],
          requested_model: "particle-ai/particle-1:free",
          spend_usd: 0.0,
          static_check: { status: "passed", errors: [] },
          summary:
            "Successfully completed the task:\n1. Updated src/math_utils.py with subtract() and multiply()\n2. Created src/stats_utils.py with mean()",
          notes: ["Both files are valid Python with proper type hints."],
          escalation_reason: null,
          stop_reason: null,
          touched_files: [
            { rel_path: "src/math_utils.py", abs_uri: "", kind: "modified", stat: "+8 / -0" },
            { rel_path: "src/stats_utils.py", abs_uri: "", kind: "created", stat: "+7 (created)" },
          ],
        },
      ],
    },
    custom_instructions: "",
    config: {
      config_path: "~/.local/share/overseer/config.json",
      max_concurrent_workers: 3,
      env_overrides: {},
      tiers: {
        free: {
          primary: "particle-ai/particle-1:free",
          fallbacks: ["nvidia/nemotron-3-ultra-550b-a55b:free", "google/gemma-4-31b-it:free"],
          max_run_usd: null,
        },
        cheap: {
          primary: "deepseek/deepseek-v4.1-flash-1017",
          fallbacks: ["z-ai/glm-5.3-flash"],
          max_run_usd: 0.25,
        },
        heavy: { primary: null, fallbacks: [], max_run_usd: 2.0 },
      },
      provider_routing: { tiers: {}, models: {} },
      api_keys_configured: { free: true, cheap: true, heavy: false, shared: false },
    },
  };
}

function loadStandalonePreview(
  payload: OpenMonitorPayload,
  options?: { tab?: TabId; theme?: "light" | "dark" }
): void {
  standaloneMode = true;
  state.connected = true;
  state.receivedInitialResult = true;
  state.canSendMessage = false;
  state.auxPaneSupported = false;
  configDirty = false;
  instructionsDirty = false;
  previewSnapshots.clear();
  if (options?.theme === "light" || options?.theme === "dark") {
    applyDocumentTheme(options.theme);
  }
  if (modelsCatalog.length === 0) {
    modelsCatalog = [
      {
        id: "particle-ai/particle-1:free",
        name: "Particle 1 (Free)",
        context_length: 131072,
        prompt_price_per_m: 0,
        completion_price_per_m: 0,
        is_free: true,
        supports_tools: true,
      },
      {
        id: "nvidia/nemotron-3-ultra-550b-a55b:free",
        name: "Nemotron 3 Ultra (Free)",
        context_length: 131072,
        prompt_price_per_m: 0,
        completion_price_per_m: 0,
        is_free: true,
        supports_tools: true,
      },
      {
        id: "deepseek/deepseek-v4.1-flash-1017",
        name: "DeepSeek V4.1 Flash",
        context_length: 128000,
        prompt_price_per_m: 0.14,
        completion_price_per_m: 0.28,
        is_free: false,
        supports_tools: true,
      },
      {
        id: "anthropic/claude-sonnet-4.6",
        name: "Claude Sonnet 4.6",
        context_length: 200000,
        prompt_price_per_m: 3.0,
        completion_price_per_m: 15.0,
        is_free: false,
        supports_tools: true,
      },
    ];
    modelsCatalogLoaded = true;
  }
  applyMonitorPayload(payload);
  if (options?.tab === "runs" || options?.tab === "models" || options?.tab === "instructions") {
    state.activeTab = options.tab;
  }
  render();
}

(
  window as unknown as {
    __OVERSEER_LOAD_PREVIEW__?: typeof loadStandalonePreview;
    __OVERSEER_PREVIEW__?: OpenMonitorPayload;
  }
).__OVERSEER_LOAD_PREVIEW__ = loadStandalonePreview;

// Register all handlers BEFORE calling app.connect()
app.ontoolinput = ({ arguments: args }) => {
  if (args && typeof args.project_root === "string" && args.project_root.trim()) {
    state.projectFilter = args.project_root.trim();
  }
  state.loading = true;
  render();
};

app.ontoolresult = (result) => {
  state.receivedInitialResult = true;
  if (fallbackTimer !== null) {
    window.clearTimeout(fallbackTimer);
    fallbackTimer = null;
  }
  try {
    const payload = extractToolPayload<OpenMonitorPayload | RunSnapshot>(result);
    state.loading = false;
    if (payload && "workers" in payload && Array.isArray(payload.workers)) {
      const snap = payload as RunSnapshot;
      state.selectedRun = snap;
      state.selectedKey = { project_root: snap.project_root, run_id: snap.run_id };
    } else if (payload) {
      applyMonitorPayload(payload as OpenMonitorPayload);
    }
  } catch (err) {
    state.loading = false;
    const msg = err instanceof Error ? err.message : String(err);
    setBanner("error", msg);
  }
  render();
  schedulePollingIfNeeded();
};

app.ontoolcancelled = (params) => {
  state.loading = false;
  clearPolling();
  const reason = params?.reason ? ` (${params.reason})` : "";
  setBanner("info", `Monitor request cancelled${reason}.`);
};

app.onhostcontextchanged = (ctx) => {
  applyThemeFromContext(ctx);
};

async function init(): Promise<void> {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      if (state.connected && !standaloneMode) {
        void refreshAll(true);
      }
    } else {
      clearPolling();
    }
  });
  if (isStandaloneWindow()) {
    const params = new URLSearchParams(window.location.search);
    const qTheme = params.get("theme");
    const theme: "light" | "dark" =
      qTheme === "light" || qTheme === "dark"
        ? qTheme
        : window.matchMedia?.("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
    const qTab = params.get("tab");
    const tab: TabId =
      qTab === "models" || qTab === "instructions" || qTab === "runs" ? qTab : "runs";
    const injected = (window as unknown as { __OVERSEER_PREVIEW__?: OpenMonitorPayload })
      .__OVERSEER_PREVIEW__;
    loadStandalonePreview(injected ?? buildDefaultPreviewPayload(), { tab, theme });
    return;
  }
  render();
  try {
    await app.connect();
    state.connected = true;
    applyThemeFromContext(app.getHostContext());
    const caps = app.getHostCapabilities();
    state.canSendMessage = Boolean(caps?.message);
    render();
    if (pendingPostConnectUpgrade === "snapshot" && state.selectedKey) {
      pendingPostConnectUpgrade = null;
      void fetchSnapshot(state.selectedKey.project_root, state.selectedKey.run_id).then(
        () => render(),
        () => {}
      );
    } else if (pendingPostConnectUpgrade === "refresh") {
      pendingPostConnectUpgrade = null;
      void refreshAll(true);
    }
    schedulePollingIfNeeded();
    if (!instructionsLoaded) {
      void loadCustomInstructions();
    }
    if (!configLoaded) {
      void loadConfig();
    }

    if (!state.receivedInitialResult) {
      fallbackTimer = window.setTimeout(() => {
        fallbackTimer = null;
        if (!state.receivedInitialResult) {
          void refreshAll(false);
        }
      }, 150);
    }
  } catch (err) {
    state.loading = false;
    const msg = err instanceof Error ? err.message : String(err);
    setBanner("error", `Failed to connect to MCP host: ${msg}`);
  }
}

void init();
