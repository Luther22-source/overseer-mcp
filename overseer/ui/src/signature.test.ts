import assert from "node:assert/strict";
import test from "node:test";
import {
  buildConfigUpdatePayload,
  configToDraft,
  parseCsvList,
} from "./config_helpers.ts";
import { buildRenderSignature, type RenderSignatureState } from "./signature.ts";
import type { OverseerConfigSnapshot } from "./types.ts";

function makeBaseState(): RenderSignatureState {
  return {
    connected: true,
    loading: false,
    refreshing: false,
    cancelling: false,
    rollingBack: false,
    confirmingRollback: false,
    sendingMessage: false,
    auxPaneSupported: true,
    canSendMessage: true,
    projectFilter: null,
    activeCount: 1,
    runsHasMore: false,
    loadingMoreRuns: false,
    workerDiffsKey: "",
    banner: null,
    selectedKey: { project_root: "C:/repo", run_id: "run_1" },
    runs: [{ run_id: "run_1", status: "running", spend_usd: 0, completed_workers: 0 }],
    selectedRun: {
      run_id: "run_1",
      status: "running",
      spend_usd: 0,
      workers: [
        {
          worker_id: "w1",
          status: "running",
          steps_used: 2,
          spend_usd: 0,
          static_check: { status: "not_run", errors: [] },
          summary: "",
          notes: [],
          touched_files: [
            { rel_path: "src/a.py", abs_uri: "file:///C:/repo/src/a.py", kind: "modified", stat: "+2 / -0" },
          ],
        },
      ],
    },
  };
}

function makeConfigSnapshot(): OverseerConfigSnapshot {
  return {
    config_path: "C:/appdata/overseer/config.json",
    max_concurrent_workers: 3,
    env_overrides: {},
    tiers: {
      free: {
        primary: "particle-ai/particle-1:free",
        fallbacks: ["nvidia/nemotron-3-ultra-550b-a55b:free"],
        max_run_usd: null,
      },
      cheap: {
        primary: "deepseek/deepseek-v4.1-flash-1017",
        fallbacks: ["z-ai/glm-5.3-flash"],
        max_run_usd: 0.25,
      },
      heavy: {
        primary: null,
        fallbacks: [],
        max_run_usd: 2.0,
      },
    },
    provider_routing: {
      tiers: { cheap: { sort: "price", allow_fallbacks: true } },
      models: { "anthropic/claude-sonnet-4.6": { only: ["Anthropic"], allow_fallbacks: false } },
    },
    api_keys_configured: { free: true, cheap: true, heavy: false, shared: false },
  };
}

test("buildRenderSignature detects touched_files edits with same length, steps, and spend", () => {
  const base = makeBaseState();
  const baseSig = buildRenderSignature(base);
  assert.equal(baseSig, buildRenderSignature(makeBaseState()));

  const statChanged = makeBaseState();
  (statChanged.selectedRun as any).workers[0].touched_files[0].stat = "+5 / -1";
  assert.notEqual(buildRenderSignature(statChanged), baseSig);

  const pathChanged = makeBaseState();
  (pathChanged.selectedRun as any).workers[0].touched_files[0].rel_path = "src/b.py";
  assert.notEqual(buildRenderSignature(pathChanged), baseSig);

  const summaryChanged = makeBaseState();
  (summaryChanged.selectedRun as any).workers[0].summary = "Updated helper";
  assert.notEqual(buildRenderSignature(summaryChanged), baseSig);

  const checkErrChanged = makeBaseState();
  (checkErrChanged.selectedRun as any).workers[0].static_check.errors = ["SyntaxError line 4"];
  assert.notEqual(buildRenderSignature(checkErrChanged), baseSig);
});

test("parseCsvList trims, splits commas/newlines, and deduplicates", () => {
  assert.deepEqual(parseCsvList(" a, b \n a , , c "), ["a", "b", "c"]);
  assert.deepEqual(parseCsvList("  "), []);
});

test("configToDraft and buildConfigUpdatePayload validate tiers, USD bounds, and overrides", () => {
  const draft = configToDraft(makeConfigSnapshot());
  assert.equal(draft.tiers.free.primary, "particle-ai/particle-1:free");
  assert.equal(draft.tiers.heavy.primary, "");
  assert.equal(draft.tiers.heavy.max_run_usd, "2");
  assert.equal(draft.modelOverrides.length, 1);

  const okRes = buildConfigUpdatePayload(draft);
  assert.equal(okRes.ok, true);
  if (okRes.ok) {
    assert.equal((okRes.payload.tiers.heavy as any).primary, null);
    assert.equal((okRes.payload.tiers.cheap as any).max_run_usd, 0.25);
    assert.equal((okRes.payload.provider_routing as any).replace_models, true);
  }

  const badFree = configToDraft(makeConfigSnapshot());
  badFree.tiers.free.primary = "openai/gpt-5.2";
  assert.equal(buildConfigUpdatePayload(badFree).ok, false);

  const badUsd = configToDraft(makeConfigSnapshot());
  badUsd.tiers.heavy.max_run_usd = "75";
  assert.equal(buildConfigUpdatePayload(badUsd).ok, false);

  const dupModel = configToDraft(makeConfigSnapshot());
  dupModel.modelOverrides.push({
    model: "anthropic/claude-sonnet-4.6",
    routing: {
      sort: "latency",
      order: "",
      only: "",
      allow_fallbacks: "default",
      zdr: "default",
      quantizations: "",
    },
  });
  assert.equal(buildConfigUpdatePayload(dupModel).ok, false);

  // Diff-only tier payloads when baseline is passed (protects untouched .env tiers & single-key saves)
  const baseline = makeConfigSnapshot();
  const unchangedRes = buildConfigUpdatePayload(configToDraft(baseline), baseline);
  assert.equal(unchangedRes.ok, true);
  if (unchangedRes.ok) {
    assert.deepEqual(unchangedRes.payload.tiers, {});
  }

  const freeOnlyDraft = configToDraft(baseline);
  freeOnlyDraft.tiers.free.primary = "qwen/qwen3-coder:free";
  const freeOnlyRes = buildConfigUpdatePayload(freeOnlyDraft, baseline);
  assert.equal(freeOnlyRes.ok, true);
  if (freeOnlyRes.ok) {
    assert.deepEqual(freeOnlyRes.payload.tiers, {
      free: { primary: "qwen/qwen3-coder:free" },
    });
  }

  const heavyCapDraft = configToDraft(baseline);
  heavyCapDraft.tiers.heavy.max_run_usd = "4.5";
  const heavyCapRes = buildConfigUpdatePayload(heavyCapDraft, baseline);
  assert.equal(heavyCapRes.ok, true);
  if (heavyCapRes.ok) {
    assert.deepEqual(heavyCapRes.payload.tiers, {
      heavy: { max_run_usd: 4.5 },
    });
    assert.equal(heavyCapRes.payload.max_concurrent_workers, undefined);
  }

  // max_concurrent_workers validation & diffing
  const workersDraft = configToDraft(baseline);
  workersDraft.max_concurrent_workers = "6";
  const workersRes = buildConfigUpdatePayload(workersDraft, baseline);
  assert.equal(workersRes.ok, true);
  if (workersRes.ok) {
    assert.equal(workersRes.payload.max_concurrent_workers, 6);
  }

  const badWorkersLow = configToDraft(baseline);
  badWorkersLow.max_concurrent_workers = "0";
  assert.equal(buildConfigUpdatePayload(badWorkersLow, baseline).ok, false);

  const badWorkersHigh = configToDraft(baseline);
  badWorkersHigh.max_concurrent_workers = "11";
  assert.equal(buildConfigUpdatePayload(badWorkersHigh, baseline).ok, false);

  const badWorkersFloat = configToDraft(baseline);
  badWorkersFloat.max_concurrent_workers = "2.5";
  assert.equal(buildConfigUpdatePayload(badWorkersFloat, baseline).ok, false);
});

test("configToDraft and buildConfigUpdatePayload preserve zdr and quantizations", () => {
  const cfg = makeConfigSnapshot();
  cfg.provider_routing.tiers.heavy = {
    zdr: true,
    quantizations: ["fp8", "bf16"],
  };
  cfg.provider_routing.models["deepseek/deepseek-v4.1-flash-1017"] = {
    zdr: false,
    quantizations: ["int8"],
  };

  const draft = configToDraft(cfg);
  assert.equal(draft.tiers.heavy.routing.zdr, "true");
  assert.equal(draft.tiers.heavy.routing.quantizations, "fp8, bf16");
  const modelRow = draft.modelOverrides.find(
    (row) => row.model === "deepseek/deepseek-v4.1-flash-1017"
  );
  assert.ok(modelRow);
  assert.equal(modelRow.routing.zdr, "false");
  assert.equal(modelRow.routing.quantizations, "int8");

  // Unrelated save (e.g. max_concurrent_workers) must preserve zdr & quantizations
  draft.max_concurrent_workers = "5";
  const res = buildConfigUpdatePayload(draft, cfg);
  assert.equal(res.ok, true);
  if (res.ok) {
    assert.equal(res.payload.max_concurrent_workers, 5);
    const pr = res.payload.provider_routing as {
      tiers: Record<string, any>;
      models: Record<string, any>;
      replace_models: boolean;
    };
    assert.equal(pr.tiers.heavy.zdr, true);
    assert.deepEqual(pr.tiers.heavy.quantizations, ["fp8", "bf16"]);
    assert.equal(pr.models["deepseek/deepseek-v4.1-flash-1017"].zdr, false);
    assert.deepEqual(pr.models["deepseek/deepseek-v4.1-flash-1017"].quantizations, ["int8"]);
  }
});
