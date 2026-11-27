import type {
  OverseerConfigSnapshot,
  ProviderRoutingRule,
  ProviderSortPreset,
  TierName,
} from "./types.ts";

export interface RoutingRuleDraft {
  sort: "" | ProviderSortPreset;
  order: string;
  only: string;
  allow_fallbacks: "default" | "true" | "false";
  zdr: "default" | "true" | "false";
  quantizations: string;
}

export interface TierFormDraft {
  primary: string;
  fallbacks: string;
  max_run_usd: string;
  routing: RoutingRuleDraft;
}

export interface ModelOverrideDraft {
  model: string;
  routing: RoutingRuleDraft;
}

export interface ConfigFormDraft {
  max_concurrent_workers: string;
  tiers: Record<TierName, TierFormDraft>;
  modelOverrides: ModelOverrideDraft[];
}

export function parseCsvList(raw: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of raw.split(/[,\n]/)) {
    const cleaned = part.trim();
    if (cleaned && !seen.has(cleaned)) {
      seen.add(cleaned);
      out.push(cleaned);
    }
  }
  return out;
}

export function formatCsvList(items?: string[] | null): string {
  if (!Array.isArray(items) || items.length === 0) return "";
  return items.join(", ");
}

export function emptyRoutingDraft(): RoutingRuleDraft {
  return {
    sort: "",
    order: "",
    only: "",
    allow_fallbacks: "default",
    zdr: "default",
    quantizations: "",
  };
}

function boolTriStateToDraft(val?: boolean | null): "default" | "true" | "false" {
  return val === true ? "true" : val === false ? "false" : "default";
}

function draftToBoolTriState(val: "default" | "true" | "false"): boolean | null {
  return val === "true" ? true : val === "false" ? false : null;
}

function ruleToDraft(rule?: ProviderRoutingRule | null): RoutingRuleDraft {
  if (!rule) return emptyRoutingDraft();
  return {
    sort: rule.sort ?? "",
    order: formatCsvList(rule.order),
    only: formatCsvList(rule.only),
    allow_fallbacks: boolTriStateToDraft(rule.allow_fallbacks),
    zdr: boolTriStateToDraft(rule.zdr),
    quantizations: formatCsvList(rule.quantizations),
  };
}

export function configToDraft(cfg: OverseerConfigSnapshot): ConfigFormDraft {
  const tierNames: TierName[] = ["free", "cheap", "heavy"];
  const tiers = {} as Record<TierName, TierFormDraft>;
  for (const tier of tierNames) {
    const t = cfg.tiers[tier];
    tiers[tier] = {
      primary: t?.primary ?? "",
      fallbacks: formatCsvList(t?.fallbacks),
      max_run_usd:
        tier !== "free" && typeof t?.max_run_usd === "number" ? String(t.max_run_usd) : "",
      routing: ruleToDraft(cfg.provider_routing?.tiers?.[tier]),
    };
  }
  const modelOverrides: ModelOverrideDraft[] = Object.entries(
    cfg.provider_routing?.models ?? {}
  ).map(([model, rule]) => ({
    model,
    routing: ruleToDraft(rule),
  }));
  const max_concurrent_workers =
    typeof cfg.max_concurrent_workers === "number" ? String(cfg.max_concurrent_workers) : "3";
  return { max_concurrent_workers, tiers, modelOverrides };
}

export function buildRulePayload(draft: RoutingRuleDraft): ProviderRoutingRule | null {
  const order = parseCsvList(draft.order);
  const only = parseCsvList(draft.only);
  const quantizations = parseCsvList(draft.quantizations);
  const sort = draft.sort || null;
  const allow_fallbacks = draftToBoolTriState(draft.allow_fallbacks);
  const zdr = draftToBoolTriState(draft.zdr);
  if (
    sort === null &&
    order.length === 0 &&
    only.length === 0 &&
    allow_fallbacks === null &&
    zdr === null &&
    quantizations.length === 0
  ) {
    return null;
  }
  return { sort, order, only, allow_fallbacks, zdr, quantizations };
}

function arraysEqual(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export function buildConfigUpdatePayload(
  draft: ConfigFormDraft,
  baseline?: OverseerConfigSnapshot | null
):
  | {
      ok: true;
      payload: {
        tiers: Record<string, unknown>;
        provider_routing: Record<string, unknown>;
        max_concurrent_workers?: number;
      };
    }
  | { ok: false; error: string } {
  const rawWorkers = draft.max_concurrent_workers.trim();
  const parsedWorkers = Number(rawWorkers);
  if (!rawWorkers || !Number.isInteger(parsedWorkers) || parsedWorkers < 1 || parsedWorkers > 10) {
    return { ok: false, error: "Max concurrent workers must be an integer between 1 and 10." };
  }

  const freePrimary = draft.tiers.free.primary.trim();
  if (!freePrimary) {
    return { ok: false, error: "Free tier requires a primary model ending with ':free'." };
  }
  if (!freePrimary.endsWith(":free")) {
    return { ok: false, error: `Free primary model "${freePrimary}" must end with ':free'.` };
  }
  const freeFallbacks = parseCsvList(draft.tiers.free.fallbacks);
  for (const fb of freeFallbacks) {
    if (!fb.endsWith(":free")) {
      return { ok: false, error: `Free fallback model "${fb}" must end with ':free'.` };
    }
  }

  const cheapPrimary = draft.tiers.cheap.primary.trim();
  if (!cheapPrimary) {
    return { ok: false, error: "Cheap tier requires a primary model slug." };
  }
  const cheapFallbacks = parseCsvList(draft.tiers.cheap.fallbacks);

  const parseCap = (raw: string, label: string): { ok: true; val: number | null } | { ok: false; error: string } => {
    const trimmed = raw.trim();
    if (!trimmed) return { ok: true, val: null };
    const n = Number(trimmed);
    if (!Number.isFinite(n) || n <= 0 || n > 50) {
      return { ok: false, error: `${label} per-run USD limit must be > 0 and <= 50.` };
    }
    return { ok: true, val: n };
  };

  const cheapCap = parseCap(draft.tiers.cheap.max_run_usd, "Cheap");
  if (!cheapCap.ok) return cheapCap;

  const heavyCap = parseCap(draft.tiers.heavy.max_run_usd, "Heavy");
  if (!heavyCap.ok) return heavyCap;

  const heavyPrimary = draft.tiers.heavy.primary.trim() || null;
  const heavyFallbacks = parseCsvList(draft.tiers.heavy.fallbacks);

  const tierRouting: Record<string, ProviderRoutingRule | null> = {};
  for (const tier of ["free", "cheap", "heavy"] as const) {
    tierRouting[tier] = buildRulePayload(draft.tiers[tier].routing);
  }

  const modelRouting: Record<string, ProviderRoutingRule> = {};
  for (const row of draft.modelOverrides) {
    const slug = row.model.trim();
    const rule = buildRulePayload(row.routing);
    if (!slug && rule === null) {
      continue;
    }
    if (!slug) {
      return { ok: false, error: "Enter a model slug for every provider routing override row." };
    }
    if (slug in modelRouting) {
      return { ok: false, error: `Duplicate provider routing override for "${slug}".` };
    }
    if (rule === null) {
      return { ok: false, error: `Select at least one routing preference for "${slug}" or remove the row.` };
    }
    modelRouting[slug] = rule;
  }

  const tiers: Record<string, Record<string, unknown>> = {};
  const freePatch: Record<string, unknown> = {};
  if (!baseline || freePrimary !== (baseline.tiers.free.primary ?? "")) {
    freePatch.primary = freePrimary;
  }
  if (!baseline || !arraysEqual(freeFallbacks, baseline.tiers.free.fallbacks ?? [])) {
    freePatch.fallbacks = freeFallbacks;
  }
  if (Object.keys(freePatch).length > 0) {
    tiers.free = freePatch;
  }

  const cheapPatch: Record<string, unknown> = {};
  if (!baseline || cheapPrimary !== (baseline.tiers.cheap.primary ?? "")) {
    cheapPatch.primary = cheapPrimary;
  }
  if (!baseline || !arraysEqual(cheapFallbacks, baseline.tiers.cheap.fallbacks ?? [])) {
    cheapPatch.fallbacks = cheapFallbacks;
  }
  if (!baseline || cheapCap.val !== (baseline.tiers.cheap.max_run_usd ?? null)) {
    cheapPatch.max_run_usd = cheapCap.val;
  }
  if (Object.keys(cheapPatch).length > 0) {
    tiers.cheap = cheapPatch;
  }

  const heavyPatch: Record<string, unknown> = {};
  if (!baseline || heavyPrimary !== (baseline.tiers.heavy.primary ?? null)) {
    heavyPatch.primary = heavyPrimary;
  }
  if (!baseline || !arraysEqual(heavyFallbacks, baseline.tiers.heavy.fallbacks ?? [])) {
    heavyPatch.fallbacks = heavyFallbacks;
  }
  if (!baseline || heavyCap.val !== (baseline.tiers.heavy.max_run_usd ?? null)) {
    heavyPatch.max_run_usd = heavyCap.val;
  }
  if (Object.keys(heavyPatch).length > 0) {
    tiers.heavy = heavyPatch;
  }

  const payload: {
    tiers: Record<string, unknown>;
    provider_routing: Record<string, unknown>;
    max_concurrent_workers?: number;
  } = {
    tiers,
    provider_routing: {
      tiers: tierRouting,
      models: modelRouting,
      replace_models: true,
    },
  };
  if (
    !baseline ||
    typeof baseline.max_concurrent_workers !== "number" ||
    parsedWorkers !== baseline.max_concurrent_workers
  ) {
    payload.max_concurrent_workers = parsedWorkers;
  }

  return {
    ok: true,
    payload,
  };
}
