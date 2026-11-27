export type TabId = "runs" | "models" | "instructions";

export type TierName = "free" | "cheap" | "heavy";

export type ProviderSortPreset = "price" | "throughput" | "latency";

export interface RunSummary {
  run_id: string;
  project_root: string;
  status: string;
  created_at_iso: string;
  completed_at_iso: string | null;
  spend_usd: number;
  max_run_usd: number;
  worker_count: number;
  completed_workers: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

export interface TouchedFile {
  rel_path: string;
  abs_uri: string;
  kind: "created" | "modified" | string;
  stat: string;
}

export interface StaticCheck {
  status: string;
  errors: string[];
}

export interface WorkerSnapshot {
  worker_id: string;
  folder: string;
  status: string;
  steps_used: number;
  max_steps: number;
  models_used: string[];
  requested_model?: string;
  spend_usd: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  static_check: StaticCheck;
  summary: string;
  notes: string[];
  escalation_reason: string | null;
  stop_reason: string | null;
  touched_files: TouchedFile[];
}

export interface RunLimits {
  max_run_usd: number;
  max_steps_per_worker: number;
  max_429_wait_seconds?: number;
  request_timeout_sec?: number;
}

export interface RunSnapshot {
  run_id: string;
  project_root: string;
  status: string;
  created_at_iso: string;
  completed_at_iso: string | null;
  spend_usd: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  limits: RunLimits;
  workers: WorkerSnapshot[];
}

export interface ProviderRoutingRule {
  sort?: ProviderSortPreset | null;
  order?: string[];
  only?: string[];
  allow_fallbacks?: boolean | null;
  zdr?: boolean | null;
  quantizations?: string[];
}

export interface TierConfigSnapshot {
  primary: string | null;
  fallbacks: string[];
  max_run_usd: number | null;
}

export interface OverseerConfigSnapshot {
  config_path: string;
  max_concurrent_workers?: number;
  env_overrides?: Record<string, string>;
  tiers: Record<TierName, TierConfigSnapshot>;
  provider_routing: {
    tiers: Partial<Record<TierName, ProviderRoutingRule>>;
    models: Record<string, ProviderRoutingRule>;
  };
  api_keys_configured: {
    free: boolean;
    cheap: boolean;
    heavy: boolean;
    shared: boolean;
  };
}

export interface OpenMonitorPayload {
  project_root?: string | null;
  runs?: RunSummary[];
  active_count?: number;
  selected_run?: RunSnapshot | null;
  custom_instructions?: string;
  config?: OverseerConfigSnapshot;
}

export interface CustomInstructionsPayload {
  custom_instructions: string;
}

export interface ListRunsPayload {
  runs: RunSummary[];
  active_count: number;
  total_count?: number;
  has_more?: boolean;
}

export interface ModelCatalogEntry {
  id: string;
  name: string;
  context_length: number | null;
  prompt_price_per_m: number;
  completion_price_per_m: number;
  is_free: boolean;
  supports_tools: boolean;
}

export interface ListModelsPayload {
  models: ModelCatalogEntry[];
  count: number;
}

export interface RollbackRunPayload {
  run_id: string;
  status: string;
  restored_files: string[];
  deleted_files: string[];
  skipped_conflicts: string[];
  already_rolled_back: boolean;
}

export interface WorkerPatchPayload {
  run_id: string;
  worker_id: string;
  patch: string;
}

export interface BannerState {
  kind: "info" | "success" | "error";
  text: string;
}
