export interface RenderSignatureState {
  connected: boolean;
  loading: boolean;
  refreshing: boolean;
  cancelling: boolean;
  rollingBack: boolean;
  confirmingRollback: boolean;
  sendingMessage: boolean;
  auxPaneSupported: boolean;
  canSendMessage: boolean;
  projectFilter: string | null;
  activeCount: number;
  runsHasMore: boolean;
  loadingMoreRuns: boolean;
  workerDiffsKey: string;
  banner: unknown;
  selectedKey: unknown;
  runs: unknown;
  selectedRun: unknown;
}

export function buildRenderSignature(s: RenderSignatureState): string {
  return JSON.stringify([
    s.connected,
    s.loading,
    s.refreshing,
    s.cancelling,
    s.rollingBack,
    s.confirmingRollback,
    s.sendingMessage,
    s.auxPaneSupported,
    s.canSendMessage,
    s.projectFilter,
    s.activeCount,
    s.runsHasMore,
    s.loadingMoreRuns,
    s.workerDiffsKey,
    s.banner,
    s.selectedKey,
    s.runs,
    s.selectedRun,
  ]);
}
