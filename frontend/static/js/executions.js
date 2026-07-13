const API_V1 = "/api/v1";

function executionsPage() {
  return {
    executions: [],
    selected: null,

    async load() {
      const store = Alpine.store("console");
      store.setLoading(true);
      try {
        const payload = await fetch(`${API_V1}/executions/latest?limit=20`).then((r) => r.json());
        this.executions = payload.executions || [];
        if (this.executions.length && !this.selected) {
          await this.select(this.executions[0].id);
        }
      } catch (error) {
        store.showToast("Executions", error.message || "Failed to load executions.");
      } finally {
        store.setLoading(false);
      }
    },

    async select(executionId) {
      const store = Alpine.store("console");
      store.setLoading(true);
      try {
        const response = await fetch(`${API_V1}/executions/${encodeURIComponent(executionId)}`);
        if (!response.ok) {
          throw new Error("Execution not found.");
        }
        this.selected = await response.json();
      } catch (error) {
        store.showToast("Executions", error.message);
      } finally {
        store.setLoading(false);
      }
    },

    statusBadge(status) {
      const map = {
        success: "text-bg-success",
        warning: "text-bg-warning",
        failed: "text-bg-danger",
        running: "text-bg-primary",
        skipped: "text-bg-secondary",
      };
      return map[status] || "text-bg-secondary";
    },

    stageWidth(durationMs) {
      if (!this.selected?.stages?.length) return 0;
      const max = Math.max(...this.selected.stages.map((stage) => stage.duration_ms), 1);
      return Math.max(12, Math.round((durationMs / max) * 100));
    },

    failedStages() {
      if (!this.selected) return [];
      return this.selected.stages.filter(
        (stage) => stage.status === "failed" || stage.errors.length > 0,
      );
    },

    formatWhen(value) {
      return new Date(value).toLocaleString();
    },
  };
}

window.executionsPage = executionsPage;
