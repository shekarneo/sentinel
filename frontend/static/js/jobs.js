const API_V1 = "/api/v1";

function jobsPage() {
  return {
    jobs: [],
    summary: {},

    async load() {
      const store = Alpine.store("console");
      store.setLoading(true);
      try {
        const payload = await fetch(`${API_V1}/jobs`).then((r) => r.json());
        this.jobs = payload.jobs || [];
        this.summary = payload.summary || {};
      } catch (error) {
        store.showToast("Jobs", error.message || "Failed to load jobs.");
      } finally {
        store.setLoading(false);
      }
    },

    stateBadge(state) {
      const map = {
        queued: "text-bg-secondary",
        running: "text-bg-primary",
        completed: "text-bg-success",
        failed: "text-bg-danger",
        cancelled: "text-bg-dark",
      };
      return map[state] || "text-bg-secondary";
    },

    formatWhen(value) {
      return new Date(value).toLocaleString();
    },
  };
}

window.jobsPage = jobsPage;
