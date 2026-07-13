const API_V1 = "/api/v1";

function datasetsPage() {
  return {
    jobs: [],
    selectedJob: null,
    results: null,
    form: {
      dataset_type: "image_folder",
      operation: "recognition",
      source_path: "/data/faces",
      item_count: 3,
      pipeline_profile: "search",
    },

    async load() {
      const store = Alpine.store("console");
      store.setLoading(true);
      try {
        const payload = await fetch(`${API_V1}/datasets/jobs`).then((r) => r.json());
        this.jobs = payload.jobs || [];
        if (this.selectedJob) {
          this.selectedJob = this.jobs.find((job) => job.id === this.selectedJob.id) || null;
        }
      } catch (error) {
        store.showToast("Datasets", error.message || "Failed to load dataset jobs.");
      } finally {
        store.setLoading(false);
      }
    },

    async submitJob() {
      const store = Alpine.store("console");
      store.setLoading(true);
      try {
        const response = await fetch(`${API_V1}/datasets/process`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(this.form),
        });
        if (!response.ok) throw new Error(await response.text());
        const job = await response.json();
        this.selectedJob = job;
        await this.load();
        await this.loadResults(job.id);
        store.showToast("Datasets", "Dataset job queued.");
      } catch (error) {
        store.showToast("Datasets", error.message || "Failed to queue dataset job.");
      } finally {
        store.setLoading(false);
      }
    },

    async selectJob(job) {
      this.selectedJob = job;
      await this.loadResults(job.id);
    },

    async loadResults(jobId) {
      const response = await fetch(`${API_V1}/datasets/results/${jobId}`);
      if (!response.ok) return;
      this.results = await response.json();
    },

    progressLabel() {
      const summary = this.selectedJob?.summary;
      if (!summary || !summary.total_items) return "0%";
      const done = summary.processed_items + summary.failed_items;
      return `${Math.round((done / summary.total_items) * 100)}%`;
    },
  };
}

window.datasetsPage = datasetsPage;
