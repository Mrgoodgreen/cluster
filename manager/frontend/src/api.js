const jsonHeaders = { "Content-Type": "application/json" };

async function request(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  health: (signal) => request("/api/health", { signal }),
  storageTree: (path) =>
    request(`/api/storage/tree${path ? `?path=${encodeURIComponent(path)}` : ""}`),
  storageMkdir: (path) =>
    request("/api/storage/mkdir", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ path }),
    }),
  listTasks: (page = 1, pageSize = 20, { status = '', q = '' } = {}, signal) =>
    request(`/api/tasks?${new URLSearchParams({ page, page_size: pageSize, ...(status ? { status } : {}), ...(q ? { q } : {}) })}`, { signal }),
  getTask: (id, signal) => request(`/api/tasks/${id}`, { signal }),
  createTask: (input_path, output_path) =>
    request("/api/tasks", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ input_path, output_path }),
    }),
  cancelTask: (id) =>
    request(`/api/tasks/${id}/cancel`, {
      method: "POST",
      headers: jsonHeaders,
    }),
  restartTask: (id) =>
    request(`/api/tasks/${id}/restart`, {
      method: "POST",
      headers: jsonHeaders,
    }),
};
