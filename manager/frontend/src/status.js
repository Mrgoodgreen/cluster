export function statusColor(status) {
  switch (status) {
    case "completed":
    case "success":
    case "skipped":
      return "green";
    case "processing":
    case "pending":
      return "blue";
    case "error":
      return "red";
    case "cancelled":
      return "default";
    default:
      return "default";
  }
}
