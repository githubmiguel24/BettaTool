import { useCallback, useEffect, useState } from "react";
import { deleteReports, listReports } from "../api/reports.js";

/**
 * Loads the user's reports and owns the multi-select + delete flow shared by
 * the History and Profile pages. Deleting goes through `pending` so the page
 * can show a confirmation dialog first.
 */
export function useReportList() {
  const [reports, setReports] = useState(null);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(() => new Set());
  const [pending, setPending] = useState(null); // ids awaiting confirmation
  const [deleting, setDeleting] = useState(false);

  const load = useCallback(
    () => listReports().then(setReports).catch((e) => setError(e.message)),
    [],
  );

  useEffect(() => {
    load();
  }, [load]);

  const toggle = (id) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (!next.delete(id)) next.add(id);
      return next;
    });

  const toggleAll = (ids) =>
    setSelected((prev) =>
      ids.every((id) => prev.has(id)) ? new Set() : new Set(ids),
    );

  const requestDelete = (ids) => ids.length && setPending(ids);
  const cancelDelete = () => setPending(null);

  async function confirmDelete() {
    setDeleting(true);
    setError(null);
    try {
      await deleteReports(pending);
    } catch (e) {
      setError(e.message);
    }
    // Re-read either way: after a partial failure the list must match the DB.
    await load();
    setSelected(new Set());
    setPending(null);
    setDeleting(false);
  }

  return {
    reports,
    error,
    selected,
    toggle,
    toggleAll,
    pending,
    deleting,
    requestDelete,
    cancelDelete,
    confirmDelete,
  };
}

export function deleteMessage(count) {
  return count === 1
    ? "This permanently deletes the image and its results."
    : `This permanently deletes ${count} images and their results.`;
}
