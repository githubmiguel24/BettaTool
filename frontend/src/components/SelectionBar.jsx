import { TrashIcon } from "./Icons.jsx";

/** "Select all" checkbox plus a Delete button for the checked rows. */
function SelectionBar({ visibleIds, selected, onToggleAll, onDelete }) {
  const count = visibleIds.filter((id) => selected.has(id)).length;
  const allSelected = visibleIds.length > 0 && count === visibleIds.length;

  return (
    <div className="mb-5 flex items-center justify-between gap-4">
      <label className="flex cursor-pointer items-center gap-3 text-base text-slate-500">
        <input
          type="checkbox"
          checked={allSelected}
          onChange={() => onToggleAll(visibleIds)}
          className="h-5 w-5 rounded border-slate-300 accent-betta-600"
        />
        {count > 0 ? `${count} selected` : "Select all"}
      </label>
      <button
        type="button"
        disabled={count === 0}
        onClick={onDelete}
        className="flex items-center gap-2 rounded-full bg-red-50 px-5 py-2.5 text-base font-medium text-red-600 transition hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-red-50"
      >
        <TrashIcon className="h-5 w-5" />
        Delete{count > 0 ? ` (${count})` : ""}
      </button>
    </div>
  );
}

export default SelectionBar;
