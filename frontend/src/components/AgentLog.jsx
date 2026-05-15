const TYPE_STYLE = {
  anomaly_detected: 'border-yellow-700 bg-yellow-950 text-yellow-200',
  diagnosis_ready:  'border-green-700  bg-green-950  text-green-200',
};

const TYPE_LABEL = {
  anomaly_detected: 'ANOMALY',
  diagnosis_ready:  'DIAGNOSIS',
};

export default function AgentLog({ events }) {
  return (
    <div
      className="bg-gray-900 rounded-lg border border-gray-800 flex flex-col"
      style={{ height: 260 }}
    >
      <div className="px-4 py-2 border-b border-gray-800 flex items-center justify-between shrink-0">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Agent Events
        </h2>
        <span className="text-xs text-gray-600">{events.length}</span>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1 min-h-0">
        {events.length === 0 && (
          <p className="text-gray-600 text-xs p-1">
            Start monitor to receive live events…
          </p>
        )}
        {events.map((event, i) => {
          const style =
            TYPE_STYLE[event.type] ?? 'border-gray-700 bg-gray-800 text-gray-300';
          const label =
            TYPE_LABEL[event.type] ?? (event.type ?? 'EVENT').toUpperCase();
          const d = event.data ?? {};

          return (
            <div key={i} className={`text-xs rounded border px-2 py-1.5 ${style}`}>
              <div className="flex items-center justify-between mb-0.5">
                <span className="font-bold">{label}</span>
                <span className="opacity-60">
                  {d.timestamp?.slice(11, 19) ?? ''}
                </span>
              </div>

              {event.type === 'anomaly_detected' && (
                <div className="opacity-90">
                  {d.cell_id} &middot; {d.fault_type} &middot; {d.severity} &middot;{' '}
                  {((d.confidence ?? 0) * 100).toFixed(1)}%
                </div>
              )}

              {event.type === 'diagnosis_ready' && (
                <div className="opacity-90 truncate">
                  {d.incident_id} &middot; {d.confidence} confidence
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
