import { useState, useRef, useEffect } from 'react';
import { askAssistant } from '../api/mcpClient.js';

export default function NOCAssistant({ rcaResult }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function send() {
    const question = input.trim();
    if (!question || loading) return;
    setInput('');
    setMessages((m) => [...m, { role: 'user', text: question }]);
    setLoading(true);
    try {
      const context = rcaResult ? { latest_rca: rcaResult } : {};
      const res = await askAssistant(question, context);
      setMessages((m) => [...m, { role: 'assistant', text: res.answer }]);
    } catch (e) {
      setMessages((m) => [...m, { role: 'error', text: e.message }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="bg-gray-900 rounded-lg border border-gray-800 flex flex-col"
      style={{ height: 300 }}
    >
      <div className="px-4 py-2 border-b border-gray-800 shrink-0">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          NOC Assistant
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2 min-h-0">
        {messages.length === 0 && (
          <p className="text-gray-600 text-xs">
            Ask about network state, anomalies, or RCA results...
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={[
              'text-xs rounded px-2 py-1.5 break-words whitespace-pre-wrap',
              m.role === 'user'      && 'bg-blue-950 text-blue-200 ml-6',
              m.role === 'assistant' && 'bg-gray-800 text-gray-200 mr-6',
              m.role === 'error'     && 'bg-red-950 text-red-300',
            ]
              .filter(Boolean)
              .join(' ')}
          >
            {m.text}
          </div>
        ))}
        {loading && (
          <div className="text-gray-500 text-xs animate-pulse">Thinking...</div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-gray-800 p-2 flex gap-2 shrink-0">
        <input
          className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-gray-500 placeholder-gray-600"
          placeholder="Ask the NOC assistant…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          className="px-3 py-1 text-xs bg-blue-700 hover:bg-blue-600 disabled:opacity-40 rounded text-white transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}
