// components/ConsoleOutput.tsx
import React from 'react';
import { Terminal } from 'lucide-react';

interface ConsoleOutputProps {
  logs: Array<{ message: string; type: 'info' | 'success' | 'warning' | 'error' }>;
}

const ConsoleOutput: React.FC<ConsoleOutputProps> = ({ logs }) => {
  const getTypeClass = (type: string) => {
    const classes = {
      info: 'text-blue-400',
      success: 'text-emerald-400',
      warning: 'text-amber-400',
      error: 'text-red-400',
    };
    return classes[type as keyof typeof classes] || classes.info;
  };

  return (
    <div className="bg-slate-900 rounded-xl p-6 mt-6 shadow-xl">
      <div className="flex items-center gap-2 mb-4 text-slate-400">
        <Terminal size={20} />
        <h3 className="font-medium">Console Output</h3>
      </div>
      <div className="font-mono text-sm overflow-y-auto max-h-96 custom-scrollbar">
        {logs.map((log, index) => (
          <div key={index} className={`py-1 leading-relaxed ${getTypeClass(log.type)}`}>
            {log.message}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ConsoleOutput;