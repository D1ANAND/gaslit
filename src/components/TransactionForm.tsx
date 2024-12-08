import React, { useState } from 'react';
import { Send, Loader2 } from 'lucide-react';

interface TransactionFormProps {
  isDisabled: boolean;
  isProcessing: boolean;
  onSubmit: () => Promise<void>;
}

const TransactionForm: React.FC<TransactionFormProps> = ({
  isDisabled,
  isProcessing,
  onSubmit,
}) => {
  const [amount, setAmount] = useState('');

  return (
    <div className="bg-white rounded-xl p-6 shadow-xl">
      <div className="flex items-center gap-3 mb-6">
        <div className="bg-blue-100 p-2 rounded-lg">
          <Send className="w-6 h-6 text-blue-600" />
        </div>
        <h2 className="text-xl font-semibold text-gray-800">Send Prompt</h2>
      </div>

      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <div className="relative flex-grow">
            <input
              id="amount"
              type="text"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-orange-500 transition-all"
              placeholder="Enter a prompt..."
            />
          </div>
        </div>

        <button
          onClick={onSubmit}
          disabled={isDisabled || isProcessing}
          className="w-full bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 disabled:from-gray-400 disabled:to-gray-500 text-white px-6 py-3 rounded-lg transition-all duration-200 shadow-md hover:shadow-lg flex items-center justify-center gap-2"
        >
          {isProcessing ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Processing...
            </>
          ) : (
            <>
             
              Send
            </>
          )}
        </button>

        <p className="text-sm text-gray-500 text-center">
          {/* Transaction will be processed on Base Sepolia testnet */}
        </p>
      </div>
    </div>
  );
};

export default TransactionForm;
