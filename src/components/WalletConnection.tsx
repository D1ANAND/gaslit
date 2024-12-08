// components/WalletConnection.tsx
import React from 'react';
import { Wallet, ExternalLink } from 'lucide-react';
 import { Alert, AlertDescription } from '@/components/ui/alert';

interface WalletConnectionProps {
  walletAddress: string;
  onConnect: () => Promise<void>;
}

const WalletConnection: React.FC<WalletConnectionProps> = ({ walletAddress, onConnect }) => {
  const shortenAddress = (address: string) => {
    if (!address) return '';
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  return (
    <div className="bg-white rounded-xl p-6 shadow-xl mb-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="bg-orange-100 p-2 rounded-lg">
            <Wallet className="w-6 h-6 text-orange-600" />
          </div>
          <h1 className="text-xl font-semibold text-gray-800">dApp</h1>
        </div>
        <a 
          href="https://sepolia.basescan.org" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 transition-colors"
        >
          <ExternalLink size={16} />
          Explorer
        </a>
      </div>
      
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
        <button
          onClick={onConnect}
          className="bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white px-6 py-2.5 rounded-lg transition-all duration-200 shadow-md hover:shadow-lg flex items-center gap-2"
        >
          <Wallet size={18} />
          {walletAddress ? 'Change Wallet' : 'Connect Wallet'}
        </button>
        
        {walletAddress && (
          <div className="flex items-center gap-2 bg-gray-50 px-4 py-2 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-sm text-gray-600">Connected: </span>
            <span className="font-mono text-sm">{shortenAddress(walletAddress)}</span>
          </div>
        )}
      </div>

      {!walletAddress && (
        <Alert className="mt-4">
          <AlertDescription>
            Connect your wallet .
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default WalletConnection;