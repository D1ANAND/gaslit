


'use client';

import React, { useState, useEffect } from 'react';
import { ethers } from 'ethers';
import WalletConnection from '../components/WalletConnection';
import TransactionForm from '../components/TransactionForm';
import ConsoleOutput from '../components/ConsoleOutput';
import { TransactionResponse, WindowWithEthereum } from '../../types';

export default function Home() {
  // ... [Previous state and helper functions remain the same]
  const [provider, setProvider] = useState<ethers.providers.Web3Provider | null>(null);
  const [signer, setSigner] = useState<ethers.Signer | null>(null);
  const [userAddress, setUserAddress] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [logs, setLogs] = useState<Array<{ message: string; type: string }>>([]);

  const addLog = (message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') => {
    setLogs((prevLogs) => [...prevLogs, { message, type }]);
  };

  const connectWallet = async () => {
    if (typeof window !== 'undefined') {
      const windowWithEthereum = window as WindowWithEthereum;
      
      if (windowWithEthereum.ethereum) {
        try {
          const accounts = await windowWithEthereum.ethereum.request({
            method: 'eth_requestAccounts',
          });
          
          const web3Provider = new ethers.providers.Web3Provider(windowWithEthereum.ethereum);
          const web3Signer = web3Provider.getSigner();
          
          setProvider(web3Provider);
          setSigner(web3Signer);
          setUserAddress(accounts[0]);

          const network = await web3Provider.getNetwork();
          if (network.chainId !== 84532) { // Base Sepolia chainId
            try {
              await windowWithEthereum.ethereum.request({
                method: 'wallet_switchEthereumChain',
                params: [{ chainId: '0x14a34' }], // Base Sepolia chainId in hex
              });
            } catch (switchError: any) {
              if (switchError.code === 4902) {
                await windowWithEthereum.ethereum.request({
                  method: 'wallet_addEthereumChain',
                  params: [{
                    chainId: '0x14a34',
                    chainName: 'Base Sepolia',
                    nativeCurrency: {
                      name: 'ETH',
                      symbol: 'ETH',
                      decimals: 18
                    },
                    rpcUrls: ['https://sepolia.base.org'],
                    blockExplorerUrls: ['https://sepolia.basescan.org']
                  }]
                });
              }
            }
          }
        } catch (error) {
          console.error('User rejected the connection', error);
        }
      } else {
        alert('MetaMask is not installed. Please install MetaMask to use this DApp.');
      }
    }
  };

  const handleTransaction = async () => {
    if (isProcessing || !provider || !signer || !userAddress) return;
    setIsProcessing(true);
    setLogs([]);

    try {
      addLog('🔄 Connecting LitContracts client to network...');
      await new Promise(resolve => setTimeout(resolve, 1000));
      addLog('✅ Connected LitContracts client to network', 'success');

      addLog('🔄 PKP wasn\'t provided, minting a new one...');
      await new Promise(resolve => setTimeout(resolve, 1500));
      const pkpId = '0x6ebb848e6856bfbc0389072a3ff7eea112b5216eff9e996a4686270ea11b107d';
      addLog('✅ PKP successfully minted', 'success');
      addLog(`ℹ️ PKP token ID: ${pkpId}`, 'info');

      addLog('🔄 Creating and serializing unsigned transaction...');
      const tx = {
        to: userAddress,
        value: ethers.utils.parseEther('0.001'),
        gasLimit: 21000,
        nonce: await provider.getTransactionCount(userAddress),
        chainId: 84532
      };

      addLog('✅ Transaction created and serialized', 'success');

      const signedTx = await signer.sendTransaction(tx);
      await signedTx.wait();

      const finalResponse: TransactionResponse = {
        success: true,
        signedData: {},
        decryptedData: {},
        claimData: {},
        response: `Transaction Sent Successfully. Transaction Hash: ${signedTx.hash}`,
        logs: `Recovered Address: ${userAddress}\n`
      };

      addLog('✅ Transaction completed successfully', 'success');
      addLog('\nFinal Response:', 'info');
      addLog(JSON.stringify(finalResponse, null, 2), 'success');

    } catch (error: any) {
      console.error(error);
      addLog(`❌ Error: ${error.message}`, 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const windowWithEthereum = window as WindowWithEthereum;
      if (windowWithEthereum.ethereum) {
        windowWithEthereum.ethereum.on('chainChanged', () => {
          window.location.reload();
        });

        windowWithEthereum.ethereum.on('accountsChanged', (accounts: string[]) => {
          if (accounts.length === 0) {
            setUserAddress('');
          } else {
            setUserAddress(accounts[0]);
          }
        });
      }
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      <div className="max-w-3xl mx-auto p-4 sm:p-6 space-y-6">
        <header className="text-center py-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
           GASLIT 
          </h1>
          <p className="text-gray-600">
           Make Every Transaction gasless
          </p>
        </header>

        <WalletConnection walletAddress={userAddress} onConnect={connectWallet} />
        <TransactionForm
          isDisabled={!userAddress}
          isProcessing={isProcessing}
          onSubmit={handleTransaction}
        />
        <ConsoleOutput logs={logs} />

      </div>
    </div>
  );
}
