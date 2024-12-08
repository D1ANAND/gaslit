// types.ts
import { ethers } from 'ethers';

export interface TransactionResponse {
  success: boolean;
  signedData: Record<string, any>;
  decryptedData: Record<string, any>;
  claimData: Record<string, any>;
  response: string;
  logs: string;
}

export interface WindowWithEthereum extends Window {
  ethereum?: any;
}
