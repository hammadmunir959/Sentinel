"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Shield, AlertTriangle, CheckCircle, Activity, Server, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Sample data
const LEGIT_SAMPLE = {
    Time: 10000,
    Amount: 15.50,
    V1: -0.5, V2: 0.2, V3: 1.1, V4: -0.5, V5: -0.2, V6: 0.1, V7: 0.2, V8: 0.1, V9: -0.1,
    V10: 0.1, V11: -0.2, V12: 0.3, V13: 0.1, V14: -0.1, V15: 0.1, V16: -0.2, V17: 0.1,
    V18: 0.1, V19: -0.1, V20: 0.1, V21: -0.1, V22: 0.2, V23: 0.1, V24: -0.1, V25: -0.1,
    V26: 0.1, V27: 0.1, V28: 0.1
};

const FRAUD_SAMPLE = {
    Time: 406,
    Amount: 0.0,
    V1: -2.31, V2: 1.95, V3: -1.6, V4: 3.99, V5: -0.52, V6: -1.42, V7: -2.53, V8: 1.39,
    V9: -2.77, V10: -2.77, V11: 3.2, V12: -2.89, V13: -0.59, V14: -4.28, V15: 0.38,
    V16: -1.14, V17: -2.83, V18: -0.01, V19: 0.41, V20: 0.12, V21: 0.51, V22: -0.03,
    V23: -0.46, V24: 0.32, V25: 0.04, V26: 0.17, V27: 0.26, V28: -0.14
};

export default function FraudScanner() {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<any>(null);
    const [currentData, setCurrentData] = useState<any>(null);
    const [error, setError] = useState("");

    const handlePredict = async (data: any) => {
        setLoading(true);
        setError("");
        setResult(null);
        setCurrentData(data);

        // Fill in missing V columns with 0 if needed (just in case)
        const payload = { ...data };
        for (let i = 1; i <= 28; i++) {
            if (payload[`V${i}`] === undefined) payload[`V${i}`] = 0;
        }

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!response.ok) throw new Error("API Connection Failed");

            const resData = await response.json();
            // Artificial delay for "scanning" effect
            setTimeout(() => {
                setResult(resData);
                setLoading(false);
            }, 800);
        } catch (err) {
            setError("Failed to connect to Sentinel API");
            setLoading(false);
        }
    };

    return (
        <div className="w-full max-w-4xl mx-auto p-4 grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Control Panel */}
            <div className="space-y-6">
                <div className="bg-slate-900/50 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl">
                    <div className="flex items-center gap-3 mb-6">
                        <Activity className="w-6 h-6 text-blue-400" />
                        <h2 className="text-xl font-bold text-white">Transaction Simulator</h2>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-6">
                        <button
                            onClick={() => handlePredict(LEGIT_SAMPLE)}
                            disabled={loading}
                            className="p-4 bg-emerald-500/10 border border-emerald-500/30 hover:bg-emerald-500/20 text-emerald-400 rounded-xl transition-all flex flex-col items-center gap-2 group"
                        >
                            <Shield className="w-8 h-8 group-hover:scale-110 transition-transform" />
                            <span className="font-semibold">Simulate Legit</span>
                        </button>
                        <button
                            onClick={() => handlePredict(FRAUD_SAMPLE)}
                            disabled={loading}
                            className="p-4 bg-red-500/10 border border-red-500/30 hover:bg-red-500/20 text-red-400 rounded-xl transition-all flex flex-col items-center gap-2 group"
                        >
                            <Zap className="w-8 h-8 group-hover:scale-110 transition-transform" />
                            <span className="font-semibold">Simulate Fraud</span>
                        </button>
                    </div>

                    <div className="bg-slate-950 rounded-lg p-4 font-mono text-xs text-slate-400 overflow-hidden relative">
                        <div className="absolute top-2 right-2 text-slate-600 flex items-center gap-1">
                            <Server className="w-3 h-3" /> Payload
                        </div>
                        <pre className="overflow-x-auto">
                            {currentData
                                ? JSON.stringify(currentData, null, 2)
                                : "// Select a simulation to view payload"}
                        </pre>
                    </div>
                </div>
            </div>

            {/* Analysis Panel */}
            <div className="relative">
                <div className="bg-slate-900/50 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl h-full flex flex-col items-center justify-center min-h-[400px]">

                    {loading ? (
                        <div className="flex flex-col items-center gap-4">
                            <div className="relative w-24 h-24">
                                <motion.div
                                    className="absolute inset-0 border-4 border-blue-500/30 rounded-full"
                                    animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                                    transition={{ repeat: Infinity, duration: 2 }}
                                />
                                <motion.div
                                    className="absolute inset-0 border-t-4 border-blue-500 rounded-full"
                                    animate={{ rotate: 360 }}
                                    transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                                />
                            </div>
                            <p className="text-blue-400 font-mono animate-pulse">Analyzing Pattern...</p>
                        </div>
                    ) : result ? (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="text-center w-full"
                        >
                            <div className={cn(
                                "w-32 h-32 mx-auto rounded-full flex items-center justify-center mb-6",
                                result.is_fraud ? "bg-red-500/20 text-red-500" : "bg-emerald-500/20 text-emerald-500"
                            )}>
                                {result.is_fraud ? (
                                    <AlertTriangle className="w-16 h-16" />
                                ) : (
                                    <CheckCircle className="w-16 h-16" />
                                )}
                            </div>

                            <h3 className="text-3xl font-bold text-white mb-2">
                                {result.is_fraud ? "FRAUD DETECTED" : "LEGITIMATE"}
                            </h3>

                            <div className="flex items-center justify-center gap-2 mb-8">
                                <span className="text-slate-400">Confidence Score:</span>
                                <span className={cn(
                                    "text-xl font-mono font-bold",
                                    result.is_fraud ? "text-red-400" : "text-emerald-400"
                                )}>
                                    {(result.fraud_probability * 100).toFixed(2)}%
                                </span>
                            </div>

                            <div className="grid grid-cols-2 gap-4 text-left bg-slate-950/50 p-4 rounded-xl text-sm">
                                <div>
                                    <div className="text-slate-500">Prediction Class</div>
                                    <div className="text-white font-mono">{result.prediction}</div>
                                </div>
                                <div>
                                    <div className="text-slate-500">Processing Time</div>
                                    <div className="text-white font-mono">~45ms</div>
                                </div>
                            </div>
                        </motion.div>
                    ) : (
                        <div className="text-center text-slate-500">
                            <Shield className="w-16 h-16 mx-auto mb-4 opacity-20" />
                            <p>Ready to analyze transactions</p>
                        </div>
                    )}

                    {error && (
                        <div className="mt-4 p-4 bg-red-900/20 border border-red-900/50 rounded-lg text-red-400 text-sm flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4" />
                            {error}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
