import FraudScanner from "@/components/FraudScanner";
import { ShieldCheck, Github, Lock } from "lucide-react";

export default function Home() {
  return (
    <main className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black text-white selection:bg-blue-500/30">

      {/* Navbar */}
      <nav className="border-b border-slate-800/50 bg-slate-950/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
              <ShieldCheck className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-400">
              Sentinel
            </span>
          </div>

          <div className="flex items-center gap-6 text-sm font-medium text-slate-400">
            <a href="https://sentinel-08u6.onrender.com/docs" target="_blank" className="hover:text-blue-400 transition-colors flex items-center gap-2">
              <Lock className="w-4 h-4" /> API Docs
            </a>
            <a href="https://github.com/hammadmunir959/Sentinel" target="_blank" className="hover:text-white transition-colors">
              <Github className="w-5 h-5" />
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative pt-20 pb-12 px-6">
        <div className="max-w-4xl mx-auto text-center space-y-4 mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold mb-4">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
            </span>
            System Operational
          </div>

          <h1 className="text-5xl md:text-6xl font-bold tracking-tight text-white mb-6">
            AI-Powered <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-cyan-400">
              Fraud Detection
            </span>
          </h1>

          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Real-time transaction analysis using XGBoost with 97% ROC-AUC accuracy.
            Detect anomalies and secure payments instantly.
          </p>
        </div>

        {/* Interactive Scanner */}
        <FraudScanner />
      </div>

      {/* Footer */}
      <footer className="border-t border-slate-800/50 mt-20 py-8 text-center text-slate-600 text-sm">
        <p>Sentinel Project © 2026. Powered by XGBoost & FastAPI.</p>
      </footer>
    </main>
  );
}
