import React, { useState, useEffect, useRef } from 'react';

interface Message {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
}

export default function App() {
  // --- Inputs ---
  const [resumeLatex, setResumeLatex] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  
  // --- State ---
  const [score, setScore] = useState<number | null>(null);
  const [scoreLoading, setScoreLoading] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);
  const [launchLoading, setLaunchLoading] = useState(false);
  
  // --- Agent Chat ---
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isAgentWorking, setIsAgentWorking] = useState(false);
  const [currentLatex, setCurrentLatex] = useState('');
  const [ws, setWs] = useState<WebSocket | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleScore = async () => {
    if (!resumeLatex || !jobDescription) return;
    setScoreLoading(true);
    try {
      const response = await fetch('http://localhost:8000/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resume_text: resumeLatex, job_description: jobDescription }),
      });
      const data = await response.json();
      setScore(data.confidence_score);
    } catch (err) {
      console.error(err);
      alert('Failed to get score. Is the backend running?');
    } finally {
      setScoreLoading(false);
    }
  };

  const startSession = () => {
    if (!resumeLatex || !jobDescription) return;
    if (ws) ws.close(); // Close existing if any

    setLaunchLoading(true);
    setChatMessages([]);
    setCurrentLatex(resumeLatex);
    
    const socket = new WebSocket('ws://localhost:8000/ws/agent');
    
    socket.onopen = () => {
      socket.send(JSON.stringify({
        type: 'init',
        latex_code: resumeLatex,
        job_description: jobDescription,
      }));
      setSessionActive(true);
      setLaunchLoading(false);
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'system':
          setChatMessages(prev => [...prev, { role: 'system', content: data.content }]);
          break;
        case 'message':
          setChatMessages(prev => [...prev, { role: 'assistant', content: data.content }]);
          setIsAgentWorking(false);
          break;
        case 'thinking':
          setIsAgentWorking(true);
          break;
        case 'tool_call':
          setChatMessages(prev => [...prev, { role: 'tool', content: `🛠️ ${data.tool}...` }]);
          break;
        case 'latex_update':
          setCurrentLatex(data.content);
          setResumeLatex(data.content); // Sync back to input
          break;
        case 'error':
          setChatMessages(prev => [...prev, { role: 'system', content: `❌ Error: ${data.content}` }]);
          setIsAgentWorking(false);
          setLaunchLoading(false);
          break;
        case 'done':
          setIsAgentWorking(false);
          break;
      }
    };

    socket.onclose = () => {
      setIsAgentWorking(false);
      setLaunchLoading(false);
    };

    socket.onerror = () => {
      setLaunchLoading(false);
      alert('WebSocket connection failed.');
    };

    setWs(socket);
  };

  const sendMessage = () => {
    if (!chatInput.trim() || !ws) return;
    const userMsg = chatInput.trim();
    setChatMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    ws.send(JSON.stringify({ type: 'message', content: userMsg }));
    setChatInput('');
    setIsAgentWorking(true);
  };

  const resetSession = () => {
    ws?.close();
    setWs(null);
    setSessionActive(false);
    setChatMessages([]);
    setScore(null);
  };

  return (
    <div className="h-screen flex flex-col bg-[#03030a] text-white overflow-hidden font-sans">
      {/* Header */}
      <header className="h-14 flex-shrink-0 border-b border-white/5 bg-black/40 backdrop-blur-md flex items-center px-6 justify-between z-50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center font-bold text-lg shadow-lg">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight">JobFit AI Dashboard</h1>
            <p className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">v1.0 Production</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          {score !== null && (
            <div className="flex items-center gap-3 glass-panel px-4 py-1 rounded-full border-white/10">
              <span className="text-[9px] font-bold text-gray-500 uppercase tracking-widest">Match Score</span>
              <span className={`text-sm font-black ${score > 70 ? 'text-green-400' : score > 40 ? 'text-yellow-400' : 'text-red-400'}`}>
                {score.toFixed(1)}%
              </span>
            </div>
          )}
          {sessionActive && (
            <button onClick={resetSession} className="text-[9px] font-bold text-red-400 border border-red-400/20 px-3 py-1 rounded-md hover:bg-red-400/10 transition-all">
              RESET DASHBOARD
            </button>
          )}
        </div>
      </header>

      {/* Main Dashboard Layout */}
      <main className="flex-1 flex overflow-hidden relative">
        {/* Background Effects */}
        <div className="absolute top-[-10%] left-[-10%] w-[30%] h-[30%] bg-purple-600/5 rounded-full blur-[100px] pointer-events-none"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[30%] h-[30%] bg-blue-600/5 rounded-full blur-[100px] pointer-events-none"></div>

        {/* --- COLUMN 1: INPUTS (25%) --- */}
        <div className="w-[350px] flex-shrink-0 border-r border-white/5 flex flex-col bg-black/10">
          <div className="p-4 border-b border-white/5 bg-white/5 flex items-center justify-between">
            <h3 className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Context & JD</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar">
            <div className="space-y-2">
              <label className="text-[9px] font-bold text-gray-500 uppercase tracking-widest">Target Job Description</label>
              <textarea 
                className="w-full h-48 glass-panel glass-input rounded-xl p-3 text-xs text-gray-300 resize-none focus:h-80 transition-all duration-300"
                placeholder="Paste JD here..."
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
              />
            </div>
            
            <div className="pt-4 border-t border-white/5 space-y-4">
              <button 
                onClick={handleScore} 
                disabled={scoreLoading || !resumeLatex || !jobDescription}
                className="w-full py-2.5 glass-panel rounded-xl text-[10px] font-bold uppercase tracking-widest hover:bg-white/5 transition-all"
              >
                {scoreLoading ? 'Scoring...' : '📊 Calculate Score'}
              </button>
              
              {!sessionActive && (
                <button 
                  onClick={startSession} 
                  disabled={launchLoading || !resumeLatex || !jobDescription}
                  className="w-full py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl text-[10px] font-bold uppercase tracking-widest shadow-xl hover:brightness-110 active:scale-95 transition-all"
                >
                  {launchLoading ? '⌛ Initializing...' : '🚀 Launch AI Agent'}
                </button>
              )}
            </div>

            {score !== null && (
              <div className="glass-panel p-4 rounded-xl border-white/10 text-center animate-in zoom-in-95 duration-500">
                <div className="text-3xl font-black mb-1">{score.toFixed(0)}%</div>
                <div className="text-[10px] font-bold uppercase tracking-widest text-gray-500">ATS Compatibility</div>
              </div>
            )}
          </div>
        </div>

        {/* --- COLUMN 2: AGENT CHAT (35%) --- */}
        <div className="w-[450px] flex-shrink-0 border-r border-white/5 flex flex-col bg-black/20">
          <div className="h-12 border-b border-white/5 flex items-center justify-between px-4 bg-white/5">
            <div className="flex items-center gap-2">
              <div className={`w-1.5 h-1.5 rounded-full ${isAgentWorking ? 'bg-purple-500 animate-pulse' : 'bg-green-500'}`}></div>
              <span className="text-[9px] font-bold uppercase tracking-widest text-gray-400">AI Assistant {isAgentWorking ? '(Thinking)' : '(Ready)'}</span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
            {!sessionActive ? (
              <div className="h-full flex flex-col items-center justify-center text-center px-6 space-y-4 opacity-40">
                <div className="w-12 h-12 rounded-full border-2 border-dashed border-gray-600 flex items-center justify-center">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path></svg>
                </div>
                <p className="text-xs italic">Launch the AI Agent to start the conversation and optimize your resume.</p>
              </div>
            ) : (
              chatMessages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[90%] rounded-xl px-3.5 py-2 text-xs leading-relaxed ${
                    msg.role === 'user' ? 'bg-blue-600 text-white' : 
                    msg.role === 'tool' ? 'bg-white/5 text-blue-300 font-mono text-[9px] border border-blue-400/20' :
                    msg.role === 'system' ? 'text-gray-500 text-[10px] italic w-full text-center' : 'glass-panel text-gray-200'
                  }`}>
                    {msg.content}
                  </div>
                </div>
              ))
            )}
            {isAgentWorking && (
              <div className="flex justify-start">
                <div className="glass-panel px-3 py-1.5 rounded-lg flex gap-1 items-center">
                  <span className="w-1 h-1 bg-purple-400 rounded-full animate-bounce"></span>
                  <span className="w-1 h-1 bg-purple-400 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                  <span className="w-1 h-1 bg-purple-400 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="p-4 bg-white/5 border-t border-white/5">
            <div className={`flex gap-2 bg-black/40 p-1.5 rounded-xl border ${sessionActive ? 'border-white/10' : 'border-white/5 opacity-30'} transition-all`}>
              <input 
                className="flex-1 bg-transparent px-3 py-2 text-xs focus:outline-none disabled:cursor-not-allowed"
                placeholder={sessionActive ? "Ask the agent to improve..." : "Launch agent first..."}
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                disabled={!sessionActive || isAgentWorking}
              />
              <button 
                onClick={sendMessage} 
                disabled={!sessionActive || isAgentWorking || !chatInput.trim()}
                className="bg-white text-black px-4 py-2 rounded-lg font-bold text-[10px] uppercase tracking-widest disabled:opacity-40"
              >
                Send
              </button>
            </div>
          </div>
        </div>

        {/* --- COLUMN 3: LATEX EDITOR (FLEX-1) --- */}
        <div className="flex-1 flex flex-col bg-[#020207]">
          <div className="h-12 border-b border-white/5 flex items-center justify-between px-6 bg-white/5">
            <span className="text-[9px] font-bold uppercase tracking-widest text-gray-400">LaTeX Source Editor</span>
            <div className="flex gap-4">
              <button onClick={() => navigator.clipboard.writeText(currentLatex || resumeLatex)} className="text-[9px] font-bold text-gray-500 hover:text-white uppercase tracking-widest">Copy Code</button>
            </div>
          </div>
          <div className="flex-1 overflow-hidden relative">
            <textarea 
              className="absolute inset-0 w-full h-full bg-transparent p-8 text-gray-300 font-mono text-xs leading-relaxed resize-none focus:outline-none custom-scrollbar"
              value={sessionActive ? currentLatex : resumeLatex}
              onChange={(e) => {
                if (sessionActive) {
                   // In session, let the agent control it mainly, but allow user edit
                   setCurrentLatex(e.target.value);
                } else {
                   setResumeLatex(e.target.value);
                }
              }}
              placeholder="% Paste your LaTeX resume source code here..."
            />
          </div>
          {/* Editor Footer */}
          <div className="h-8 border-t border-white/5 flex items-center px-6 bg-black/40">
             <span className="text-[8px] text-gray-600 uppercase font-bold tracking-widest">
               {sessionActive ? 'Agentic Mode Active' : 'Waiting for Input'} | Chars: {(sessionActive ? currentLatex : resumeLatex).length}
             </span>
          </div>
        </div>
      </main>
    </div>
  );
}
