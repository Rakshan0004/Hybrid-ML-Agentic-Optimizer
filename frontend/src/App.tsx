import { useState, useRef, useEffect } from 'react';

// Types for WebSocket events
interface AgentEvent {
  type: 'thinking' | 'tool_call' | 'tool_result' | 'latex_update' | 'message' | 'done' | 'error' | 'system';
  content?: string;
  tool?: string;
  args?: Record<string, string>;
  result?: string;
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  toolName?: string;
  toolArgs?: Record<string, string>;
}

function App() {
  // ── Input State ──
  const [resumeLatex, setResumeLatex] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  
  // ── Session State ──
  const [sessionActive, setSessionActive] = useState(false);
  const [currentLatex, setCurrentLatex] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isAgentWorking, setIsAgentWorking] = useState(false);
  
  // ── Score State ──
  const [score, setScore] = useState<number | null>(null);
  const [scoreLoading, setScoreLoading] = useState(false);
  
  // ── WebSocket ──
  const wsRef = useRef<WebSocket | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // ── Quick Score (no agent, just API call) ──
  const handleScore = async () => {
    setScoreLoading(true);
    try {
      const response = await fetch('http://localhost:8000/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resume_text: resumeLatex, job_description: jobDescription }),
      });
      const data = await response.json();
      if (response.ok) {
        setScore(data.confidence_score);
      } else {
        alert(data.detail || 'Error scoring resume');
      }
    } catch {
      alert('Failed to connect to API. Is the backend running?');
    }
    setScoreLoading(false);
  };

  // ── Start Agent Session ──
  const startSession = () => {
    if (!resumeLatex || !jobDescription) {
      alert('Please paste both your LaTeX resume and the job description.');
      return;
    }

    const ws = new WebSocket('ws://localhost:8000/ws/agent');
    wsRef.current = ws;

    ws.onopen = () => {
      // Send init message with resume and JD
      ws.send(JSON.stringify({
        type: 'init',
        latex_code: resumeLatex,
        job_description: jobDescription,
      }));
      setSessionActive(true);
      setCurrentLatex(resumeLatex);
      setChatMessages([]);
    };

    ws.onmessage = (event) => {
      const data: AgentEvent = JSON.parse(event.data);
      
      switch (data.type) {
        case 'system':
          setChatMessages(prev => [...prev, { role: 'system', content: data.content || '' }]);
          break;
          
        case 'thinking':
          setChatMessages(prev => [...prev, { role: 'assistant', content: `💭 ${data.content}` }]);
          break;
          
        case 'tool_call':
          setChatMessages(prev => [...prev, { 
            role: 'tool', 
            content: `🔧 Calling \`${data.tool}\``,
            toolName: data.tool,
            toolArgs: data.args
          }]);
          break;
          
        case 'tool_result':
          setChatMessages(prev => [...prev, { 
            role: 'tool', 
            content: `✅ ${data.tool} result: ${data.result}` 
          }]);
          break;
          
        case 'latex_update':
          setCurrentLatex(data.content || '');
          setChatMessages(prev => [...prev, { role: 'system', content: '📝 LaTeX document updated!' }]);
          break;
          
        case 'message':
          setChatMessages(prev => [...prev, { role: 'assistant', content: data.content || '' }]);
          setIsAgentWorking(false);
          break;
          
        case 'error':
          setChatMessages(prev => [...prev, { role: 'system', content: `❌ Error: ${data.content}` }]);
          setIsAgentWorking(false);
          break;
          
        case 'done':
          setIsAgentWorking(false);
          break;
      }
    };

    ws.onerror = () => {
      alert('WebSocket error. Is the backend running?');
      setIsAgentWorking(false);
    };

    ws.onclose = () => {
      setSessionActive(false);
      setIsAgentWorking(false);
    };
  };

  // ── Send Chat Message ──
  const sendMessage = () => {
    if (!chatInput.trim() || !wsRef.current || isAgentWorking) return;
    
    const message = chatInput.trim();
    setChatMessages(prev => [...prev, { role: 'user', content: message }]);
    setChatInput('');
    setIsAgentWorking(true);
    
    wsRef.current.send(JSON.stringify({
      type: 'message',
      content: message,
    }));
  };

  // ── Disconnect Session ──
  const endSession = () => {
    wsRef.current?.close();
    setSessionActive(false);
  };

  // ──────────────────────────────────────────
  // RENDER
  // ──────────────────────────────────────────
  return (
    <div className="min-h-screen bg-[#0a0a1a] text-white font-sans">
      {/* Header */}
      <header className="border-b border-gray-800 bg-[#0d0d20]/80 backdrop-blur-lg sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center font-bold text-lg">J</div>
            <div>
              <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">JobFit Agent</h1>
              <p className="text-xs text-gray-500">AI-Powered Resume Optimizer</p>
            </div>
          </div>
          {score !== null && (
            <div className="flex items-center gap-3 bg-gray-800/50 px-4 py-2 rounded-full border border-gray-700">
              <span className="text-sm text-gray-400">Match Score:</span>
              <span className={`text-lg font-bold ${score > 70 ? 'text-green-400' : score > 40 ? 'text-yellow-400' : 'text-red-400'}`}>
                {score.toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      </header>

      {!sessionActive ? (
        /* ──────── SETUP VIEW ──────── */
        <div className="max-w-5xl mx-auto px-6 py-12">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-3">Get Started</h2>
            <p className="text-gray-400">Paste your LaTeX resume and the target job description to begin.</p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                LaTeX Resume Code
              </label>
              <textarea 
                className="w-full h-72 bg-[#12122a] border border-gray-700/50 rounded-xl p-4 text-gray-300 focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 focus:outline-none font-mono text-sm resize-none transition-all"
                placeholder={"\\documentclass{article}\n\\begin{document}\n  % Your resume here...\n\\end{document}"}
                value={resumeLatex}
                onChange={(e) => setResumeLatex(e.target.value)}
              />
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                Job Description
              </label>
              <textarea 
                className="w-full h-72 bg-[#12122a] border border-gray-700/50 rounded-xl p-4 text-gray-300 focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 focus:outline-none text-sm resize-none transition-all"
                placeholder="Paste the full job description here..."
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
              />
            </div>
          </div>

          <div className="flex gap-4 mt-8 justify-center">
            <button 
              onClick={handleScore}
              disabled={scoreLoading || !resumeLatex || !jobDescription}
              className="px-8 py-3 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-xl font-medium transition disabled:opacity-40"
            >
              {scoreLoading ? '⏳ Scoring...' : '📊 Quick Score'}
            </button>
            <button 
              onClick={startSession}
              disabled={!resumeLatex || !jobDescription}
              className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 rounded-xl font-bold shadow-lg shadow-purple-500/20 transition transform active:scale-95 disabled:opacity-40"
            >
              🚀 Start AI Agent Session
            </button>
          </div>

          {score !== null && (
            <div className="mt-8 text-center">
              <div className="inline-flex items-center gap-4 bg-gray-800/50 px-8 py-4 rounded-2xl border border-gray-700">
                <div className="relative w-20 h-20">
                  <svg className="w-full h-full transform -rotate-90">
                    <circle cx="40" cy="40" r="34" stroke="currentColor" strokeWidth="6" fill="transparent" className="text-gray-700" />
                    <circle cx="40" cy="40" r="34" stroke="currentColor" strokeWidth="6" fill="transparent"
                      strokeDasharray={2 * Math.PI * 34}
                      strokeDashoffset={2 * Math.PI * 34 * (1 - score / 100)}
                      className={`${score > 70 ? 'text-green-500' : score > 40 ? 'text-yellow-500' : 'text-red-500'} transition-all duration-1000`}
                    />
                  </svg>
                  <span className="absolute inset-0 flex items-center justify-center text-lg font-bold">{score.toFixed(0)}%</span>
                </div>
                <div className="text-left">
                  <p className="font-semibold">{score > 70 ? 'Strong Match' : score > 40 ? 'Moderate Match' : 'Needs Work'}</p>
                  <p className="text-sm text-gray-400">Start an agent session to improve it!</p>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* ──────── AGENT SESSION VIEW ──────── */
        <div className="flex h-[calc(100vh-73px)]">
          {/* Left Panel: Chat */}
          <div className="w-1/2 flex flex-col border-r border-gray-800">
            <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between bg-[#0d0d20]/50">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isAgentWorking ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'}`}></div>
                <span className="text-sm font-medium text-gray-300">
                  {isAgentWorking ? 'Agent is working...' : 'Agent Ready'}
                </span>
              </div>
              <button onClick={endSession} className="text-xs text-red-400 hover:text-red-300 transition">
                End Session
              </button>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
              {chatMessages.map((msg, i) => (
                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm whitespace-pre-wrap ${
                    msg.role === 'user' 
                      ? 'bg-blue-600/30 border border-blue-500/30 text-blue-100' 
                      : msg.role === 'tool'
                      ? 'bg-amber-900/20 border border-amber-700/30 text-amber-200 font-mono text-xs'
                      : msg.role === 'system'
                      ? 'bg-gray-800/50 border border-gray-700/30 text-gray-400 text-xs'
                      : 'bg-gray-800/60 border border-gray-700/40 text-gray-200'
                  }`}>
                    {msg.content}
                  </div>
                </div>
              ))}
              {isAgentWorking && (
                <div className="flex justify-start">
                  <div className="bg-gray-800/60 border border-gray-700/40 rounded-xl px-4 py-3 text-sm text-gray-400">
                    <span className="inline-flex gap-1">
                      <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></span>
                      <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></span>
                      <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></span>
                    </span>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input */}
            <div className="px-4 py-3 border-t border-gray-800 bg-[#0d0d20]/50">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                  placeholder="e.g. 'Improve my resume to 90%' or 'Optimize the skills section'"
                  disabled={isAgentWorking}
                  className="flex-1 bg-[#12122a] border border-gray-700/50 rounded-xl px-4 py-3 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 disabled:opacity-50"
                />
                <button
                  onClick={sendMessage}
                  disabled={isAgentWorking || !chatInput.trim()}
                  className="px-5 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl font-medium text-sm transition active:scale-95 disabled:opacity-40"
                >
                  Send
                </button>
              </div>
              <div className="flex gap-2 mt-2">
                {['Improve to 90%', 'Optimize skills section', 'Score my resume', 'Rewrite summary'].map(q => (
                  <button 
                    key={q}
                    onClick={() => { setChatInput(q); }}
                    className="text-xs px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-gray-400 transition"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right Panel: Live LaTeX Viewer */}
          <div className="w-1/2 flex flex-col">
            <div className="px-4 py-3 border-b border-gray-800 bg-[#0d0d20]/50 flex items-center justify-between">
              <span className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-green-400"></span>
                Live LaTeX Preview
              </span>
              <button 
                onClick={() => navigator.clipboard.writeText(currentLatex)}
                className="text-xs text-gray-400 hover:text-white transition px-3 py-1 border border-gray-700 rounded-lg"
              >
                📋 Copy
              </button>
            </div>
            <pre className="flex-1 overflow-auto p-4 bg-[#0a0a18] text-gray-300 font-mono text-xs leading-relaxed whitespace-pre-wrap">
              {currentLatex || 'LaTeX content will appear here once the session starts...'}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
