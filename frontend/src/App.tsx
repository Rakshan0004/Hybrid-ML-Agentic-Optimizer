import { useState } from 'react';

function App() {
  const [resumeText, setResumeText] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [score, setScore] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const handleScore = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resume_text: resumeText, job_description: jobDescription }),
      });
      const data = await response.json();
      if (response.ok) {
        setScore(data.confidence_score);
      } else {
        alert(data.detail || 'Error scoring resume');
      }
    } catch (error) {
      console.error(error);
      alert('Failed to connect to API');
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8 font-sans">
      <div className="max-w-6xl mx-auto">
        <header className="mb-12">
          <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
            AI Resume Matcher & Improver
          </h1>
          <p className="text-gray-400 mt-2">Analyze and optimize your LaTeX resume against any Job Description.</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-xl">
              <label className="block text-sm font-medium text-gray-300 mb-2">Job Description</label>
              <textarea 
                className="w-full h-40 bg-gray-900 border border-gray-700 rounded-lg p-4 text-gray-300 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                placeholder="Paste the job description here..."
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
              />
            </div>

            <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-xl">
              <label className="block text-sm font-medium text-gray-300 mb-2">Resume (Text or LaTeX)</label>
              <textarea 
                className="w-full h-64 bg-gray-900 border border-gray-700 rounded-lg p-4 text-gray-300 focus:ring-2 focus:ring-purple-500 focus:outline-none font-mono text-sm"
                placeholder="\documentclass{article}... or paste plain text"
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
              />
            </div>

            <button 
              onClick={handleScore}
              disabled={loading || !resumeText || !jobDescription}
              className="w-full py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 rounded-xl font-bold text-lg shadow-lg transform transition active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Analyzing...' : 'Calculate Match Score'}
            </button>
          </div>

          <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-xl flex flex-col items-center justify-center min-h-[400px]">
            {score !== null ? (
              <div className="text-center">
                <h2 className="text-2xl font-semibold text-gray-300 mb-6">Confidence Score</h2>
                <div className="relative w-64 h-64 flex items-center justify-center rounded-full bg-gray-900 border-8 border-gray-800 shadow-[inset_0_0_20px_rgba(0,0,0,0.5)]">
                  {/* Simple CSS gauge representation */}
                  <svg className="absolute w-full h-full transform -rotate-90">
                    <circle cx="128" cy="128" r="120" stroke="currentColor" strokeWidth="16" fill="transparent" className="text-gray-700" />
                    <circle 
                      cx="128" cy="128" r="120" stroke="currentColor" strokeWidth="16" fill="transparent" 
                      strokeDasharray={2 * Math.PI * 120}
                      strokeDashoffset={2 * Math.PI * 120 * (1 - score / 100)}
                      className="text-blue-500 transition-all duration-1000 ease-out" 
                    />
                  </svg>
                  <span className="text-6xl font-black text-white relative z-10">{score.toFixed(1)}<span className="text-2xl text-gray-500">%</span></span>
                </div>
                <p className="mt-8 text-gray-400">
                  {score > 80 ? "Excellent match! You're a strong candidate." : score > 50 ? "Moderate match. Consider optimizing keywords." : "Low match. Major revisions recommended."}
                </p>
                
                {/* Improvement Section (Placeholder for Phase 3) */}
                <div className="mt-12 w-full pt-8 border-t border-gray-700">
                  <h3 className="text-lg font-medium text-left mb-4">Agentic AI Improver</h3>
                  <div className="flex items-center space-x-4">
                    <input type="range" min="1" max="50" defaultValue="15" className="w-full accent-purple-500" />
                    <span className="text-sm text-gray-400 w-24">Target: +15%</span>
                  </div>
                  <button className="mt-4 w-full py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition text-purple-400 border border-gray-600">
                    Auto-Improve LaTeX Code (Coming Soon)
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500 space-y-4">
                <svg className="w-24 h-24 mx-auto opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p>Paste your resume and job description to see your match score.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
