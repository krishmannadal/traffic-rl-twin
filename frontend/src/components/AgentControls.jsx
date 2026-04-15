import React, { useState, useEffect } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// AgentControls.jsx
// Comprehensive RL Agent lifecycle and model management panel.
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE = "http://localhost:8000";

const AgentControls = ({ agentStatus, trainingProgress, onAction }) => {
  // ── UI State ──
  const [configOpen, setConfigOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  
  // ── Training Config ──
  const [timesteps, setTimesteps] = useState(500000);
  const [learningRate, setLearningRate] = useState(1e-4);
  const [batchSize, setBatchSize] = useState(256);
  const [agentType, setAgentType] = useState('traffic');

  // ── Model Management ──
  const [savedModels, setSavedModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  
  // ── Evaluation ──
  const [evalResults, setEvalResults] = useState(null);

  // Fetch models on mount and after changes
  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/agents/models`);
      const data = await res.json();
      setSavedModels(data.models || []);
      if (data.models && data.models.length > 0 && !selectedModel) {
        setSelectedModel(data.models[0].filename);
      }
    } catch (e) {
      console.error("Failed to fetch models", e);
    }
  };

  useEffect(() => {
    fetchModels();
  }, [agentStatus?.traffic_agent?.model_file]); // refetch when loaded model changes

  // ── Actions ──
  const handleApiAction = async (method, endpoint, body = null) => {
    setIsLoading(true);
    try {
      const opts = { method };
      if (body) {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify(body);
      }
      const res = await fetch(`${API_BASE}/agents/${endpoint}`, opts);
      if (!res.ok) {
        const errorData = await res.json();
        alert(`Error: ${errorData.detail}`);
      }
      if (onAction) onAction(); // notify parent to refresh status
      await fetchModels();
    } catch (e) {
      alert(`Network error: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const startTraining = () => handleApiAction('POST', 'train/start', {
    agent_type: agentType,
    total_timesteps: timesteps,
    learning_rate: learningRate,
    batch_size: batchSize
  });

  const pauseTraining = () => handleApiAction('POST', 'train/pause');
  const resumeTraining = () => handleApiAction('POST', 'train/resume');
  const stopTraining = () => handleApiAction('POST', 'train/stop');

  const runEvaluation = async () => {
    setIsLoading(true);
    setEvalResults(null);
    try {
      const res = await fetch(`${API_BASE}/agents/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_episodes: 10, agent_type: agentType })
      });
      const data = await res.json();
      setEvalResults(data);
    } catch (e) {
      alert(`Eval split failed: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // ── Helper Flags ──
  const isTraining = trainingProgress?.status === 'training';
  const isPaused = trainingProgress?.status === 'paused';
  const progressPct = trainingProgress?.total_timesteps > 0 
    ? (trainingProgress.timestep / trainingProgress.total_timesteps) * 100 
    : 0;

  // ── Styles ──
  const panelStyle = {
    backgroundColor: '#141414',
    border: '1px solid #333',
    borderRadius: '12px',
    padding: '20px',
    color: '#e0e0e0',
    fontFamily: "'Inter', sans-serif",
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
    overflowY: 'auto'
  };

  const sectionStyle = {
    backgroundColor: '#1c1c1c',
    borderRadius: '8px',
    padding: '15px',
    border: '1px solid #2a2a2a'
  };

  const titleStyle = {
    margin: '0 0 15px 0',
    fontSize: '14px',
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '1px'
  };

  const btnStyle = (bg, text, disabled) => ({
    backgroundColor: disabled ? '#333' : bg,
    color: disabled ? '#888' : text,
    padding: '10px 15px',
    border: 'none',
    borderRadius: '6px',
    fontWeight: 'bold',
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: isLoading ? 0.7 : 1,
    flex: 1
  });

  const selectStyle = {
    backgroundColor: '#0a0a0a',
    color: '#fff',
    border: '1px solid #444',
    padding: '8px',
    borderRadius: '4px',
    width: '100%',
    marginBottom: '10px'
  };

  // ── Render ──

  return (
    <div style={panelStyle}>
      {/* ── SECTION 1: TRAINING CONTROLS ── */}
      <div style={sectionStyle}>
        <h3 style={titleStyle}>Training Controls</h3>
        
        <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
          <button 
            style={btnStyle('#00ff88', '#000', isTraining || isPaused)} 
            onClick={startTraining}
            disabled={isTraining || isPaused || isLoading}
          >
            {isLoading ? '⏳' : '▶'} Start
          </button>
          
          {isPaused ? (
            <button style={btnStyle('#ffcc00', '#000', isLoading)} onClick={resumeTraining}>
              Resume
            </button>
          ) : (
            <button style={btnStyle('#ffcc00', '#000', !isTraining || isLoading)} onClick={pauseTraining} disabled={!isTraining}>
              Pause
            </button>
          )}

          <button 
            style={btnStyle('#ff3333', '#fff', (!isTraining && !isPaused) || isLoading)} 
            onClick={stopTraining}
            disabled={(!isTraining && !isPaused) || isLoading}
          >
            Stop & Save
          </button>
        </div>

        {/* Collapsible Config */}
        <div style={{ color: '#00ff88', cursor: 'pointer', fontSize: '12px' }} onClick={() => setConfigOpen(!configOpen)}>
          {configOpen ? '▼ Hide Hyperparameters' : '▶ Show Hyperparameters'}
        </div>
        
        {configOpen && (
          <div style={{ marginTop: '10px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '12px' }}>
            <div>
              Timesteps: {(timesteps/1000).toFixed(0)}k
              <input type="range" min="100000" max="1000000" step="50000" value={timesteps} onChange={e => setTimesteps(Number(e.target.value))} style={{width:'100%'}}/>
            </div>
            <div>
              LR: 
              <select value={learningRate} onChange={e => setLearningRate(Number(e.target.value))} style={{...selectStyle, padding: '4px', marginBottom: 0}}>
                <option value={1e-3}>1e-3</option>
                <option value={1e-4}>1e-4</option>
                <option value={1e-5}>1e-5</option>
              </select>
            </div>
            <div>
              Batch Size:
              <select value={batchSize} onChange={e => setBatchSize(Number(e.target.value))} style={{...selectStyle, padding: '4px', marginBottom: 0}}>
                <option value={64}>64</option>
                <option value={128}>128</option>
                <option value={256}>256</option>
              </select>
            </div>
            <div>
              Agent Type:
              <select value={agentType} onChange={e => setAgentType(e.target.value)} style={{...selectStyle, padding: '4px', marginBottom: 0}}>
                <option value="traffic">Traffic (DQN)</option>
                <option value="emergency">Emergency (PPO)</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {/* ── SECTION 2: TRAINING PROGRESS ── */}
      {(isTraining || isPaused) && trainingProgress && (
        <div style={sectionStyle}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={titleStyle}>{trainingProgress.status === 'paused' ? 'Paused' : 'Training'} {trainingProgress.agent_type}...</h3>
            <div style={{ fontSize: '12px', color: '#888' }}>{Math.floor(progressPct)}%</div>
          </div>

          <div style={{ width: '100%', height: '8px', backgroundColor: '#333', borderRadius: '4px', marginBottom: '15px', overflow: 'hidden' }}>
            <div style={{ width: `${progressPct}%`, height: '100%', backgroundColor: '#00ff88', transition: 'width 0.3s' }}/>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '12px' }}>
            <div style={{ backgroundColor: '#0a0a0a', padding: '10px', borderRadius: '6px' }}>
              <div style={{ color: '#888', marginBottom: '4px' }}>Reward</div>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: trainingProgress.mean_reward >= 0 ? '#00ff88' : '#ff3333' }}>
                {trainingProgress.mean_reward?.toFixed(3) || 0}
              </div>
            </div>
            <div style={{ backgroundColor: '#0a0a0a', padding: '10px', borderRadius: '6px' }}>
              <div style={{ color: '#888', marginBottom: '4px' }}>Epsilon (Explore)</div>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#3498db' }}>
                {trainingProgress.epsilon?.toFixed(3) || 0}
              </div>
            </div>
            <div style={{ backgroundColor: '#0a0a0a', padding: '10px', borderRadius: '6px' }}>
              <div style={{ color: '#888', marginBottom: '4px' }}>Speed</div>
              <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#fff' }}>
                {trainingProgress.fps || 0} FPS
              </div>
            </div>
            <div style={{ backgroundColor: '#0a0a0a', padding: '10px', borderRadius: '6px' }}>
              <div style={{ color: '#888', marginBottom: '4px' }}>Time Remaining</div>
              <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#fff' }}>
                {trainingProgress.estimated_remaining || "00:00"}
              </div>
            </div>
          </div>

          {/* GPU memory bar if available */}
          {agentStatus?.gpu?.available && (
            <div style={{ marginTop: '15px' }}>
              <div style={{ fontSize: '10px', color: '#888', display: 'flex', justifyContent: 'space-between' }}>
                <span>GPU VRAM: {agentStatus.gpu.name}</span>
                <span>{agentStatus.gpu.vram_used_gb} / {agentStatus.gpu.vram_total_gb} GB</span>
              </div>
              <div style={{ width: '100%', height: '4px', backgroundColor: '#333', borderRadius: '2px', marginTop: '4px' }}>
                <div style={{ width: `${(agentStatus.gpu.vram_used_gb / agentStatus.gpu.vram_total_gb) * 100}%`, height: '100%', backgroundColor: '#9b59b6' }}/>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── SECTION 3: MODEL MANAGEMENT ── */}
      <div style={sectionStyle}>
        <h3 style={titleStyle}>Model Management</h3>
        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={selectStyle}>
          {savedModels.length === 0 && <option disabled value="">No saved models</option>}
          {savedModels.map(m => (
            <option key={m.filename} value={m.filename}>
              {m.is_loaded ? '🟢 ' : ''}{m.filename} ({m.file_size_mb}MB)
            </option>
          ))}
        </select>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            style={btnStyle('#3498db', '#fff', !selectedModel || isLoading)} 
            onClick={() => handleApiAction('POST', 'models/load', { filename: selectedModel, agent_type: agentType })}
          >
            Load
          </button>
          <button 
            style={btnStyle('#2ecc71', '#000', isLoading)} 
            onClick={() => handleApiAction('POST', 'models/save', {})}
          >
            Save Current
          </button>
          <button 
            style={btnStyle('#e74c3c', '#fff', !selectedModel || isLoading)} 
            onClick={() => {
              if (window.confirm(`Delete ${selectedModel}?`)) {
                handleApiAction('DELETE', `models/${selectedModel}`);
              }
            }}
          >
            Delete
          </button>
        </div>
      </div>

      {/* ── EMERGENCY AGENT MODE ── */}
      <div style={sectionStyle}>
        <h3 style={titleStyle}>Emergency Mode</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <select 
            value={agentStatus?.emergency_agent?.mode || 'rule_based'}
            onChange={(e) => handleApiAction('POST', 'emergency/mode', { mode: e.target.value })}
            style={{ ...selectStyle, marginBottom: 0, flex: 1 }}
          >
            <option value="rule_based">Rule Based (Safe)</option>
            <option value="learned">Learned PPO (Experimental)</option>
          </select>
        </div>
        <p style={{ fontSize: '11px', color: '#888', marginTop: '10px', marginBottom: 0 }}>
          {agentStatus?.emergency_agent?.mode === 'rule_based' 
            ? 'Deterministic green corridors. 100% reliable for physical demos.' 
            : 'AI-driven preemptive timing. Requires fully trained weights to avoid collisions.'}
        </p>
      </div>

      {/* ── EVALUATION ── */}
      <div style={sectionStyle}>
        <h3 style={titleStyle}>Evaluation</h3>
        <button 
          style={btnStyle('#9b59b6', '#fff', isLoading || isTraining)} 
          onClick={runEvaluation}
          disabled={isLoading || isTraining}
        >
          {isLoading && !isTraining ? 'Evaluating...' : 'Run Deterministic Eval (10 eps)'}
        </button>

        {evalResults && (
          <div style={{ marginTop: '15px', backgroundColor: '#0a0a0a', padding: '10px', borderRadius: '6px', fontSize: '12px' }}>
            <div style={{ borderBottom: '1px solid #333', paddingBottom: '5px', marginBottom: '5px', color: '#00ff88' }}>
              Evaluation Complete
            </div>
            <div>Mean Wait: {evalResults.mean_waiting_time}s</div>
            <div>Mean Reward: {evalResults.mean_reward}</div>
            
            <div style={{ marginTop: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#888' }}>
                <span>Vs Fixed Timer</span>
                <span style={{ color: evalResults.vs_fixed_timer_improvement > 0 ? '#00ff88' : '#e74c3c' }}>
                  {evalResults.vs_fixed_timer_improvement > 0 ? '+' : ''}{evalResults.vs_fixed_timer_improvement}%
                </span>
              </div>
              <div style={{ width: '100%', height: '4px', backgroundColor: '#333', borderRadius: '2px', marginTop: '2px' }}>
                <div style={{ width: `${Math.min(Math.max(50 + evalResults.vs_fixed_timer_improvement/2, 0), 100)}%`, height: '100%', backgroundColor: evalResults.vs_fixed_timer_improvement > 0 ? '#00ff88' : '#e74c3c' }}/>
              </div>
            </div>
          </div>
        )}
      </div>

    </div>
  );
};

export default AgentControls;
