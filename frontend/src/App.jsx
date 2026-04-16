import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { useSimulationSocket } from './hooks/useWebSocket';
import SimulationMap from './components/SimulationMap';
import MetricsDashboard from './components/MetricsDashboard';
import EmergencyPanel from './components/EmergencyPanel';
import AgentControls from './components/AgentControls';
import RewardChart from './components/RewardChart';
import MapBuilder from './pages/MapBuilder';
import LiveSimulation from './pages/LiveSimulation';
import './styles/index.css';

const API_BASE = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/dashboard";

const Dashboard = () => {
  const { connected, simulationState, trainingMetrics, history } = useSimulationSocket(WS_URL);
  
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [speed, setSpeed] = useState(1);
  const [agentStatus, setAgentStatus] = useState(null);

  const fetchAgentStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/agents/status`);
      const data = await res.json();
      setAgentStatus(data);
    } catch (e) {
      console.warn("Could not fetch agent status", e);
    }
  };

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(res => res.json())
      .then(data => {
        setSimulationRunning(data.sumo_running);
        setApiError(null);
        fetchAgentStatus();
      })
      .catch(err => {
        setApiError("Backend API Unreachable. Is FastAPI running?");
      });
  }, []);

  const handleStart = async () => {
    setApiError(null);
    try {
      const res = await fetch(`${API_BASE}/simulation/start`, { method: 'POST' });
      if (!res.ok) {
        const data = await res.json();
        if (data.detail && data.detail.includes("already running")) {
           setSimulationRunning(true);
           return;
        }
        throw new Error(data.detail || "Failed to start simulation");
      }
      setSimulationRunning(true);
      setSpeed(1);
    } catch (err) {
      setApiError(err.message);
    }
  };

  const handleStop = async () => {
    try {
      await fetch(`${API_BASE}/simulation/stop`, { method: 'POST' });
      setSimulationRunning(false);
    } catch (err) {
      setApiError(err.message);
    }
  };

  const handleSpeedChange = async (multiplier) => {
    try {
      const res = await fetch(`${API_BASE}/simulation/speed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ multiplier })
      });
      if (!res.ok) throw new Error("Failed to set speed");
      setSpeed(multiplier);
    } catch (err) {
      setApiError(err.message);
    }
  };

  const handleEmergency = async (route) => {
    try {
      const res = await fetch(`${API_BASE}/simulation/emergency`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ origin_edge: route.origin, destination_edge: route.dest })
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to trigger emergency");
      }
    } catch (err) {
      setApiError(err.message);
    }
  };

  return (
    <div style={appStyle}>
      {!connected && !apiError && (
        <div style={overlayStyle}>📡 Connecting to Simulation Backend...</div>
      )}

      {apiError && (
        <div style={errorBannerStyle}>
          ⚠️ API Error: {apiError}
        </div>
      )}

      <div style={topBarStyle}>
        <h1 style={titleStyle}>Traffic RL <span style={neonAccent}>Twin</span></h1>
        
        <div style={controlsStyle}>
          <Link to="/builder" style={linkStyle}>MAP BUILDER</Link>
          <Link to="/simulation" style={linkStyle}>LIVE SANDBOX</Link>

          <div style={{ width: '1px', height: '24px', backgroundColor: '#333', margin: '0 10px' }}></div>

          <div style={statusIndicatorStyle}>
            <div style={dotStyle(connected)}></div>
            {connected ? 'Connected' : 'Offline'}
          </div>

          <div style={{ width: '1px', height: '24px', backgroundColor: '#333', margin: '0 10px' }}></div>

          <button 
            style={buttonClass(true, simulationRunning || !connected)} 
            onClick={handleStart}
            disabled={simulationRunning || !connected}
          >
            ▶ START
          </button>
          
          <button 
            style={buttonClass(false, !simulationRunning || !connected)} 
            onClick={handleStop}
            disabled={!simulationRunning || !connected}
          >
            ◾ STOP
          </button>

          <div style={{ marginLeft: '15px', display: 'flex', gap: '5px', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', color: '#888', marginRight: '5px' }}>SPEED:</span>
            {[1, 2, 5].map(multiplier => (
              <button 
                key={multiplier}
                style={speedButtonStyle(speed === multiplier)}
                onClick={() => handleSpeedChange(multiplier)}
                disabled={!simulationRunning}
              >
                {multiplier}x
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={gridContainerStyle}>
        <div style={{ gridColumn: '1 / 2', gridRow: '1 / 2', display: 'flex', justifyContent: 'center' }}>
          <SimulationMap 
            simulationState={simulationState?.type === 'simulation_state' ? simulationState : null} 
            width={700}
            height={700}
          />
        </div>

        <div style={{ gridColumn: '2 / 3', gridRow: '1 / 2' }}>
          <MetricsDashboard 
            trainingMetrics={trainingMetrics} 
            history={history} 
          />
        </div>

        <div style={{ gridColumn: '1 / 2', gridRow: '2 / 3' }}>
          <EmergencyPanel 
            onTriggerEmergency={handleEmergency}
            emergencyActive={simulationState?.emergency_active || false}
            simulationRunning={simulationRunning}
          />
        </div>

        <div style={{ gridColumn: '2 / 3', gridRow: '2 / 3' }}>
          <AgentControls 
            agentStatus={agentStatus} 
            trainingProgress={trainingMetrics} 
            onAction={fetchAgentStatus} 
          />
        </div>

        <div style={{ gridColumn: '1 / 3', gridRow: '3 / 4' }}>
          <RewardChart 
            history={history}
            title="Live Simulation Reward"
            height={280}
          />
        </div>
      </div>
    </div>
  );
};

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/builder" element={<MapBuilder />} />
        <Route path="/simulation" element={<LiveSimulation />} />
      </Routes>
    </Router>
  );
};

// Styles
const appStyle = { backgroundColor: '#050505', color: '#fff', minHeight: '100vh', minWidth: '1280px', fontFamily: "'Inter', system-ui, sans-serif", display: 'flex', flexDirection: 'column', overflowX: 'hidden' };
const topBarStyle = { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '15px 30px', backgroundColor: '#111', borderBottom: '1px solid #333' };
const titleStyle = { margin: 0, fontSize: '24px', fontWeight: '800', color: '#fff', textShadow: '0 0 10px rgba(0, 255, 136, 0.5)' };
const neonAccent = { color: '#00ff88' };
const statusIndicatorStyle = { display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', color: '#888' };
const dotStyle = (isActive) => ({ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: isActive ? '#00ff88' : '#ff3333', boxShadow: isActive ? '0 0 8px #00ff88' : 'none' });
const controlsStyle = { display: 'flex', gap: '15px', alignItems: 'center' };
const buttonClass = (primary, disabled) => ({ padding: '8px 16px', borderRadius: '6px', border: 'none', fontWeight: 'bold', cursor: disabled ? 'not-allowed' : 'pointer', backgroundColor: disabled ? '#333' : (primary ? '#00ff88' : '#333'), color: disabled ? '#666' : (primary ? '#000' : '#fff'), transition: 'all 0.2s' });
const speedButtonStyle = (isActive) => ({ padding: '4px 10px', borderRadius: '4px', border: '1px solid #444', background: isActive ? '#00ff88' : 'transparent', color: isActive ? '#000' : '#fff', cursor: 'pointer', fontWeight: 'bold' });
const gridContainerStyle = { display: 'grid', gridTemplateColumns: '60% 40%', gridTemplateRows: 'auto auto', gap: '20px', padding: '20px', flex: 1 };
const errorBannerStyle = { backgroundColor: '#ff3333', color: '#fff', padding: '10px', textAlign: 'center', fontWeight: 'bold' };
const overlayStyle = { position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.8)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000, color: '#00ff88', fontSize: '24px', fontWeight: 'bold' };
const linkStyle = { color: '#00ff88', textDecoration: 'none', fontSize: '12px', fontWeight: 'bold', border: '1px solid #00ff88', padding: '5px 10px', borderRadius: '4px' };

export default App;
