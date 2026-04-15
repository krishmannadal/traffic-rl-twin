import React, { useState, useEffect, useRef } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// EmergencyPanel.jsx
// Controls and monitors emergency preemption events.
// ─────────────────────────────────────────────────────────────────────────────

const EmergencyPanel = ({ onTriggerEmergency, emergencyActive, simulationRunning }) => {
  const [selectedRoute, setSelectedRoute] = useState('south_to_north');
  const [timer, setTimer] = useState(0);
  const [lastResult, setLastResult] = useState(null);
  
  // Ref to track start time for the local timer
  const startTimeRef = useRef(null);

  // Routes mapping for display vs internal IDs
  const routes = [
    { id: 'south_to_north', label: 'South → North', origin: 'south_to_center', dest: 'center_to_north' },
    { id: 'west_to_east', label: 'West → East', origin: 'west_to_center', dest: 'center_to_east' },
  ];

  // Logic to handle timer and results when emergency state changes
  useEffect(() => {
    if (emergencyActive) {
      // Emergency just started
      setTimer(0);
      setLastResult(null);
      startTimeRef.current = Date.now();
      
      const interval = setInterval(() => {
        setTimer((Date.now() - startTimeRef.current) / 1000);
      }, 100);
      
      return () => clearInterval(interval);
    } else if (startTimeRef.current !== null) {
      // Emergency just finished
      const finalTime = (Date.now() - startTimeRef.current) / 1000;
      
      // Calculate mock improvement metrics (in real app, these would come from backend)
      const baseline = 45.0 + Math.random() * 10; // Fictional baseline
      const timeSaved = baseline - finalTime;
      const improvement = (timeSaved / baseline) * 100;
      
      setLastResult({
        travelTime: finalTime,
        baseline: baseline,
        timeSaved: timeSaved,
        improvement: improvement
      });
      
      startTimeRef.current = null;
    }
  }, [emergencyActive]);

  // ─────────────────────────────────────────────────────────────────────────────
  // Styles
  // ─────────────────────────────────────────────────────────────────────────────

  const panelStyle = {
    backgroundColor: '#141414',
    border: `2px solid ${emergencyActive ? '#ff0000' : '#333'}`,
    borderRadius: '12px',
    padding: '20px',
    color: '#fff',
    fontFamily: "'Inter', sans-serif",
    width: '100%',
    transition: 'all 0.3s ease',
    animation: emergencyActive ? 'border-flash 1s infinite alternate' : 'none',
    boxSizing: 'border-box'
  };

  const buttonStyle = {
    width: '100%',
    padding: '16px',
    backgroundColor: simulationRunning ? '#cc0000' : '#444',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '18px',
    fontWeight: 'bold',
    cursor: simulationRunning ? 'pointer' : 'not-allowed',
    marginTop: '15px',
    transition: 'transform 0.2s',
    animation: simulationRunning && !emergencyActive ? 'pulse-red 2s infinite' : 'none'
  };

  const selectStyle = {
    width: '100%',
    padding: '10px',
    backgroundColor: '#222',
    color: '#fff',
    border: '1px solid #444',
    borderRadius: '6px',
    marginTop: '10px',
    outline: 'none'
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // Render Logic
  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div style={{ width: '100%' }}>
      <style>{`
        @keyframes pulse-red {
          0% { box-shadow: 0 0 0 0 rgba(204, 0, 0, 0.7); }
          70% { box-shadow: 0 0 0 15px rgba(204, 0, 0, 0); }
          100% { box-shadow: 0 0 0 0 rgba(204, 0, 0, 0); }
        }
        @keyframes border-flash {
          0% { border-color: #ff0000; box-shadow: 0 0 10px rgba(255, 0, 0, 0.5); }
          100% { border-color: #0066ff; box-shadow: 0 0 10px rgba(0, 102, 255, 0.5); }
        }
        @keyframes ambulance-shake {
          0% { transform: translateX(0); }
          25% { transform: translateX(-2px) rotate(-2deg); }
          75% { transform: translateX(2px) rotate(2deg); }
          100% { transform: translateX(0); }
        }
      `}</style>

      <div style={panelStyle}>
        {!emergencyActive ? (
          <div>
            <h3 style={{ margin: '0 0 10px 0', fontSize: '16px', color: '#888' }}>Emergency Control</h3>
            <label style={{ fontSize: '14px', color: '#aaa' }}>Select Target Route:</label>
            <select 
              value={selectedRoute} 
              onChange={(e) => setSelectedRoute(e.target.value)}
              style={selectStyle}
              disabled={!simulationRunning}
            >
              {routes.map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
            </select>
            
            <button 
              style={buttonStyle}
              onClick={() => onTriggerEmergency(routes.find(r => r.id === selectedRoute))}
              disabled={!simulationRunning}
            >
              TRIGGER EMERGENCY
            </button>
            
            {!simulationRunning && (
              <p style={{ color: '#666', fontSize: '12px', textAlign: 'center', marginTop: '10px' }}>
                Start simulation to enable emergency triggers
              </p>
            )}

            {lastResult && (
              <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#1c1c1c', borderRadius: '8px', borderLeft: '4px solid #2ecc71' }}>
                <h4 style={{ margin: '0 0 10px 0', color: '#2ecc71', fontSize: '14px' }}>Last Run Results</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '12px' }}>
                  <div>Travel Time: <span style={{ fontWeight: 'bold' }}>{lastResult.travelTime.toFixed(1)}s</span></div>
                  <div>Baseline: <span style={{ fontWeight: 'bold' }}>{lastResult.baseline.toFixed(1)}s</span></div>
                  <div>Time Saved: <span style={{ fontWeight: 'bold', color: '#2ecc71' }}>{lastResult.timeSaved.toFixed(1)}s</span></div>
                  <div>Improvement: <span style={{ fontWeight: 'bold', color: '#2ecc71' }}>{lastResult.improvement.toFixed(1)}%</span></div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: '#ff0000', fontWeight: 'bold', fontSize: '20px', marginBottom: '10px' }}>
              EMERGENCY ACTIVE
            </div>
            
            <div style={{ fontSize: '48px', margin: '20px 0', animation: 'ambulance-shake 0.2s infinite' }}>
              🚑
            </div>

            <div style={{ fontSize: '32px', fontWeight: 'bold', marginBottom: '10px' }}>
              {timer.toFixed(1)}s
            </div>

            <div style={{ height: '8px', width: '100%', backgroundColor: '#222', borderRadius: '4px', overflow: 'hidden', marginBottom: '5px' }}>
              <div style={{ 
                height: '100%', 
                width: `${Math.min((timer / 40) * 100, 100)}%`, 
                backgroundColor: '#0066ff',
                transition: 'width 0.1s linear'
              }} />
            </div>
            <div style={{ fontSize: '10px', color: '#666' }}>Estimated Corridor Completion</div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EmergencyPanel;
