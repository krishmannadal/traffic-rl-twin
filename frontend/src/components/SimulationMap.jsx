import React from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// SimulationMap.jsx
// A dynamic SVG-based intersection visualization using data from the WebSockets.
// ─────────────────────────────────────────────────────────────────────────────

const SimulationMap = ({ simulationState, width = 600, height = 600 }) => {
  // Graceful fallback if no state is available
  const state = simulationState || {
    step: 0,
    simulation_time: 0,
    current_reward: 0,
    emergency_active: false,
    emergency_position: null,
    signal: {
      phase_per_direction: { north: "RED", south: "RED", east: "RED", west: "RED" }
    },
    vehicle_counts: { north: 0, south: 0, east: 0, west: 0 }
  };

  const { signal, vehicle_counts, emergency_active, emergency_position } = state;
  const phases = signal?.phase_per_direction || { north: "RED", south: "RED", east: "RED", west: "RED" };

  // Common sizes
  const center = { x: width / 2, y: height / 2 };
  const roadWidth = 120; // 60px per lane (2 lanes per road segment)
  const halfRoad = roadWidth / 2;
  const laneWidth = halfRoad; // simplified: 1 incoming, 1 outgoing lane per arm

  // ─────────────────────────────────────────────────────────────────────────────
  // Helper functions
  // ─────────────────────────────────────────────────────────────────────────────

  // Returns badge color based on congestion
  const getBadgeColor = (count) => {
    if (count > 20) return "var(--color-critical)";
    if (count > 10) return "var(--color-warning)";
    return "var(--color-badge-bg)";
  };

  // Traffic Light Component (rendered inside SVG)
  const TrafficLight = ({ x, y, direction, stateColor, isCrossTrafficInEmergency }) => {
    // If there's an emergency, force cross-traffic lights visually to RED overriding actual state 
    // to show priority handling in the UI (if emergency agent didn't already enforce it).
    const activeColor = isCrossTrafficInEmergency ? "RED" : stateColor;

    // Arrange lights horizontally or vertically depending on arm
    const isVertical = direction === "east" || direction === "west";
    const bgWidth = isVertical ? 24 : 64;
    const bgHeight = isVertical ? 64 : 24;

    const redGlow = activeColor === "RED" ? 'url(#glow-red)' : '';
    const yellowGlow = activeColor === "YELLOW" ? 'url(#glow-yellow)' : '';
    const greenGlow = activeColor === "GREEN" ? 'url(#glow-green)' : '';

    return (
      <g transform={`translate(${x - bgWidth / 2}, ${y - bgHeight / 2})`}>
        {/* Light Box Body */}
        <rect width={bgWidth} height={bgHeight} rx="4" fill="var(--color-tl-body)" stroke="#000" strokeWidth="2" />

        {/* The 3 light bulbs */}
        <circle 
          cx={isVertical ? 12 : 12} 
          cy={isVertical ? 12 : 12} 
          r="6" 
          fill={activeColor === "RED" ? "var(--color-red)" : "var(--color-tl-dim)"} 
          filter={redGlow}
          style={{ transition: "all 0.3s ease" }}
        />
        <circle 
          cx={isVertical ? 12 : 32} 
          cy={isVertical ? 32 : 12} 
          r="6" 
          fill={activeColor === "YELLOW" ? "var(--color-yellow)" : "var(--color-tl-dim)"} 
          filter={yellowGlow}
          style={{ transition: "all 0.3s ease" }}
        />
        <circle 
          cx={isVertical ? 12 : 52} 
          cy={isVertical ? 52 : 12} 
          r="6" 
          fill={activeColor === "GREEN" ? "var(--color-green)" : "var(--color-tl-dim)"} 
          filter={greenGlow}
          style={{ transition: "all 0.3s ease" }}
        />
      </g>
    );
  };

  // Emergency highlighting
  // Note: edge IDs in backend match 'north_to_center'
  const isEmergencyRoute = (dir) => emergency_active && emergency_position?.startsWith(dir);

  return (
    <div style={{ position: 'relative', width, height, borderRadius: '12px', overflow: 'hidden' }}>
      <svg width={width} height={height} style={{ backgroundColor: 'var(--color-bg)' }}>
        <defs>
          <style>{`
            :root {
              --color-bg: #0a0a0a;
              --color-road: #1a1a1a;
              --color-road-line: #444;
              --color-intersection: #222;
              
              /* Traffic Light Colors */
              --color-tl-body: #111;
              --color-tl-dim: #333;
              --color-red: #ff3333;
              --color-yellow: #ffcc00;
              --color-green: #33ff33;

              /* Badges */
              --color-badge-bg: #ffffff;
              --color-warning: #ff9900;
              --color-critical: #ff3300;

              /* Emergency */
              --color-emergency: rgba(0, 150, 255, 0.4);
            }

            .emergency-pulse {
              animation: pulse 1s infinite alternate;
            }

            @keyframes pulse {
              0% { opacity: 0.3; }
              100% { opacity: 0.8; }
            }
            
            .stat-text {
              font-family: 'Inter', system-ui, sans-serif;
              font-size: 14px;
              fill: #fff;
            }
          `}</style>
          
          {/* Neon Glow Filters */}
          <filter id="glow-red" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
          <filter id="glow-yellow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
          <filter id="glow-green" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* ── ROADS ────────────────────────────────────────────────────────── */}
        
        {/* North Road */}
        <rect x={center.x - halfRoad} y={0} width={roadWidth} height={center.y - halfRoad} fill="var(--color-road)" />
        {/* North Emergency Highlight */}
        {isEmergencyRoute('north') && <rect x={center.x - halfRoad} y={0} width={halfRoad} height={center.y - halfRoad} fill="var(--color-emergency)" className="emergency-pulse" />}
        {/* North Dashed Line */}
        <line x1={center.x} y1={0} x2={center.x} y2={center.y - halfRoad} stroke="var(--color-road-line)" strokeWidth="2" strokeDasharray="10 10" />

        {/* South Road */}
        <rect x={center.x - halfRoad} y={center.y + halfRoad} width={roadWidth} height={height / 2 - halfRoad} fill="var(--color-road)" />
        {isEmergencyRoute('south') && <rect x={center.x} y={center.y + halfRoad} width={halfRoad} height={height / 2 - halfRoad} fill="var(--color-emergency)" className="emergency-pulse" />}
        <line x1={center.x} y1={center.y + halfRoad} x2={center.x} y2={height} stroke="var(--color-road-line)" strokeWidth="2" strokeDasharray="10 10" />

        {/* West Road */}
        <rect x={0} y={center.y - halfRoad} width={center.x - halfRoad} height={roadWidth} fill="var(--color-road)" />
        {isEmergencyRoute('west') && <rect x={0} y={center.y} width={center.x - halfRoad} height={halfRoad} fill="var(--color-emergency)" className="emergency-pulse" />}
        <line x1={0} y1={center.y} x2={center.x - halfRoad} y2={center.y} stroke="var(--color-road-line)" strokeWidth="2" strokeDasharray="10 10" />

        {/* East Road */}
        <rect x={center.x + halfRoad} y={center.y - halfRoad} width={width / 2 - halfRoad} height={roadWidth} fill="var(--color-road)" />
        {isEmergencyRoute('east') && <rect x={center.x + halfRoad} y={center.y - halfRoad} width={width / 2 - halfRoad} height={halfRoad} fill="var(--color-emergency)" className="emergency-pulse" />}
        <line x1={center.x + halfRoad} y1={center.y} x2={width} y2={center.y} stroke="var(--color-road-line)" strokeWidth="2" strokeDasharray="10 10" />

        {/* ── INTERSECTION BOX ──────────────────────────────────────────────── */}
        <rect 
          x={center.x - halfRoad} 
          y={center.y - halfRoad} 
          width={roadWidth} 
          height={roadWidth} 
          fill="var(--color-intersection)" 
        />

        {/* ── TRAFFIC LIGHTS ─────────────────────────────────────────────────── */}
        {/* North inbound is on the West side (left of dashed line) */}
        <TrafficLight 
          direction="north" 
          x={center.x - halfRoad - 20} 
          y={center.y - halfRoad - 40} 
          stateColor={phases.north} 
          isCrossTrafficInEmergency={emergency_active && (isEmergencyRoute('east') || isEmergencyRoute('west'))}
        />
        {/* South inbound is on the East side (right of dashed line) */}
        <TrafficLight 
          direction="south" 
          x={center.x + halfRoad + 20} 
          y={center.y + halfRoad + 40} 
          stateColor={phases.south}
          isCrossTrafficInEmergency={emergency_active && (isEmergencyRoute('east') || isEmergencyRoute('west'))}
        />
        {/* West inbound is on the South side (bottom of dashed line) */}
        <TrafficLight 
          direction="west" 
          x={center.x - halfRoad - 40} 
          y={center.y + halfRoad + 20} 
          stateColor={phases.west}
          isCrossTrafficInEmergency={emergency_active && (isEmergencyRoute('north') || isEmergencyRoute('south'))} 
        />
        {/* East inbound is on the North side (top of dashed line) */}
        <TrafficLight 
          direction="east" 
          x={center.x + halfRoad + 40} 
          y={center.y - halfRoad - 20} 
          stateColor={phases.east}
          isCrossTrafficInEmergency={emergency_active && (isEmergencyRoute('north') || isEmergencyRoute('south'))}
        />

        {/* ── VEHICLE COUNT BADGES ───────────────────────────────────────────── */}
        {/* North Approach Badge */}
        <g transform={`translate(${center.x - laneWidth/2}, 40)`}>
          <circle cx="0" cy="0" r="16" fill={getBadgeColor(vehicle_counts.north || 0)} stroke="#222" strokeWidth="2"/>
          <text x="0" y="5" textAnchor="middle" fontSize="14" fontWeight="bold" fill="#000">
            {vehicle_counts.north || 0}
          </text>
        </g>
        {/* South Approach Badge */}
        <g transform={`translate(${center.x + laneWidth/2}, ${height - 40})`}>
          <circle cx="0" cy="0" r="16" fill={getBadgeColor(vehicle_counts.south || 0)} stroke="#222" strokeWidth="2"/>
          <text x="0" y="5" textAnchor="middle" fontSize="14" fontWeight="bold" fill="#000">
            {vehicle_counts.south || 0}
          </text>
        </g>
        {/* West Approach Badge */}
        <g transform={`translate(40, ${center.y + laneWidth/2})`}>
          <circle cx="0" cy="0" r="16" fill={getBadgeColor(vehicle_counts.west || 0)} stroke="#222" strokeWidth="2"/>
          <text x="0" y="5" textAnchor="middle" fontSize="14" fontWeight="bold" fill="#000">
            {vehicle_counts.west || 0}
          </text>
        </g>
        {/* East Approach Badge */}
        <g transform={`translate(${width - 40}, ${center.y - laneWidth/2})`}>
          <circle cx="0" cy="0" r="16" fill={getBadgeColor(vehicle_counts.east || 0)} stroke="#222" strokeWidth="2"/>
          <text x="0" y="5" textAnchor="middle" fontSize="14" fontWeight="bold" fill="#000">
            {vehicle_counts.east || 0}
          </text>
        </g>

        {/* ── EMERGENCY VEHICLE ICON ─────────────────────────────────────────── */}
        {emergency_active && emergency_position && (
          <text 
            fontSize="32" 
            textAnchor="middle" 
            style={{ filter: "drop-shadow(0px 0px 8px rgba(0, 150, 255, 0.8))" }}
            x={
              emergency_position.startsWith('north') ? center.x - laneWidth/2 :
              emergency_position.startsWith('south') ? center.x + laneWidth/2 :
              emergency_position.startsWith('west') ? 80 :
              emergency_position.startsWith('east') ? width - 80 : center.x
            }
            y={
              emergency_position.startsWith('north') ? 90 :
              emergency_position.startsWith('south') ? height - 90 :
              emergency_position.startsWith('west') ? center.y + laneWidth/2 + 10:
              emergency_position.startsWith('east') ? center.y - laneWidth/2 + 10 : center.y
            }
          >
            🚑
          </text>
        )}

        {/* ── STATS OVERLAY ──────────────────────────────────────────────────── */}
        <rect x="15" y={height - 95} width="220" height="80" rx="8" fill="rgba(20, 20, 20, 0.85)" stroke="#333" />
        <text x="30" y={height - 65} className="stat-text" fontWeight="bold">Time: {(state.simulation_time || 0).toFixed(1)} s</text>
        <text x="30" y={height - 45} className="stat-text">Step: {state.step || 0}</text>
        <text x="30" y={height - 25} className="stat-text" fill={state.current_reward < 0 ? "#ff5555" : "#55ff55"}>
          Reward: {(state.current_reward || 0).toFixed(3)}
        </text>

      </svg>
    </div>
  );
};

export default SimulationMap;
