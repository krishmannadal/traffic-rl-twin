import React, { useState, useEffect } from 'react';
import { Stage, Layer, Circle, Group, Text, Rect } from 'react-konva';
import { useSimulationSocket } from '../hooks/useWebSocket';
import axios from 'axios';

const WS_URL = "ws://localhost:8000/ws/dashboard";

// -- Static Components (Defined outside to prevent re-renders) --

const Scoreboard = ({ stats }) => {
    const { total = 0, moving = 0, waiting = 0, avgWait = 0 } = stats;
    return (
        <Group x={20} y={20}>
            <Rect width={200} height={120} fill="rgba(0,0,0,0.7)" stroke="#00ff88" strokeWidth={2} cornerRadius={8} />
            <Text x={15} y={15} text="LIVE STATS" fill="#00ff88" fontSize={14} fontStyle="bold" />
            <Text x={15} y={40} text={`Total Vehicles: ${total}`} fill="#fff" fontSize={12} />
            <Text x={15} y={60} text={`Moving: ${moving}`} fill="#00ff88" fontSize={12} />
            <Text x={15} y={80} text={`Waiting: ${waiting}`} fill="#ff3333" fontSize={12} />
            <Text x={15} y={100} text={`Avg Wait: ${avgWait}s`} fill="#fff" fontSize={12} fontStyle="bold" />
        </Group>
    );
};

const GridLine = ({ points, stroke }) => (
    <Group>
        <Rect 
            x={points[0]} 
            y={points[1]} 
            width={Math.max(1, points[2] - points[0])} 
            height={Math.max(1, points[3] - points[1])} 
            fill={stroke} 
        />
    </Group>
);

const LegendItem = ({ color, label }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '5px', fontSize: '13px' }}>
        <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: color }}></div>
        <span>{label}</span>
    </div>
);

// -- Main Component --

const LiveSimulation = () => {
    const { connected, simulationState } = useSimulationSocket(WS_URL);
    const [availableMaps, setAvailableMaps] = useState([]);
    const [selectedMap, setSelectedMap] = useState("default");
    const [loadingMap, setLoadingMap] = useState(false);

    useEffect(() => {
        const fetchMaps = async () => {
            try {
                const res = await axios.get('http://localhost:8000/admin/map/list');
                if (res.data && res.data.maps) {
                    setAvailableMaps(res.data.maps);
                }
            } catch (err) {
                console.error("Failed to fetch maps", err);
            }
        };
        fetchMaps();
    }, []);

    const handleReloadMap = async () => {
        setLoadingMap(true);
        try {
            // Must stop the current simulation first to free TraCI port
            await axios.post('http://localhost:8000/simulation/stop').catch(() => {});
            
            // Wait a second for port closure
            await new Promise(r => setTimeout(r, 1000));
            
            // Restart with new map
            await axios.post('http://localhost:8000/simulation/start', { map_name: selectedMap });
        } catch (err) {
            console.error("Failed to reload map", err);
        } finally {
            setLoadingMap(false);
        }
    };
    
    // Safety checks for simulation state
    const score = simulationState?.scoreboard || {};
    const vehicles = simulationState?.vehicle_positions || [];

    const stats = {
        total: score.total ?? 0,
        moving: score.moving ?? 0,
        waiting: score.waiting ?? 0,
        avgWait: score.avg_wait ?? 0
    };

    return (
        <div style={{ backgroundColor: '#050505', color: '#fff', minHeight: '100vh', overflow: 'hidden' }}>
            {/* Top Bar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '15px 30px', backgroundColor: '#111', borderBottom: '1px solid #333' }}>
                <h2 style={{ margin: 0 }}>Live <span style={{ color: '#00ff88' }}>Simulation</span> Dashboard</h2>
                
                {/* Map Selector UI */}
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center', backgroundColor: '#222', padding: '5px 15px', borderRadius: '4px' }}>
                    <span style={{ fontSize: '13px', color: '#aaa' }}>Current Map:</span>
                    <select 
                        value={selectedMap} 
                        onChange={(e) => setSelectedMap(e.target.value)}
                        style={{ backgroundColor: '#111', color: '#fff', border: '1px solid #444', padding: '5px' }}
                    >
                        <option value="default">Default (Centerville)</option>
                        {availableMaps.map(m => (
                            <option key={m} value={m}>{m} (Custom)</option>
                        ))}
                    </select>
                    <button 
                        onClick={handleReloadMap} 
                        disabled={loadingMap}
                        style={{ padding: '5px 15px', backgroundColor: '#0088ff', color: '#fff', border: 'none', borderRadius: '4px', cursor: loadingMap ? 'wait' : 'pointer' }}
                    >
                        {loadingMap ? "Loading..." : "Load Map"}
                    </button>
                </div>

                <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                        <div style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: connected ? '#00ff88' : '#ff3333' }}></div>
                        {connected ? 'BROADCAST ACTIVE' : 'RECONNECTING...'}
                    </div>
                    <button onClick={() => window.location.href = '/'} style={btnStyle}>Dashboard</button>
                </div>
            </div>

            <div style={{ position: 'relative', width: '100vw', height: 'calc(100vh - 70px)', backgroundColor: '#000' }}>
                <Stage width={window.innerWidth} height={window.innerHeight - 70} draggable>
                    <Layer>
                        {/* Render simple grid for reference */}
                        {Array.from({ length: 20 }).map((_, i) => (
                            <GridLine key={`h${i}`} points={[0, i * 100, 3000, i * 100]} stroke="#151515" />
                        ))}
                        {Array.from({ length: 30 }).map((_, i) => (
                            <GridLine key={`v${i}`} points={[i * 100, 0, i * 100, 3000]} stroke="#151515" />
                        ))}

                        {/* Render Vehicles */}
                        {vehicles.map(v => (
                            <Group key={v.id} x={v.x} y={v.y}>
                                <Circle 
                                    radius={v.type === 'emergency' ? 8 : 4} 
                                    fill={v.type === 'emergency' ? '#00ccff' : (v.speed < 0.1 ? '#ff3333' : '#00ff88')}
                                    shadowBlur={v.type === 'emergency' ? 10 : 0}
                                    shadowColor="#00ccff"
                                />
                                {v.type === 'emergency' && (
                                    <Circle 
                                        radius={15} 
                                        stroke="#00ccff" 
                                        strokeWidth={1} 
                                        opacity={0.3}
                                    />
                                )}
                            </Group>
                        ))}

                        {/* HUD / Scoreboard */}
                        <Scoreboard stats={stats} />
                    </Layer>
                </Stage>

                {/* Legend */}
                <div style={{ position: 'absolute', bottom: 20, right: 20, backgroundColor: 'rgba(0,0,0,0.7)', padding: '15px', borderRadius: '8px', border: '1px solid #333' }}>
                    <div style={{ fontSize: '12px', fontWeight: 'bold', marginBottom: '10px', color: '#888' }}>LEGEND</div>
                    <LegendItem color="#00ff88" label="Moving Vehicle" />
                    <LegendItem color="#ff3333" label="Stopped Vehicle" />
                    <LegendItem color="#00ccff" label="Emergency Vehicle (PPO Managed)" />
                </div>
            </div>
        </div>
    );
};

const btnStyle = {
    padding: '6px 15px',
    backgroundColor: '#333',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontWeight: 'bold'
};

export default LiveSimulation;
