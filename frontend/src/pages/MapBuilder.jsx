import React, { useState, useEffect } from 'react';
import { Stage, Layer, Circle, Line, Text, Group } from 'react-konva';
import axios from 'axios';

const API_BASE = "http://localhost:8000";

const MapBuilder = () => {
    const [nodes, setNodes] = useState([]);
    const [edges, setEdges] = useState([]);
    const [selectedNode, setSelectedNode] = useState(null);
    const [contextMenu, setContextMenu] = useState(null);
    const [message, setMessage] = useState("");

    const handleStageClick = (e) => {
        if (contextMenu) {
            setContextMenu(null);
            return;
        }

        const stage = e.target.getStage();
        const pointerPosition = stage.getPointerPosition();

        // If clicking on an empty space, create a node
        if (e.target === stage) {
            const newNode = {
                id: `node_${nodes.length}`,
                x: Math.round(pointerPosition.x),
                y: Math.round(pointerPosition.y),
                type: "priority"
            };
            setNodes([...nodes, newNode]);
            setSelectedNode(null);
        }
    };

    const handleNodeClick = (nodeId) => {
        if (selectedNode === null) {
            setSelectedNode(nodeId);
        } else if (selectedNode === nodeId) {
            setSelectedNode(null);
        } else {
            // Create edge between selectedNode and this node
            const edgeId = `edge_${edges.length}`;
            const newEdge = {
                id: edgeId,
                from_node: selectedNode,
                to_node: nodeId,
                lanes: 1,
                speed: 13.89
            };
            setEdges([...edges, newEdge]);
            setSelectedNode(null);
        }
    };

    const handleEdgeClick = (e, edge) => {
        const stage = e.target.getStage();
        const pointerPosition = stage.getPointerPosition();
        
        setContextMenu({
            x: pointerPosition.x,
            y: pointerPosition.y,
            edgeId: edge.id
        });
    };

    const updateEdge = (edgeId, updates) => {
        setEdges(edges.map(e => e.id === edgeId ? { ...e, ...updates } : e));
    };

    const handleAddVehicles = async (count) => {
        if (!contextMenu?.edgeId) return;
        setMessage(`Injecting ${count} vehicles into ${contextMenu.edgeId}...`);
        try {
            await axios.post(`${API_BASE}/admin/simulation/add_vehicles`, {
                edge_id: contextMenu.edgeId,
                count: count,
                vehicle_type: "car"
            });
            setMessage(`Success: ${count} vehicles added to ${contextMenu.edgeId}`);
        } catch (err) {
            setMessage(`Injection failed: ${err.response?.data?.detail || "Is the simulation running?"}`);
        }
    };

    const handleSetEmergency = async () => {
        if (!contextMenu?.edgeId) return;
        setMessage(`Setting ${contextMenu.edgeId} as emergency route...`);
        // In this demo, we mark it in the local state or trigger a backend PPO rule
        updateEdge(contextMenu.edgeId, { isEmergency: true });
        setMessage(`Emergency corridor activated for ${contextMenu.edgeId}`);
    };

    const handleSave = async () => {
        setMessage("Building SUMO network...");
        try {
            const res = await axios.post(`${API_BASE}/admin/map/build`, {
                nodes: nodes,
                edges: edges,
                map_name: "custom_map"
            });
            setMessage("Success! Network generated at simulation/sumo_configs/custom_maps/custom_map.net.xml");
        } catch (err) {
            setMessage(`Build failed: ${err.response?.data?.detail || err.message}`);
        }
    };

    const handleReset = () => {
        setNodes([]);
        setEdges([]);
        setSelectedNode(null);
        setContextMenu(null);
        setMessage("");
    };

    return (
        <div style={{ backgroundColor: '#111', color: '#fff', minHeight: '100vh', padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px', alignItems: 'center' }}>
                <h1 style={{ margin: 0, color: '#00ff88' }}>Admin Map Builder</h1>
                <div style={{ display: 'flex', gap: '10px' }}>
                    <button onClick={handleReset} style={btnStyle('#333', '#fff')}>Reset Canvas</button>
                    <button onClick={handleSave} style={btnStyle('#00ff88', '#000')}>Save & Build SUMO Map</button>
                    <button onClick={() => window.location.href = '/'} style={btnStyle('#333', '#fff')}>Dashboard</button>
                </div>
            </div>

            {message && (
                <div style={{ backgroundColor: message.includes('Success') ? '#004422' : '#440022', padding: '10px', borderRadius: '4px', marginBottom: '20px', border: '1px solid #00ff88' }}>
                    {message}
                </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '20px' }}>
                <div style={{ border: '2px solid #333', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#000' }}>
                    <Stage width={800} height={600} onClick={handleStageClick}>
                        <Layer>
                            {/* Render Edges */}
                            {edges.map(edge => {
                                const from = nodes.find(n => n.id === edge.from_node);
                                const to = nodes.find(n => n.id === edge.to_node);
                                if (!from || !to) return null;
                                return (
                                    <Line
                                        key={edge.id}
                                        points={[from.x, from.y, to.x, to.y]}
                                        stroke={contextMenu?.edgeId === edge.id ? '#00ff88' : '#666'}
                                        strokeWidth={4 + (edge.lanes * 2)}
                                        onClick={(e) => handleEdgeClick(e, edge)}
                                    />
                                );
                            })}

                            {/* Render Nodes */}
                            {nodes.map(node => (
                                <Group key={node.id} onClick={() => handleNodeClick(node.id)}>
                                    <Circle
                                        x={node.x}
                                        y={node.y}
                                        radius={15}
                                        fill={selectedNode === node.id ? '#00ff88' : '#333'}
                                        stroke="#fff"
                                        strokeWidth={2}
                                    />
                                    <Text
                                        x={node.x - 20}
                                        y={node.y + 20}
                                        text={node.id}
                                        fill="#fff"
                                        fontSize={12}
                                    />
                                </Group>
                            ))}
                        </Layer>
                    </Stage>
                </div>

                <div style={{ backgroundColor: '#1a1a1a', padding: '20px', borderRadius: '8px', border: '1px solid #333' }}>
                    <h3 style={{ marginTop: 0, borderBottom: '1px solid #333', paddingBottom: '10px' }}>Instructions</h3>
                    <ul style={{ fontSize: '14px', color: '#ccc', paddingLeft: '20px' }}>
                        <li>Click empty space to place an <b>Intersection Node</b>.</li>
                        <li>Click node A then node B to draw a <b>Road</b>.</li>
                        <li>Click a road to configure <b>Lanes</b> and <b>Speed</b>.</li>
                        <li>Exported map will be available for simulation.</li>
                    </ul>

                    <div style={{ marginTop: '20px' }}>
                        <h4>Stats</h4>
                        <div style={{ fontSize: '14px' }}>Nodes: {nodes.length}</div>
                        <div style={{ fontSize: '14px' }}>Edges: {edges.length}</div>
                    </div>
                </div>
            </div>

            {contextMenu && (
                <div style={{
                    position: 'absolute',
                    top: contextMenu.y + 100,
                    left: contextMenu.x + 20,
                    backgroundColor: '#222',
                    border: '1px solid #00ff88',
                    padding: '15px',
                    borderRadius: '8px',
                    zIndex: 1000,
                    boxShadow: '0 4px 15px rgba(0,0,0,0.5)'
                }}>
                    <h4 style={{ margin: '0 0 10px 0' }}>Edge: {contextMenu.edgeId}</h4>
                    <div style={{ marginBottom: '10px' }}>
                        <label style={{ fontSize: '12px', display: 'block' }}>Lanes (1-4):</label>
                        <input 
                            type="number" 
                            min="1" max="4" 
                            value={edges.find(e => e.id === contextMenu.edgeId)?.lanes}
                            onChange={(e) => updateEdge(contextMenu.edgeId, { lanes: parseInt(e.target.value) })}
                            style={inputStyle}
                        />
                    </div>
                    <div style={{ marginBottom: '15px' }}>
                        <label style={{ fontSize: '12px', display: 'block' }}>Speed (km/h):</label>
                        <input 
                            type="number" 
                            value={Math.round(edges.find(e => e.id === contextMenu.edgeId)?.speed * 3.6)}
                            onChange={(e) => updateEdge(contextMenu.edgeId, { speed: parseFloat(e.target.value) / 3.6 })}
                            style={inputStyle}
                        />
                    </div>
                    <div style={{ borderTop: '1px solid #444', paddingTop: '10px', marginTop: '10px' }}>
                        <div style={{ fontSize: '12px', marginBottom: '8px', color: '#888' }}>SIMULATION COMMANDS:</div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                            <button onClick={() => handleAddVehicles(5)} style={btnStyle('#0088ff', '#fff', '11px')}>+5 Cars</button>
                            <button onClick={() => handleAddVehicles(10)} style={btnStyle('#0088ff', '#fff', '11px')}>+10 Cars</button>
                            <button onClick={() => handleAddVehicles(20)} style={btnStyle('#0088ff', '#fff', '11px')}>+20 Cars</button>
                            <button onClick={handleSetEmergency} style={btnStyle('#ff3333', '#fff', '11px')}>🚑 Emergency</button>
                        </div>
                    </div>
                    <div style={{ marginTop: '15px' }}>
                        <button onClick={() => setContextMenu(null)} style={btnStyle('#00ff88', '#000')}>Close Settings</button>
                    </div>
                </div>
            )}
        </div>
    );
};

const btnStyle = (bg, col, fontSize = '13px') => ({
    padding: '8px 16px',
    backgroundColor: bg,
    color: col,
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontWeight: 'bold',
    fontSize: fontSize,
    transition: 'opacity 0.2s'
});

const inputStyle = {
    backgroundColor: '#333',
    color: '#fff',
    border: '1px solid #444',
    padding: '5px',
    borderRadius: '4px',
    width: '100%'
};

export default MapBuilder;
