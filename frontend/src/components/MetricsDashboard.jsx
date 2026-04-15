import React, { useMemo } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';

// ─────────────────────────────────────────────────────────────────────────────
// MetricsDashboard.jsx
// A 2x2 grid displaying real-time simulation metrics and RL telemetry using Recharts.
// ─────────────────────────────────────────────────────────────────────────────

const MetricsDashboard = ({ trainingMetrics, history = [] }) => {
  // Ensure we have a valid history array even if empty
  const safeHistory = Array.isArray(history) ? history : [];
  
  // ─────────────────────────────────────────────────────────────────────────────
  // Data processing for charts
  // ─────────────────────────────────────────────────────────────────────────────
  
  // Chart 1: Reward over time with Rolling Average
  const rewardData = useMemo(() => {
    return safeHistory.map((entry, index, arr) => {
      // Calculate a rolling average over the last 20 steps
      const startIdx = Math.max(0, index - 19);
      const windowSlice = arr.slice(startIdx, index + 1);
      const sum = windowSlice.reduce((acc, curr) => acc + (curr.current_reward || 0), 0);
      const rollingAvg = sum / windowSlice.length;

      return {
        step: entry.step,
        reward: entry.current_reward || 0,
        rollingAvg: rollingAvg
      };
    });
  }, [safeHistory]);

  // Chart 2: Stacked waiting times (limit to last 50 steps for clarity)
  const waitingTimeData = useMemo(() => {
    const last50 = safeHistory.slice(-50);
    return last50.map(entry => ({
      step: entry.step,
      north: entry.waiting_times?.north || 0,
      south: entry.waiting_times?.south || 0,
      east: entry.waiting_times?.east || 0,
      west: entry.waiting_times?.west || 0,
    }));
  }, [safeHistory]);

  // Chart 3: Current Queue Length Snapshot
  const queueData = useMemo(() => {
    const latest = safeHistory.length > 0 ? safeHistory[safeHistory.length - 1] : null;
    const queues = latest?.queue_lengths || { north: 0, south: 0, east: 0, west: 0 };
    return [
      { name: 'North', queue: queues.north },
      { name: 'South', queue: queues.south },
      { name: 'East', queue: queues.east },
      { name: 'West', queue: queues.west },
    ];
  }, [safeHistory]);

  // Chart 4: Stats calculations
  const stats = useMemo(() => {
    const last100 = safeHistory.slice(-100);
    if (last100.length === 0) {
      return { meanReward: 0, meanWait: 0, throughputEstimate: 0, improvement: 0 };
    }

    const meanReward = last100.reduce((sum, e) => sum + (e.current_reward || 0), 0) / last100.length;
    
    let totalWaitSum = 0;
    last100.forEach(e => {
      const wait = Object.values(e.waiting_times || {}).reduce((a, b) => a + b, 0);
      totalWaitSum += wait;
    });
    const meanWait = totalWaitSum / last100.length;

    // Estimate vehicles cleared based on step count for demo purposes 
    // (Actual throughput would be fetched from backend if tracked)
    const latestStep = safeHistory[safeHistory.length - 1]?.step || 0;
    const throughputEstimate = Math.floor(latestStep * 0.45); 

    // Assume fixed timer baseline wait time was roughly 80s for comparison
    const baselineWait = 80; 
    let improvement = 0;
    if (meanWait > 0) {
      improvement = ((baselineWait - meanWait) / baselineWait) * 100;
      if (improvement < -100) improvement = -100; // clamp
    }

    return { 
      meanReward, 
      meanWait, 
      throughputEstimate, 
      improvement 
    };
  }, [safeHistory]);

  // Determine bar color conditionally
  const getBarColor = (queueLen) => {
    if (queueLen > 10) return "#ff3333"; // Red critical
    if (queueLen >= 5) return "#ffcc00"; // Yellow warning
    return "#33ff33"; // Green good
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // STYLES
  // ─────────────────────────────────────────────────────────────────────────────
  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '20px',
    padding: '20px',
    backgroundColor: '#0a0a0a',
    borderRadius: '12px',
    color: '#e0e0e0',
    fontFamily: "'Inter', sans-serif"
  };

  const chartBoxStyle = {
    backgroundColor: '#141414',
    border: '1px solid #222',
    borderRadius: '10px',
    padding: '15px',
    height: '300px',
    display: 'flex',
    flexDirection: 'column'
  };

  const titleStyle = {
    marginTop: 0,
    marginBottom: '15px',
    fontSize: '16px',
    fontWeight: '600',
    color: '#fff',
    borderBottom: '1px solid #333',
    paddingBottom: '8px'
  };

  const chartStyleOptions = {
    textColor: "#a0a0a0",
    gridColor: "#2a2a2a",
    tooltipBg: "#111",
    tooltipBorder: "#333",
  };

  // Stat Cards layout
  const statsContainerStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '15px',
    height: '100%'
  };

  const statCardStyle = {
    backgroundColor: '#1c1c1c',
    borderRadius: '8px',
    padding: '15px',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    textAlign: 'center'
  };

  return (
    <div style={gridStyle}>
      
      {/* ── 1. Reward Over Time (LineChart) ─────────────────────────── */}
      <div style={chartBoxStyle}>
        <h3 style={titleStyle}>Reward Over Time</h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rewardData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartStyleOptions.gridColor} />
            <XAxis dataKey="step" stroke={chartStyleOptions.textColor} tick={{ fontSize: 12 }} />
            <YAxis domain={[-1, 1]} stroke={chartStyleOptions.textColor} tick={{ fontSize: 12 }} />
            <Tooltip 
              contentStyle={{ backgroundColor: chartStyleOptions.tooltipBg, borderColor: chartStyleOptions.tooltipBorder }}
              labelStyle={{ color: '#fff' }}
            />
            <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />
            <ReferenceLine y={0} stroke="#666" strokeWidth={2} />
            <Line type="monotone" dataKey="reward" stroke="#3498db" strokeWidth={1} dot={false} name="Instant Reward" />
            <Line type="monotone" dataKey="rollingAvg" stroke="#2ecc71" strokeWidth={2.5} dot={false} name="Rolling Avg (20)" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── 2. Waiting Times (AreaChart Stacked) ──────────────────────── */}
      <div style={chartBoxStyle}>
        <h3 style={titleStyle}>Queue Waiting Times (Last 50s)</h3>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={waitingTimeData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartStyleOptions.gridColor} />
            <XAxis dataKey="step" stroke={chartStyleOptions.textColor} tick={{ fontSize: 12 }} />
            <YAxis stroke={chartStyleOptions.textColor} tick={{ fontSize: 12 }} />
            <Tooltip 
              contentStyle={{ backgroundColor: chartStyleOptions.tooltipBg, borderColor: chartStyleOptions.tooltipBorder }}
            />
            <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />
            {/* Soft, distinct colors for directions */}
            <Area type="monotone" dataKey="north" stackId="1" stroke="#e74c3c" fill="#e74c3c" fillOpacity={0.6} />
            <Area type="monotone" dataKey="south" stackId="1" stroke="#9b59b6" fill="#9b59b6" fillOpacity={0.6} />
            <Area type="monotone" dataKey="east" stackId="1" stroke="#f1c40f" fill="#f1c40f" fillOpacity={0.6} />
            <Area type="monotone" dataKey="west" stackId="1" stroke="#e67e22" fill="#e67e22" fillOpacity={0.6} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* ── 3. Queue Length Snapshot (BarChart) ────────────────────────── */}
      <div style={chartBoxStyle}>
        <h3 style={titleStyle}>Current Queue Lengths</h3>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={queueData} margin={{ top: 15, right: 5, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartStyleOptions.gridColor} />
            <XAxis dataKey="name" stroke={chartStyleOptions.textColor} tick={{ fontSize: 12 }} />
            <YAxis stroke={chartStyleOptions.textColor} tick={{ fontSize: 12 }} allowDecimals={false} />
            <Tooltip 
              cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
              contentStyle={{ backgroundColor: chartStyleOptions.tooltipBg, borderColor: chartStyleOptions.tooltipBorder }}
            />
            <Bar dataKey="queue" radius={[4, 4, 0, 0]} name="Vehicles">
              {queueData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry.queue)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── 4. Performance Stat Cards ──────────────────────────────────── */}
      <div style={chartBoxStyle}>
        <h3 style={titleStyle}>Performance Stats (Last 100)</h3>
        <div style={statsContainerStyle}>
          
          <div style={statCardStyle}>
            <div style={{ fontSize: '13px', color: '#888', marginBottom: '8px' }}>Mean Reward</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: stats.meanReward >= 0 ? '#2ecc71' : '#e74c3c' }}>
              {stats.meanReward.toFixed(3)}
            </div>
          </div>

          <div style={statCardStyle}>
            <div style={{ fontSize: '13px', color: '#888', marginBottom: '8px' }}>Total Waiting Time</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#f1c40f' }}>
              {stats.meanWait.toFixed(1)}s
            </div>
          </div>

          <div style={statCardStyle}>
            <div style={{ fontSize: '13px', color: '#888', marginBottom: '8px' }}>Est. Cleared</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#3498db' }}>
              {stats.throughputEstimate}
            </div>
          </div>

          <div style={statCardStyle}>
            <div style={{ fontSize: '13px', color: '#888', marginBottom: '8px' }}>Vs Fixed Timer</div>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: stats.improvement > 0 ? '#2ecc71' : '#e74c3c' }}>
              {stats.improvement > 0 ? '+' : ''}{stats.improvement.toFixed(1)}%
            </div>
          </div>

        </div>
      </div>

    </div>
  );
};

export default MetricsDashboard;
