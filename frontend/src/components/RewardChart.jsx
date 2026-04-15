import React, { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine, Area, ComposedChart
} from 'recharts';

// ─────────────────────────────────────────────────────────────────────────────
// RewardChart.jsx
// A focused, standalone reward visualization with rolling average and 
// positive/negative zone shading.  Used in the dashboard when training
// metrics are streaming or when historical data is available.
// ─────────────────────────────────────────────────────────────────────────────

const RewardChart = ({ history = [], title = "Training Reward", height = 350 }) => {
  // Process history into chart data with rolling average
  const chartData = useMemo(() => {
    const safeHistory = Array.isArray(history) ? history : [];
    if (safeHistory.length === 0) return [];

    return safeHistory.map((entry, index, arr) => {
      // Rolling average over last 20 entries
      const windowSize = 20;
      const startIdx = Math.max(0, index - windowSize + 1);
      const window = arr.slice(startIdx, index + 1);
      const avgReward = window.reduce((sum, e) => sum + (e.current_reward || e.reward || 0), 0) / window.length;

      // Longer rolling average (50) for trend line
      const trendSize = 50;
      const trendStart = Math.max(0, index - trendSize + 1);
      const trendWindow = arr.slice(trendStart, index + 1);
      const trendAvg = trendWindow.reduce((sum, e) => sum + (e.current_reward || e.reward || 0), 0) / trendWindow.length;

      const reward = entry.current_reward || entry.reward || 0;

      return {
        step: entry.step || index,
        reward: reward,
        rollingAvg: avgReward,
        trend: trendAvg,
        // For area shading: split into positive and negative
        positive: reward >= 0 ? reward : 0,
        negative: reward < 0 ? reward : 0,
      };
    });
  }, [history]);

  // Calculate stats for the header
  const stats = useMemo(() => {
    if (chartData.length === 0) {
      return { current: 0, best: 0, mean: 0, improving: false };
    }
    const rewards = chartData.map(d => d.reward);
    const lastN = rewards.slice(-20);
    const prevN = rewards.slice(-40, -20);

    const current = rewards[rewards.length - 1];
    const best = Math.max(...rewards);
    const mean = lastN.reduce((a, b) => a + b, 0) / lastN.length;
    const prevMean = prevN.length > 0
      ? prevN.reduce((a, b) => a + b, 0) / prevN.length
      : mean;
    const improving = mean > prevMean;

    return { current, best, mean, improving };
  }, [chartData]);

  // ── Styles ──
  const containerStyle = {
    backgroundColor: '#141414',
    border: '1px solid #222',
    borderRadius: '12px',
    padding: '20px',
    fontFamily: "'Inter', sans-serif",
    color: '#e0e0e0',
  };

  const headerStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '15px',
    borderBottom: '1px solid #333',
    paddingBottom: '12px',
  };

  const titleStyle = {
    fontSize: '16px',
    fontWeight: '600',
    color: '#fff',
    margin: 0,
  };

  const statsRowStyle = {
    display: 'flex',
    gap: '20px',
    fontSize: '11px',
  };

  const statItemStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  };

  const emptyStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: height,
    color: '#666',
    fontSize: '14px',
    flexDirection: 'column',
    gap: '8px',
  };

  if (chartData.length === 0) {
    return (
      <div style={containerStyle}>
        <h3 style={titleStyle}>{title}</h3>
        <div style={emptyStyle}>
          <span style={{ fontSize: '32px' }}>📊</span>
          <span>No reward data yet. Start training or simulation to see live metrics.</span>
        </div>
      </div>
    );
  }

  return (
    <div style={containerStyle}>
      {/* Header with live stats */}
      <div style={headerStyle}>
        <h3 style={titleStyle}>{title}</h3>
        <div style={statsRowStyle}>
          <div style={statItemStyle}>
            <span style={{ color: '#888' }}>Current</span>
            <span style={{
              fontWeight: 'bold',
              color: stats.current >= 0 ? '#00ff88' : '#ff3333',
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {stats.current.toFixed(3)}
            </span>
          </div>
          <div style={statItemStyle}>
            <span style={{ color: '#888' }}>Best</span>
            <span style={{
              fontWeight: 'bold',
              color: '#3498db',
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {stats.best.toFixed(3)}
            </span>
          </div>
          <div style={statItemStyle}>
            <span style={{ color: '#888' }}>Avg (20)</span>
            <span style={{
              fontWeight: 'bold',
              color: '#f1c40f',
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {stats.mean.toFixed(3)}
            </span>
          </div>
          <div style={statItemStyle}>
            <span style={{ color: '#888' }}>Trend</span>
            <span style={{
              fontWeight: 'bold',
              color: stats.improving ? '#00ff88' : '#ff3333',
            }}>
              {stats.improving ? '↑ Improving' : '↓ Declining'}
            </span>
          </div>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
          <XAxis
            dataKey="step"
            stroke="#666"
            tick={{ fontSize: 11 }}
            tickFormatter={(v) => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v}
          />
          <YAxis
            stroke="#666"
            tick={{ fontSize: 11 }}
            domain={['auto', 'auto']}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#111',
              borderColor: '#333',
              borderRadius: '8px',
              fontSize: '12px',
            }}
            labelStyle={{ color: '#fff' }}
            formatter={(value, name) => [value.toFixed(4), name]}
          />
          <Legend
            verticalAlign="top"
            height={28}
            wrapperStyle={{ fontSize: 11, paddingTop: '4px' }}
          />

          {/* Zero reference line */}
          <ReferenceLine y={0} stroke="#555" strokeWidth={2} strokeDasharray="5 5" />

          {/* Raw reward (thin, transparent) */}
          <Line
            type="monotone"
            dataKey="reward"
            stroke="#3498db"
            strokeWidth={1}
            dot={false}
            opacity={0.25}
            name="Instant"
          />

          {/* Rolling average (prominent) */}
          <Line
            type="monotone"
            dataKey="rollingAvg"
            stroke="#00ff88"
            strokeWidth={2.5}
            dot={false}
            name="Avg (20)"
          />

          {/* Long-term trend */}
          <Line
            type="monotone"
            dataKey="trend"
            stroke="#9b59b6"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="6 3"
            name="Trend (50)"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RewardChart;
