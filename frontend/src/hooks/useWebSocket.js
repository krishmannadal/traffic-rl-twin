/**
 * useWebSocket.js — Custom hooks for real-time WebSocket communication
 * 
 * This file provides two hooks designed for different frontend clients:
 *  1. useSimulationSocket: For the React Dashboard (receives simulation state broadcast)
 *  2. useVehicleSocket: For the Expo mobile app (sends GPS, receives specific signal advice)
 * 
 * THE WEBSOCKET LIFECYCLE
 * -----------------------
 * 1. Connecting: new WebSocket(url) initiates the TCP handshake and HTTP upgrade.
 * 2. Open: The onopen event fires. Data can now flow securely in both directions.
 * 3. Message: The onmessage event fires when the server pushes data.
 * 4. Error/Close: If the server crashes, network drops, or tab closes, the TCP connection
 *    breaks, firing onerror and onclose. 
 * 
 * WHY AUTO-RECONNECT MATTERS
 * --------------------------
 * Mobile networks completely drop TCP connections when switching between 4G/5G or WiFi.
 * Laptops drop sockets when closing the lid briefly.
 * The server might restart during development. 
 * Without auto-reconnect, the UI becomes permanently "stuck" if the network blips.
 * A resilient frontend MUST assume WebSockets will drop periodically and attempt
 * to rebuild the connection in the background without requiring a user page refresh.
 */

import { useState, useEffect, useRef, useCallback } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
//  DASHBOARD HOOK: useSimulationSocket
// ─────────────────────────────────────────────────────────────────────────────

export function useSimulationSocket(url) {
  const [connected, setConnected] = useState(false);
  const [simulationState, setSimulationState] = useState(null);
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  
  // Use a ref to hold the active WebSocket instance so it persists across renders
  const wsRef = useRef(null);
  // Prevent multiple reconnection loops from spawning concurrently
  const reconnectTimeoutRef = useRef(null);

  const connect = useCallback(() => {
    // Prevent duplicate connections if we're already connecting/open
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) {
      return;
    }

    console.log(`[Dashboard WS] Attempting connection to ${url}`);
    const ws = new WebSocket(url);

    ws.onopen = () => {
      console.log("[Dashboard WS] Connected to simulation successfully");
      setConnected(true);
      // Optional: Inform the server what type of data we want
      ws.send(JSON.stringify({ type: "subscribe", channel: "all" }));
      
      // Clear any pending reconnection attempts
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        switch (message.type) {
          case "simulation_state":
            setSimulationState(message);
            // Append to history buffer for charts, keep max 100 entries to avoid memory leaks
            setHistory(prev => [...prev.slice(-99), message]);
            break;
            
          case "training_metrics":
            setTrainingMetrics(message);
            break;
            
          case "subscribed":
            console.log(`[Dashboard WS] Server acknowledged subscription: ${message.channel}`);
            break;
            
          default:
            // Non-state messages (ping/ack/etc) can be ignored or debugged
            break;
        }
      } catch (error) {
        console.error("[Dashboard WS] Error parsing incoming message:", error, "Raw data:", event.data);
      }
    };

    ws.onclose = (event) => {
      console.warn(`[Dashboard WS] Disconnected (code: ${event.code}). Auto-reconnecting in 3 seconds...`);
      setConnected(false);
      wsRef.current = null;
      
      // Trigger auto-reconnect
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = (error) => {
      // onerror is usually followed by onclose, so we don't need reconnect logic here
      console.error("[Dashboard WS] Connection error occurred:", error);
    };

    wsRef.current = ws;
  }, [url]);

  useEffect(() => {
    // Mount: initiate connection
    connect();

    // Unmount: clean up socket and timeouts
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  return { connected, simulationState, trainingMetrics, history };
}

// ─────────────────────────────────────────────────────────────────────────────
//  MOBILE APP HOOK: useVehicleSocket
// ─────────────────────────────────────────────────────────────────────────────

export function useVehicleSocket(urlBase, vehicleId) {
  const [connected, setConnected] = useState(false);
  const [signalAhead, setSignalAhead] = useState("unknown");
  const [timeToGreen, setTimeToGreen] = useState(0);
  const [speedAdvice, setSpeedAdvice] = useState(0);
  
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  // For the outgoing data loop
  const telemetryIntervalRef = useRef(null);

  const url = `${urlBase.replace(/\/$/, '')}/${vehicleId}`;

  const connect = useCallback(() => {
    if (!vehicleId) return;
    
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) {
      return;
    }

    console.log(`[Vehicle WS] Attempting connection for ${vehicleId}`);
    const ws = new WebSocket(url);

    ws.onopen = () => {
      console.log(`[Vehicle WS] Connected successfully as ${vehicleId}`);
      setConnected(true);
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }

      // Start the telemetry loop (simulated GPS/sensor data push)
      telemetryIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          // Construct simulated sensor payload 
          // In the real Expo app, this would use Expo Location / Sensors
          const payload = {
            vehicle_id: vehicleId,
            latitude: 12.9725 + (Math.random() * 0.0001), 
            longitude: 77.5932 + (Math.random() * 0.0001),
            speed: 10 + (Math.random() * 2), // m/s
            acceleration: 0,
            is_emergency: vehicleId.startsWith("emergency"),
            timestamp: Date.now() / 1000
          };
          ws.send(JSON.stringify(payload));
        }
      }, 1000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle server advice payload
        if (data.signal_ahead !== undefined) {
          setSignalAhead(data.signal_ahead);
          setTimeToGreen(data.time_to_green || 0);
          setSpeedAdvice(data.speed_advice || 0);
        } else if (data.type === "connected") {
          console.log(`[Vehicle WS] Server handshake confirmed: ${data.message}`);
        }
      } catch (error) {
        console.error("[Vehicle WS] JSON parse error:", error);
      }
    };

    ws.onclose = () => {
      console.warn(`[Vehicle WS] Disconnected. Reconnecting in 3s...`);
      setConnected(false);
      wsRef.current = null;
      
      if (telemetryIntervalRef.current) {
        clearInterval(telemetryIntervalRef.current);
      }
      
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = (error) => {
      console.error("[Vehicle WS] Connection error:", error);
    };

    wsRef.current = ws;
  }, [url, vehicleId]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (telemetryIntervalRef.current) clearInterval(telemetryIntervalRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  // Optionally provide a function to send custom structured events manually
  const sendMessage = useCallback((type, payload) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, ...payload }));
    }
  }, []);

  return { 
    connected, 
    signalAhead, 
    timeToGreen, 
    speedAdvice,
    sendMessage 
  };
}
