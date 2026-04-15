import { useState, useEffect, useRef } from 'react';
import * as Device from 'expo-device';

/**
 * hooks/useVehicleSocket.js
 * 
 * Two-way WebSocket pipeline exclusively for this physical device. 
 * Reads sensor states and pushes to SUMO.
 * 
 * WHY WE SEND SPECIFICALLY EVERY 1000ms
 * -------------------------------------
 * Real-time GPS networks often emit at variable speeds (10hz, 5hz). We bottleneck 
 * and buffer this onto a strict 1000ms heartbeat because the SUMO simulation runs 
 * explicitly on a standard step_length of 1.0 seconds. 
 * Pushing updates over the network faster than the physics clock ticks wastes 
 * battery and bandwidth entirely, as SUMO only resolves moveToXY() vehicle 
 * adjustments in discrete integer-second steps anyway.
 */

export default function useVehicleSocket(serverIP, isEmergency = false, sensorData) {
  const [connected, setConnected] = useState(false);
  const [advice, setAdvice] = useState({
    signalAhead: "unknown",
    timeToGreen: 0,
    speedAdvice: 0
  });

  const wsRef = useRef(null);
  const intervalRef = useRef(null);
  const deviceIdRef = useRef(`device_${Device.osBuildId?.substring(0,6) || Math.random().toString(36).substring(2,8)}`);

  useEffect(() => {
    if (!serverIP) return;

    // Use a test prefix to automatically activate route handling without manual UI routing
    const uniqueId = isEmergency ? `emergency_${deviceIdRef.current}` : `car_${deviceIdRef.current}`;
    const url = `ws://${serverIP}:8000/ws/vehicle/${uniqueId}`;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      
      // Start the heavy loop heartbeat
      intervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          // Send active sensor data buffer
          ws.send(JSON.stringify({
            vehicle_id: uniqueId,
            latitude: sensorData.latitude,
            longitude: sensorData.longitude,
            speed: sensorData.speed,
            acceleration: sensorData.acceleration,
            is_emergency: isEmergency,
            timestamp: Date.now() / 1000
          }));
        }
      }, 1000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.signal_ahead) {
          setAdvice({
            signalAhead: data.signal_ahead,
            timeToGreen: data.time_to_green || 0,
            speedAdvice: data.speed_advice || 0
          });
        }
      } catch (e) {
        console.error("Failed to parse advice from server", e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      if (intervalRef.current) clearInterval(intervalRef.current);
    };

    ws.onerror = (e) => {
      console.error("Vehicle Socket Error:", e.message);
    };

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [serverIP, isEmergency, sensorData]); // Re-bind socket if IP or emergency status toggle changes identity

  return { connected, ...advice, deviceId: deviceIdRef.current };
}
