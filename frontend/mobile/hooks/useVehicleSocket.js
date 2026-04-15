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
 *
 * BUG FIX (v2): sensorData was previously in the useEffect dependency array.
 * Since sensorData is a new object reference on every render (from useState in
 * useVehicleSensor), this caused the WebSocket to tear down and recreate on
 * every sensor reading (~1/sec). Now we use a ref to track the latest sensor
 * data and only reconnect when serverIP or isEmergency changes.
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
  
  // Track latest sensor data via ref to avoid WebSocket recreation on every reading
  const sensorDataRef = useRef(sensorData);

  // Keep the ref current without triggering reconnection
  useEffect(() => {
    sensorDataRef.current = sensorData;
  }, [sensorData]);

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
          // Read from the ref (always fresh) instead of the stale closure variable
          const currentSensor = sensorDataRef.current;
          ws.send(JSON.stringify({
            vehicle_id: uniqueId,
            latitude: currentSensor.latitude,
            longitude: currentSensor.longitude,
            speed: currentSensor.speed,
            acceleration: currentSensor.acceleration,
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
  }, [serverIP, isEmergency]); // Only reconnect on IP or emergency mode change — NOT on sensorData

  return { connected, ...advice, deviceId: deviceIdRef.current };
}
