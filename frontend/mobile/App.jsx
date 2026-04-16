import React, { useState, useEffect } from 'react';
import { 
  StyleSheet, 
  View, 
  StatusBar,
  SafeAreaView
} from 'react-native';

import MapScreen from './screens/MapScreen';
import BottomSheet from './components/BottomSheet';
import useVehicleSensor from './hooks/useVehicleSensor';
import useVehicleSocket from './hooks/useVehicleSocket';

export default function App() {
  const [serverIP, setServerIP] = useState('192.168.1.100');
  const [activelyConnecting, setActivelyConnecting] = useState('192.168.1.100');
  const [isEmergency, setIsEmergency] = useState(false);

  // 1. Read local hardware GPS & Accelerometer
  const sensorData = useVehicleSensor();

  // 2. Stream to Backend via WebSockets
  const { 
    connected, 
    signalAhead, 
    timeToGreen, 
    speedAdvice,
    deviceId
  } = useVehicleSocket(activelyConnecting, isEmergency, sensorData);

  // Mock signal data for testing BottomSheet if not connected
  const signalData = {
    phase: signalAhead === 'unknown' ? 'RED' : signalAhead.toUpperCase(),
    countdown: timeToGreen || 0,
    speedAdvice: Math.round(speedAdvice * 3.6) || 0,
    active: connected
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {/* HD Map Layer */}
      <MapScreen />

      {/* Signal Intelligence Overlay */}
      <BottomSheet signalData={signalData} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#050505',
  }
});
