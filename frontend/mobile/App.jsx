import React, { useState } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TextInput, 
  TouchableOpacity, 
  SafeAreaView, 
  StatusBar 
} from 'react-native';

import useVehicleSensor from './hooks/useVehicleSensor';
import useVehicleSocket from './hooks/useVehicleSocket';

export default function App() {
  const [serverIP, setServerIP] = useState('192.168.1.100'); // Default local network IP
  const [activelyConnecting, setActivelyConnecting] = useState('');
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

  const handleConnect = () => {
    setActivelyConnecting(serverIP.trim());
  };

  const getSignalColor = () => {
    if (signalAhead === 'green') return '#33ff33';
    if (signalAhead === 'yellow') return '#ffcc00';
    if (signalAhead === 'red') return '#ff3333';
    return '#333';
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      {/* ── TOP SECTION: CONNECTION ──────────────────────────── */}
      <View style={styles.topSection}>
        <Text style={styles.headerTitle}>Traffic RL Twin</Text>
        <Text style={styles.subtext}>Device ID: {deviceId}</Text>
        
        <View style={styles.connectRow}>
          <TextInput 
            style={styles.input}
            value={serverIP}
            onChangeText={setServerIP}
            placeholder="Server IP (e.g. 192.168.1.5)"
            placeholderTextColor="#666"
            keyboardType="numeric"
          />
          <TouchableOpacity 
            style={[styles.btn, connected ? styles.btnActive : {}]} 
            onPress={handleConnect}
          >
            <Text style={styles.btnText}>{connected ? 'Connected' : 'Connect'}</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.statusPill}>
          <View style={[styles.dot, { backgroundColor: connected ? '#00ff88' : '#ff3333' }]} />
          <Text style={styles.subtext}>{connected ? 'Online - Streaming active' : 'Disconnected from Simulation'}</Text>
        </View>
      </View>

      {/* ── MIDDLE SECTION: VEHICLE STATUS ──────────────────────────── */}
      <View style={styles.middleSection}>
        
        {/* Speedometer Gauge Mockup */}
        <View style={styles.gaugeContainer}>
          <Text style={styles.gaugeValue}>{Math.round(sensorData.speed * 3.6)}</Text>
          <Text style={styles.gaugeUnit}>km/h</Text>
        </View>

        <View style={styles.splitValues}>
          <View style={styles.valueBox}>
            <Text style={styles.valueLabel}>GPS LAT</Text>
            <Text style={styles.valueNum}>{sensorData.latitude.toFixed(5)}</Text>
          </View>
          <View style={styles.valueBox}>
            <Text style={styles.valueLabel}>GPS LNG</Text>
            <Text style={styles.valueNum}>{sensorData.longitude.toFixed(5)}</Text>
          </View>
        </View>

        {/* Signal Ahead Indicator */}
        <View style={styles.signalContainer}>
          <Text style={styles.signalTitle}>SIGNAL AHEAD</Text>
          
          <View style={[styles.largeCircle, { backgroundColor: getSignalColor(), shadowColor: getSignalColor() }]}>
            <Text style={styles.circleText}>{signalAhead.toUpperCase()}</Text>
          </View>

          <Text style={styles.countdownTitle}>
            TIME TO GREEN: <Text style={styles.countdownTime}>{timeToGreen}s</Text>
          </Text>
          
          <Text style={styles.adviceText}>
            Recommended Speed: {(speedAdvice * 3.6).toFixed(0)} km/h
          </Text>
        </View>

      </View>

      {/* ── BOTTOM SECTION: EMERGENCY TOGGLE ──────────────────────────── */}
      <View style={[styles.bottomSection, isEmergency ? styles.emergencyActiveHighlight : {}]}>
        <TouchableOpacity 
          style={[styles.emergencyBtn, isEmergency ? styles.emergencyPulse : {}]}
          onPress={() => setIsEmergency(!isEmergency)}
        >
          <Text style={styles.emergencyBtnText}>
            {isEmergency ? '🚑 CANCEL EMERGENCY MODE' : '🚨 TRIGGER EMERGENCY MODE'}
          </Text>
        </TouchableOpacity>
      </View>

    </SafeAreaView>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#050505',
    justifyContent: 'space-between',
  },
  topSection: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#222',
  },
  headerTitle: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 'bold',
  },
  subtext: {
    color: '#888',
    fontSize: 12,
    marginTop: 2,
  },
  connectRow: {
    flexDirection: 'row',
    marginTop: 15,
    gap: 10,
  },
  input: {
    flex: 1,
    backgroundColor: '#141414',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    color: '#fff',
    paddingHorizontal: 15,
    height: 45,
  },
  btn: {
    backgroundColor: '#333',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
    borderRadius: 8,
    height: 45,
  },
  btnActive: {
    backgroundColor: '#00ff88',
  },
  btnText: {
    color: '#000',
    fontWeight: 'bold',
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 15,
    gap: 8,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  middleSection: {
    flex: 1,
    padding: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  gaugeContainer: {
    width: 150,
    height: 150,
    borderRadius: 75,
    borderWidth: 4,
    borderColor: '#222',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    backgroundColor: '#0a0a0a',
  },
  gaugeValue: {
    color: '#fff',
    fontSize: 48,
    fontWeight: 'bold',
  },
  gaugeUnit: {
    color: '#666',
    fontSize: 14,
  },
  splitValues: {
    flexDirection: 'row',
    width: '100%',
    justifyContent: 'space-around',
    marginBottom: 30,
  },
  valueBox: {
    alignItems: 'center',
  },
  valueLabel: {
    color: '#666',
    fontSize: 10,
    fontWeight: 'bold',
  },
  valueNum: {
    color: '#ccc',
    fontSize: 16,
    fontFamily: 'monospace',
    marginTop: 4,
  },
  signalContainer: {
    alignItems: 'center',
    backgroundColor: '#111',
    width: '100%',
    padding: 20,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#222',
  },
  signalTitle: {
    color: '#888',
    letterSpacing: 1,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  largeCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 15,
    elevation: 10,
  },
  circleText: {
    color: '#000',
    fontWeight: 'black',
    fontSize: 14,
  },
  countdownTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  countdownTime: {
    color: '#00ff88',
    fontSize: 24,
  },
  adviceText: {
    marginTop: 10,
    color: '#aaa',
    fontSize: 12,
  },
  bottomSection: {
    padding: 20,
    paddingBottom: 40,
    borderTopWidth: 1,
    borderTopColor: '#222',
    transition: 'all 0.3s',
  },
  emergencyActiveHighlight: {
    borderTopWidth: 4,
    borderTopColor: '#0066ff',
    backgroundColor: 'rgba(0, 102, 255, 0.1)',
  },
  emergencyBtn: {
    backgroundColor: '#cc0000',
    width: '100%',
    height: 60,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emergencyPulse: {
    backgroundColor: '#0066ff',
    shadowColor: '#0066ff',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
    elevation: 8,
  },
  emergencyBtnText: {
    color: '#fff',
    fontWeight: '900',
    fontSize: 18,
    letterSpacing: 1,
  }
});
