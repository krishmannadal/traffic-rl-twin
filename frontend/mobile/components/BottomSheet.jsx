import React from 'react';
import { StyleSheet, View, Text, Animated } from 'react-native';

const BottomSheet = ({ signalData }) => {
    const { phase, countdown, speedAdvice, active } = signalData;

    if (!active) return null;

    return (
        <View style={styles.container}>
            <View style={styles.handle} />
            
            <View style={styles.content}>
                <View style={styles.signalHeader}>
                    <View style={[styles.signalDot, { backgroundColor: phase === 'GREEN' ? '#00ff88' : '#ff3333' }]} />
                    <Text style={styles.signalTitle}>Signal Intelligence Active</Text>
                </View>

                <View style={styles.statsRow}>
                    <View style={styles.statBox}>
                        <Text style={styles.statLabel}>COUNTDOWN</Text>
                        <Text style={[styles.statValue, { color: phase === 'GREEN' ? '#00ff88' : '#ff3333' }]}>
                            {countdown}s
                        </Text>
                    </View>

                    <View style={styles.divider} />

                    <View style={styles.statBox}>
                        <Text style={styles.statLabel}>ADVISED SPEED</Text>
                        <Text style={styles.statValue}>{speedAdvice} km/h</Text>
                    </View>
                </View>

                <View style={styles.aiBadge}>
                    <Text style={styles.aiText}>AI PREEMPTION OPTIMIZED</Text>
                </View>
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: '#111',
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        padding: 20,
        borderWidth: 1,
        borderColor: '#333',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: -5 },
        shadowOpacity: 0.5,
        shadowRadius: 10,
        elevation: 20
    },
    handle: {
        width: 40,
        height: 5,
        backgroundColor: '#333',
        borderRadius: 3,
        alignSelf: 'center',
        marginBottom: 20
    },
    content: { alignItems: 'center' },
    signalHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 20, gap: 10 },
    signalDot: { width: 12, height: 12, borderRadius: 6, shadowBlur: 10 },
    signalTitle: { color: '#fff', fontSize: 16, fontWeight: 'bold' },
    statsRow: { flexDirection: 'row', width: '100%', justifyContent: 'space-around', marginBottom: 20 },
    statBox: { alignItems: 'center' },
    statLabel: { color: '#555', fontSize: 10, fontWeight: '800', marginBottom: 5 },
    statValue: { color: '#fff', fontSize: 28, fontWeight: '900' },
    divider: { width: 1, height: '80%', backgroundColor: '#222' },
    aiBadge: { 
        backgroundColor: '#00ff8822', 
        paddingVertical: 5, 
        paddingHorizontal: 15, 
        borderRadius: 20,
        borderWidth: 1,
        borderColor: '#00ff8844'
    },
    aiText: { color: '#00ff88', fontSize: 10, fontWeight: 'bold' }
});

export default BottomSheet;
