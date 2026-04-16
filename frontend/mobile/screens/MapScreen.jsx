import React, { useState, useRef, useEffect } from 'react';
import { StyleSheet, View, Text, TouchableOpacity, ActivityIndicator } from 'react-native';
import { WebView } from 'react-native-webview';
import * as Location from 'expo-location';

const LEAFLET_HTML = `
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { margin: 0; padding: 0; }
        #map { height: 100vh; width: 100vw; background: #050505; }
        .leaflet-container { background: #050505 !important; }
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map', {
            zoomControl: false,
            attributionControl: false
        }).setView([12.9716, 77.5946], 15);

        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 20
        }).addTo(map);

        var userMarker = L.circleMarker([12.9716, 77.5946], {
            radius: 8,
            fillColor: "#00ff88",
            color: "#fff",
            weight: 2,
            opacity: 1,
            fillOpacity: 1
        }).addTo(map);

        var routeLine = L.polyline([], { color: '#00ff88', weight: 5, opacity: 0.7 }).addTo(map);

        function updatePosition(lat, lng) {
            userMarker.setLatLng([lat, lng]);
            map.panTo([lat, lng]);
        }

        function drawRoute(points) {
            routeLine.setLatLngs(points);
            map.fitBounds(routeLine.getBounds(), { padding: [50, 50] });
        }

        window.addEventListener('message', (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'update_pos') {
                updatePosition(data.lat, data.lng);
            } else if (data.type === 'draw_route') {
                drawRoute(data.points);
            }
        });

        document.addEventListener('message', (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'update_pos') {
                updatePosition(data.lat, data.lng);
            } else if (data.type === 'draw_route') {
                drawRoute(data.points);
            }
        });
    </script>
</body>
</html>
`;

const MapScreen = ({ onSignalAdvice }) => {
    const webViewRef = useRef(null);
    const [mapReady, setMapReady] = useState(false);
    const [location, setLocation] = useState(null);

    useEffect(() => {
        (async () => {
            let { status } = await Location.requestForegroundPermissionsAsync();
            if (status !== 'granted') return;

            let loc = await Location.getCurrentPositionAsync({});
            setLocation(loc);
            
            // Start watching location
            Location.watchPositionAsync({
                accuracy: Location.Accuracy.High,
                timeInterval: 1000,
                distanceInterval: 1
            }, (newLoc) => {
                setLocation(newLoc);
                if (mapReady) {
                    sendToWebview({
                        type: 'update_pos',
                        lat: newLoc.coords.latitude,
                        lng: newLoc.coords.longitude
                    });
                }
            });
        })();
    }, [mapReady]);

    const sendToWebview = (data) => {
        if (webViewRef.current) {
            webViewRef.current.postMessage(JSON.stringify(data));
        }
    };

    return (
        <View style={styles.container}>
            <WebView
                ref={webViewRef}
                originWhitelist={['*']}
                source={{ html: LEAFLET_HTML }}
                style={styles.map}
                onLoadEnd={() => setMapReady(true)}
                javaScriptEnabled={true}
                domStorageEnabled={true}
            />
            
            {!mapReady && (
                <View style={styles.loader}>
                    <ActivityIndicator size="large" color="#00ff88" />
                    <Text style={styles.loaderText}>Initializing HD Map...</Text>
                </View>
            )}

            <View style={styles.overlay}>
                <View style={styles.searchBar}>
                    <Text style={styles.searchText}>Search Destination...</Text>
                </View>
            </div>
        </View>
    );
};

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#050505' },
    map: { flex: 1 },
    loader: { ...StyleSheet.absoluteFillObject, backgroundColor: '#050505', justifyContent: 'center', alignItems: 'center' },
    loaderText: { color: '#00ff88', marginTop: 10, fontWeight: 'bold' },
    overlay: { position: 'absolute', top: 50, left: 20, right: 20 },
    searchBar: { 
        backgroundColor: '#1a1a1a', 
        padding: 15, 
        borderRadius: 10, 
        borderWidth: 1, 
        borderColor: '#333',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.5,
        shadowRadius: 10
    },
    searchText: { color: '#888' }
});

export default MapScreen;
