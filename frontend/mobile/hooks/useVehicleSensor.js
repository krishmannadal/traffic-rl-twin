import { useState, useEffect, useRef } from 'react';
import * as Location from 'expo-location';
import { Accelerometer } from 'expo-sensors';

/**
 * hooks/useVehicleSensor.js
 * 
 * Aggregates hardware capabilities from the mobile device to mock a connected
 * vehicle. 
 * 
 * HOW SPEED IS CALCULATED FROM GPS COORDINATES:
 * ---------------------------------------------
 * We use the Haversine formula. The earth is a sphere, so calculating the distance 
 * between two (lat, lng) dots requires spherical trigonometry. 
 * distance (m) = haversine(lat1, lon1, lat2, lon2)
 * We divide the calculated distance by the exact time delta between the two readings.
 * Speed (m/s) = distance / (time_now - time_last).
 * Although modern GPS chips provide a `.speed` property, computing it manually 
 * smooths out noise in the simulator test environment where location spoofing 
 * tools often skip sending the speed vector directly.
 */

const toRad = (value) => (value * Math.PI) / 180;

// Haversine formula for distance between two lat/lng pairs in meters
const calcDistance = (lat1, lon1, lat2, lon2) => {
  const R = 6371e3; // Earth radius in meters
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c; 
};

export default function useVehicleSensor() {
  const [data, setData] = useState({
    latitude: 0,
    longitude: 0,
    speed: 0, // m/s
    acceleration: 0, // m/s^2
  });

  const lastCoordsRef = useRef(null);
  const lastTimeRef = useRef(null);

  useEffect(() => {
    let locationSubscription;
    let accelSubscription;

    const startSensors = async () => {
      // 1. Request location permission
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        console.warn('Permission to access location was denied');
        return;
      }

      // 2. Accelerometer setup (overall magnitude without gravity constraint)
      accelSubscription = Accelerometer.addListener(accelData => {
        const { x, y, z } = accelData;
        const magnitude = Math.sqrt(x*x + y*y + z*z) - 1.0; // remove 1G gravity
        setData(prev => ({ ...prev, acceleration: magnitude * 9.81 }));
      });
      Accelerometer.setUpdateInterval(1000); // 1 update per second

      // 3. Location setup (polled every 1000ms minimum)
      locationSubscription = await Location.watchPositionAsync({
        accuracy: Location.Accuracy.BestForNavigation,
        timeInterval: 1000,
        distanceInterval: 0
      }, (location) => {
        const currentLat = location.coords.latitude;
        const currentLng = location.coords.longitude;
        const currentTime = location.timestamp;

        let computedSpeed = location.coords.speed && location.coords.speed > 0 
          ? location.coords.speed 
          : 0;

        // Manual fallback calculation using Haversine
        if (lastCoordsRef.current && lastTimeRef.current) {
          const distance = calcDistance(
            lastCoordsRef.current.latitude, 
            lastCoordsRef.current.longitude,
            currentLat, 
            currentLng
          );
          const timeDeltaSeconds = (currentTime - lastTimeRef.current) / 1000;
          
          if (timeDeltaSeconds > 0) {
            computedSpeed = distance / timeDeltaSeconds;
          }
        }

        lastCoordsRef.current = { latitude: currentLat, longitude: currentLng };
        lastTimeRef.current = currentTime;

        setData(prev => ({
          ...prev,
          latitude: currentLat,
          longitude: currentLng,
          speed: computedSpeed
        }));
      });
    };

    startSensors();

    return () => {
      if (locationSubscription) locationSubscription.remove();
      if (accelSubscription) accelSubscription.remove();
    };
  }, []);

  return data;
}
