"""
coordinate_mapper.py — GPS ↔ SUMO Coordinate Translation
==========================================================

This module bridges the gap between real-world GPS coordinates
(latitude/longitude from a phone) and the internal Cartesian coordinate
system used by SUMO's simulation network.

HOW LINEAR INTERPOLATION MAPS BETWEEN COORDINATE SYSTEMS
─────────────────────────────────────────────────────────
SUMO uses a flat 2D Cartesian grid (x, y in metres) centred roughly
on the network origin.  GPS uses a curved (lat, lng) system on Earth's
surface.  For a small area (< 5 km²), Earth's curvature is negligible,
so a simple LINEAR mapping works:

    sumo_x = (lng - min_lng) / (max_lng - min_lng) * (sumo_max_x - sumo_min_x) + sumo_min_x
    sumo_y = (lat - min_lat) / (max_lat - min_lat) * (sumo_max_y - sumo_min_y) + sumo_min_y

This treats the GPS bounding box and the SUMO bounding box as two
rectangles and linearly stretches one onto the other.

Why this works for a city intersection:
  • Our demo area is ~500m × 500m.  At this scale, 1° of longitude ≈ 1° 
    of latitude ≈ ~111km, so the curvature distortion across 500m is
    < 0.0001° — completely invisible.
  • For a city-wide network (> 10 km), you'd need a proper projection
    (UTM or Web Mercator).  For our single intersection, linear is exact
    enough.

The reverse (sumo_to_gps) is the same formula solved for lat/lng,
which lets us push vehicle positions back to the phone as GPS coords.

WHAT SUMOLIB DOES VS TRACI
───────────────────────────
Both are Python interfaces to SUMO, but they serve completely different
purposes:

  SUMOLIB (used in this file):
    • Reads SUMO network files (.net.xml) OFFLINE — no running simulation.
    • Parses the static geometry: edge shapes, lane positions, junction
      coordinates, traffic light programs.
    • Used for spatial queries: "what is the nearest edge to point (x, y)?"
    • Lightweight: just XML parsing, no sockets.
    • Available via: import sumolib

  TRACI (used in environment.py):
    • Connects to a RUNNING SUMO process via a live TCP socket.
    • Reads/writes DYNAMIC state: vehicle positions, signal phases,
      detector values — things that change every simulation step.
    • Used for control: "set traffic light phase to 2", "add vehicle".
    • Requires a SUMO process to be running.

  Analogy: sumolib is reading the BLUEPRINT of a building (static).
           TraCI is talking to the BUILDING MANAGER while people are
           inside (dynamic, real-time).

We use sumolib here because coordinate mapping depends only on the
network geometry (static), not on the simulation state (dynamic).
This means CoordinateMapper works even when SUMO isn't running.

WHY WE NEED BOTH EDGE AND LANE LEVEL PRECISION
───────────────────────────────────────────────
An "edge" in SUMO is a road segment between two junctions.  Each edge
has 1+ "lanes" (the physical lanes on that road).  For example, edge
"north_to_center" has lanes "north_to_center_0" (right) and
"north_to_center_1" (left).

  EDGE LEVEL — enough for:
    • Routing: "vehicle is on the road from North to Center"
    • Emergency trigger: "spawn ambulance on edge south_to_center"
    • Rough position display on the dashboard map

  LANE LEVEL — needed for:
    • Accurate queue measurement: queue on lane_0 ≠ queue on lane_1
    • Lane-specific signal control: some phases give green to lane_0
      (right turn) but not lane_1 (through traffic)
    • Vehicle placement: traci.vehicle.add() requires a lane, not an edge

The phone app sends GPS → we find the nearest EDGE (for routing).
The RL environment needs the nearest LANE (for state observation).
Both methods are provided so each consumer gets the precision it needs.
"""

import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import sumolib


class CoordinateMapper:
    """
    Bidirectional mapper between GPS (lat, lng) and SUMO (x, y).

    Uses linear interpolation between a real-world GPS bounding box
    and the SUMO network's Cartesian bounding box.

    Parameters
    ----------
    sumo_net_path : str or Path
        Path to the SUMO .net.xml network file.
    real_world_bounds : dict
        GPS bounding box with keys: min_lat, max_lat, min_lng, max_lng.
        This defines the physical area your demo covers.
    """

    def __init__(
        self,
        sumo_net_path: str = "simulation/sumo_configs/map.net.xml",
        real_world_bounds: Optional[Dict[str, float]] = None,
    ):
        # ── Load the SUMO network (static geometry, no running sim) ───
        # sumolib.net.readNet() parses the .net.xml file into an in-memory
        # graph of junctions, edges, lanes, and connections.  This is a
        # one-time cost (~50ms for our small network).
        self._net_path = Path(sumo_net_path)
        if not self._net_path.exists():
            raise FileNotFoundError(
                f"SUMO network file not found: {self._net_path}"
            )

        self._net = sumolib.net.readNet(str(self._net_path))

        # ── SUMO coordinate bounds ────────────────────────────────────
        # getBoundary() returns (min_x, min_y, max_x, max_y) in SUMO's
        # Cartesian space (metres from the network origin).
        boundary = self._net.getBoundary()
        self._sumo_min_x = boundary[0]
        self._sumo_min_y = boundary[1]
        self._sumo_max_x = boundary[2]
        self._sumo_max_y = boundary[3]

        self._sumo_width = self._sumo_max_x - self._sumo_min_x
        self._sumo_height = self._sumo_max_y - self._sumo_min_y

        # ── Real-world GPS bounds ─────────────────────────────────────
        # Default: a small area around a fictional intersection for demos.
        # Replace with your actual demo location's bounding box.
        if real_world_bounds is None:
            real_world_bounds = {
                "min_lat": 12.9700,    # Example: Bangalore area
                "max_lat": 12.9750,
                "min_lng": 77.5900,
                "max_lng": 77.5950,
            }

        self._min_lat = real_world_bounds["min_lat"]
        self._max_lat = real_world_bounds["max_lat"]
        self._min_lng = real_world_bounds["min_lng"]
        self._max_lng = real_world_bounds["max_lng"]

        self._lat_range = self._max_lat - self._min_lat
        self._lng_range = self._max_lng - self._min_lng

        # Cache all edge IDs for the random fallback
        self._all_edges = [
            e.getID()
            for e in self._net.getEdges()
            if not e.getID().startswith(":")  # skip internal junction edges
        ]

        print(
            f"  📍 CoordinateMapper initialized\n"
            f"     SUMO bounds: x=[{self._sumo_min_x:.0f}, {self._sumo_max_x:.0f}] "
            f"y=[{self._sumo_min_y:.0f}, {self._sumo_max_y:.0f}]\n"
            f"     GPS bounds:  lat=[{self._min_lat:.4f}, {self._max_lat:.4f}] "
            f"lng=[{self._min_lng:.4f}, {self._max_lng:.4f}]\n"
            f"     Edges: {len(self._all_edges)}"
        )

    # ──────────────────────────────────────────────────────────────────
    #  gps_to_sumo()
    # ──────────────────────────────────────────────────────────────────

    def gps_to_sumo(self, lat: float, lng: float) -> Tuple[float, float]:
        """
        Convert GPS (lat, lng) → SUMO Cartesian (x, y).

        Uses linear interpolation:
            fraction_x = (lng - min_lng) / lng_range
            sumo_x = sumo_min_x + fraction_x * sumo_width

        The longitude maps to SUMO X (horizontal) and latitude maps to
        SUMO Y (vertical).  This convention matches SUMO's default
        coordinate system where X is East-West and Y is North-South.

        Parameters
        ----------
        lat, lng : float
            GPS coordinates.

        Returns
        -------
        (x, y) : tuple of float
            SUMO Cartesian coordinates in metres.
        """
        # Fraction of the way across the bounding box [0, 1]
        frac_x = (lng - self._min_lng) / self._lng_range if self._lng_range else 0.5
        frac_y = (lat - self._min_lat) / self._lat_range if self._lat_range else 0.5

        # Map fraction → SUMO coordinate range
        sumo_x = self._sumo_min_x + frac_x * self._sumo_width
        sumo_y = self._sumo_min_y + frac_y * self._sumo_height

        return (sumo_x, sumo_y)

    # ──────────────────────────────────────────────────────────────────
    #  sumo_to_gps()
    # ──────────────────────────────────────────────────────────────────

    def sumo_to_gps(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert SUMO Cartesian (x, y) → GPS (lat, lng).

        Reverse of gps_to_sumo():
            fraction_x = (x - sumo_min_x) / sumo_width
            lng = min_lng + fraction_x * lng_range

        Used when pushing vehicle positions from the simulation back
        to phone apps that expect GPS coordinates for map display.

        Parameters
        ----------
        x, y : float
            SUMO Cartesian coordinates.

        Returns
        -------
        (lat, lng) : tuple of float
            GPS coordinates.
        """
        frac_x = (x - self._sumo_min_x) / self._sumo_width if self._sumo_width else 0.5
        frac_y = (y - self._sumo_min_y) / self._sumo_height if self._sumo_height else 0.5

        lng = self._min_lng + frac_x * self._lng_range
        lat = self._min_lat + frac_y * self._lat_range

        return (lat, lng)

    # ──────────────────────────────────────────────────────────────────
    #  find_nearest_edge()
    # ──────────────────────────────────────────────────────────────────

    def find_nearest_edge(self, x: float, y: float) -> Optional[str]:
        """
        Find the closest SUMO edge to the given (x, y) position.

        Uses sumolib's spatial index:
            net.getNeighboringEdges(x, y, radius)
        which returns all edges within `radius` metres, sorted by
        distance.  We return the closest one.

        EDGE level precision is enough for routing and spawning.
        For lane-level state observation, use find_nearest_lane().

        Parameters
        ----------
        x, y : float
            SUMO Cartesian coordinates.

        Returns
        -------
        str or None
            Edge ID (e.g. "north_to_center"), or None if no edge is
            within 500m (shouldn't happen in a normal network).
        """
        # Search expanding radii to guarantee a result
        for radius in [50, 100, 200, 500]:
            edges = self._net.getNeighboringEdges(x, y, radius)
            if edges:
                # edges is list of (edge, distance), sort by distance
                edges.sort(key=lambda e: e[1])
                edge = edges[0][0]
                # Skip internal junction edges (prefixed with ":")
                if not edge.getID().startswith(":"):
                    return edge.getID()

        return None

    # ──────────────────────────────────────────────────────────────────
    #  find_nearest_lane()
    # ──────────────────────────────────────────────────────────────────

    def find_nearest_lane(self, x: float, y: float) -> Optional[str]:
        """
        Find the closest SUMO lane to the given (x, y) position.

        More precise than find_nearest_edge() because it returns a
        specific lane on that edge (e.g. "north_to_center_0").

        This is necessary for:
          • traci.vehicle.add() which requires a lane ID, not edge ID.
          • Per-lane queue length observations in the RL state vector.
          • Lane-specific signal phase mapping (some phases give green
            to lane 0 but not lane 1 on the same edge).

        Parameters
        ----------
        x, y : float
            SUMO Cartesian coordinates.

        Returns
        -------
        str or None
            Lane ID (e.g. "north_to_center_0"), or None if not found.
        """
        # Find the nearest edge first
        for radius in [50, 100, 200, 500]:
            edges = self._net.getNeighboringEdges(x, y, radius)
            if edges:
                edges.sort(key=lambda e: e[1])
                edge = edges[0][0]
                if edge.getID().startswith(":"):
                    continue

                # Now find the closest lane on this edge
                lanes = edge.getLanes()
                if not lanes:
                    return None

                best_lane = None
                best_dist = float("inf")
                for lane in lanes:
                    # lane.getShape() is a polyline [(x1,y1), (x2,y2), ...]
                    # Compute rough distance to the lane's midpoint
                    shape = lane.getShape()
                    mid_idx = len(shape) // 2
                    mx, my = shape[mid_idx]
                    dist = ((x - mx) ** 2 + (y - my) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_lane = lane

                if best_lane:
                    return best_lane.getID()

        return None

    # ──────────────────────────────────────────────────────────────────
    #  is_within_bounds()
    # ──────────────────────────────────────────────────────────────────

    def is_within_bounds(self, lat: float, lng: float) -> bool:
        """
        Check if a GPS coordinate falls within the configured demo area.

        Parameters
        ----------
        lat, lng : float
            GPS coordinates from the phone.

        Returns
        -------
        bool
            True if (lat, lng) is inside the bounding box.
        """
        return (
            self._min_lat <= lat <= self._max_lat
            and self._min_lng <= lng <= self._max_lng
        )

    # ──────────────────────────────────────────────────────────────────
    #  get_demo_bounds()
    # ──────────────────────────────────────────────────────────────────

    def get_demo_bounds(self) -> Dict[str, float]:
        """
        Return the configured GPS bounding box.

        The frontend uses these to centre its map view and draw the
        demo area boundary rectangle.

        Returns
        -------
        dict
            {min_lat, max_lat, min_lng, max_lng, center_lat, center_lng}
        """
        return {
            "min_lat": self._min_lat,
            "max_lat": self._max_lat,
            "min_lng": self._min_lng,
            "max_lng": self._max_lng,
            "center_lat": (self._min_lat + self._max_lat) / 2,
            "center_lng": (self._min_lng + self._max_lng) / 2,
        }

    # ──────────────────────────────────────────────────────────────────
    #  map_to_random_edge() — Demo Fallback
    # ──────────────────────────────────────────────────────────────────

    def map_to_random_edge(self) -> str:
        """
        Return a random valid edge ID from the network.

        Used as a DEMO FALLBACK when:
          • The phone's GPS is outside the demo bounding box (user is
            testing from home, not at the demo intersection).
          • GPS is unavailable (indoor testing, emulator without GPS mock).
          • The nearest-edge lookup returns None (shouldn't happen, but
            defensive coding).

        For a realistic demo, the random edge is chosen from the incoming
        approaches (edges ending at the junction) so the "vehicle" appears
        to be approaching the intersection, not spawning in the middle.

        Returns
        -------
        str
            A valid edge ID (e.g. "south_to_center").
        """
        # Prefer incoming edges (approaching the intersection) for realism
        incoming_edges = [
            e for e in self._all_edges
            if e.endswith("_to_center")
        ]
        if incoming_edges:
            return random.choice(incoming_edges)
        return random.choice(self._all_edges)

    # ──────────────────────────────────────────────────────────────────
    #  Convenience: GPS → Edge (combines two steps)
    # ──────────────────────────────────────────────────────────────────

    def gps_to_edge(self, lat: float, lng: float) -> str:
        """
        Convert GPS coordinates directly to the nearest SUMO edge.

        Combines gps_to_sumo() + find_nearest_edge() and falls back
        to a random edge if the GPS is outside bounds.

        Parameters
        ----------
        lat, lng : float
            GPS coordinates from the phone.

        Returns
        -------
        str
            Edge ID.
        """
        if not self.is_within_bounds(lat, lng):
            return self.map_to_random_edge()

        x, y = self.gps_to_sumo(lat, lng)
        edge_id = self.find_nearest_edge(x, y)

        if edge_id is None:
            return self.map_to_random_edge()

        return edge_id
