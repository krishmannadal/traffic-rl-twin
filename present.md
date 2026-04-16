# 🏁 Presentation Script: Traffic RL Twin Expansion

This guide provides the exact sequence of commands and talking points for your presentation.

---

## 🛠 Step 1: Initialization
**Command:**
```bash
python start_demo.py
```
**What to say:**
> "I am starting the system orchestrator. This script validates our environment, checks for GPU acceleration via PyTorch, and launches our modular architecture: a FastAPI backend and a React-based city dashboard. The system is now live on Port 8000 and 5173."

---

## 🏗 Step 2: Custom Map Builder (Admin View)
**Action:**
1. Open `http://localhost:5173/builder` in the browser.
2. Click on the canvas to place 3-4 nodes.
3. Click and drag between nodes to create roads (Edges).
4. Click 'SAVE NETWORK' (if you've added the button) or explain the flow.

**What to say:**
> "Welcome to the Admin Map Builder. Instead of hardcoding road networks, I've built an interactive canvas. By placing nodes and connecting them, I am defining a graph topology. In the backend, we use a custom 'NetConverter' that transforms these visual coordinates into industry-standard SUMO XML files, allowing for rapid urban prototyping."

---

## 🚦 Step 3: Live Simulation & AI Metrics
**Action:**
1. Navigate to the **Live Sandbox** via the top navigation bar.
2. Hit the **START** button.
3. Observe the scoreboard numbers changing.

**What to say:**
> "We are now inside the Live Simulation. The simulation is running at 1Hz, but we can scale this up to 5x speed for faster testing. Look at the Scoreboard: these metrics—Waiting Time, Moving Vehicles, and Queue Length—are being pushed from the backend via WebSockets. The traffic signals here are being controlled by a DQN reinforcement learning agent that is actively trying to maximize flow by reducing the 'Waiting Time' metric you see on screen."

---

## 📱 Step 4: User Navigation (Mobile V2X)
**Action:**
1. Scan the QR code shown in the terminal with the **Expo Go** app.
2. Show the phone screen (Mobile UI).
3. If possible, show the matching vehicle on the Dashboard.

**What to say:**
> "On the mobile side, I've implemented a Vehicle-to-Infrastructure (V2I) interface. Using React Native and Leaflet.js, drivers see a real-time HD Map of their city. But the core feature is 'Signal Intelligence': the app communicates with the traffic lights via WebSockets. It shows a precise countdown to Green and advises the driver on the exact speed needed to avoid a red light, effectively creating a 'Green Wave' for every driver."

---

## 🚑 Step 5: Emergency Preemption (PPO Override)
**Action:**
1. On the Dashboard, click **TRIGGER EMERGENCY**.
2. Point out the color change of the vehicle/lights.

**What to say:**
> "What happens if an ambulance enters the city? I've implemented a PPO-based emergency override. The system immediately shifts into 'Green Corridor' mode, preemptively clearing traffic blocks ahead of the emergency vehicle. This demonstrates the system's ability to prioritize life-safety missions while the lower-level DQN agents manage the rest of the city's traffic."

---

## 📝 Closing Points
> "In summary, we've transitioned from a static simulation to a dynamic digital twin. We have full-stack connectivity from Admin Map Building to AI-driven city management, all the way down to the driver's mobile device."
