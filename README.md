# Adaptive Reinforcement Learning Honeypot for Zero-Day Attack Detection

This project presents a reinforcement learning-based adaptive honeypot system designed to detect zero-day cyberattacks. The system dynamically configures security settings based on attacker interaction, improving deception, adaptability, and threat detection accuracy in real time.

---

## 🎯 Objective

- Detect zero-day attacks through intelligent, evolving honeypot behavior
- Use Q-learning RL agents to simulate real-time adaptation
- Analyze performance via reward, Q-values, and action efficiency

---

## 🛠️ Technologies Used

- Python 3.8+
- OpenAI Gym (custom environment)
- NumPy
- Matplotlib

---

## 📁 Folder Structure

```
adaptive-rl-honeypot-zero-day-detection/
├── src/
│   ├── train_agent.py        # RL agent training logic
│   ├── honeypot_env.py       # Custom honeypot simulation environment
│   └── evaluate_agent.py     # Script for evaluation & testing
├── .gitignore
├── LICENSE.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install numpy gym matplotlib
```

### 2. Train the Agent
```bash
python src/train_agent.py
```

### 3. Evaluate the Agent
```bash
python src/evaluate_agent.py
```

> NOTE: You must have the honeypot environment (`honeypot_env.py`) properly configured before training.

---

## 📊 Key Results (from Report)

- Stable Q-value convergence by episode ~200
- RL agent maintained preferred state stability with ~70% action consistency
- Demonstrated capability to adapt to novel attacker behavior in real time


---

## 👥 Team Members

- Anantha Reddy Pingili – RL Agent Development & Integration
- Sai Sumanth Koppolu – Honeypot Environment Design & System Evaluation


---

## 📜 License

This project is licensed under the [MIT License](LICENSE.txt).
