import random
import statistics
import matplotlib.pyplot as plt
import csv


# ==============================
# UTILITY
# ==============================

def clamp(value):
    return max(0, min(1, value))


# ==============================
# AGENT MODEL
# ==============================

class Agent:
    def __init__(self):
        self.belief = 0.5 + random.uniform(-0.05, 0.05)
        self.susceptibility = random.uniform(0.5, 1.5)


# ==============================
# BIAS MODELS
# ==============================

def confirmation_bias(agent, strength):
    if agent.belief > 0.5:
        delta = strength * agent.susceptibility * (1 - agent.belief)
    else:
        delta = -strength * agent.susceptibility * agent.belief

    agent.belief = clamp(agent.belief + delta)


def anchoring_bias(agent, anchor_value, strength):
    delta = strength * (anchor_value - agent.belief)
    agent.belief = clamp(agent.belief + delta)


def availability_bias(agent, strength):
    emotional_event = random.uniform(0, 1)
    delta = strength * agent.susceptibility * emotional_event * (1 - agent.belief)
    agent.belief = clamp(agent.belief + delta)


def social_influence(agent, population_average, strength):
    delta = strength * (population_average - agent.belief)
    agent.belief = clamp(agent.belief + delta)


# ==============================
# SIMULATION
# ==============================

def simulate_biased(num_agents, time_steps,
                    confirmation_strength,
                    anchor_value,
                    availability_strength,
                    social_strength):

    agents = [Agent() for _ in range(num_agents)]
    avg_history = []

    for step in range(time_steps):

        population_avg = statistics.mean(a.belief for a in agents)

        for agent in agents:

            confirmation_bias(agent, confirmation_strength)

            if step == 0:
                anchoring_bias(agent, anchor_value, 0.4)

            if random.random() < 0.4:
                availability_bias(agent, availability_strength)

            social_influence(agent, population_avg, social_strength)

        avg_history.append(statistics.mean(a.belief for a in agents))

    final_beliefs = [a.belief for a in agents]
    return final_beliefs, avg_history


def simulate_rational(num_agents, time_steps):
    agents = [Agent() for _ in range(num_agents)]
    avg_history = []

    for _ in range(time_steps):
        for agent in agents:
            noise = random.uniform(-0.01, 0.01)
            agent.belief = clamp(agent.belief + noise)

        avg_history.append(statistics.mean(a.belief for a in agents))

    final_beliefs = [a.belief for a in agents]
    return final_beliefs, avg_history


# ==============================
# METRICS
# ==============================

def print_metrics(rational_beliefs, biased_beliefs):
    print("\n========== FINAL METRICS ==========")
    print(f"Rational Average: {statistics.mean(rational_beliefs):.3f}")
    print(f"Biased Average:   {statistics.mean(biased_beliefs):.3f}")
    print(f"Belief Drift:     {(statistics.mean(biased_beliefs) - statistics.mean(rational_beliefs)):.3f}")
    print(f"Rational Std Dev: {statistics.pstdev(rational_beliefs):.3f}")
    print(f"Biased Std Dev:   {statistics.pstdev(biased_beliefs):.3f}")
    print("===================================\n")


# ==============================
# CSV EXPORT
# ==============================

def export_csv(rational_history, biased_history):
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time Step", "Rational Avg", "Biased Avg"])

        for i in range(len(rational_history)):
            writer.writerow([i + 1, rational_history[i], biased_history[i]])

    print("Results exported to results.csv\n")


# ==============================
# PLOTTING
# ==============================

def plot_results(rational_history, biased_history):
    plt.figure(figsize=(10, 6))
    plt.plot(rational_history, label="Rational Model")
    plt.plot(biased_history, label="Biased Model")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Belief")
    plt.title("Belief Drift Under Cognitive Bias (Enhanced Model)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==============================
# MAIN
# ==============================

def main():
    print("===================================")
    print("     BiasLab – Enhanced Model")
    print("===================================\n")

    try:
        num_agents = int(input("Number of agents (1000): ") or 1000)
        time_steps = int(input("Time steps (20): ") or 20)
        confirmation_strength = float(input("Confirmation strength (0.04): ") or 0.04)
        anchor_value = float(input("Anchor value (0.75): ") or 0.75)
        availability_strength = float(input("Availability strength (0.05): ") or 0.05)
        social_strength = float(input("Social influence strength (0.03): ") or 0.03)
    except:
        print("Invalid input. Using defaults.\n")
        num_agents = 1000
        time_steps = 20
        confirmation_strength = 0.04
        anchor_value = 0.75
        availability_strength = 0.05
        social_strength = 0.03

    rational_beliefs, rational_history = simulate_rational(num_agents, time_steps)

    biased_beliefs, biased_history = simulate_biased(
        num_agents,
        time_steps,
        confirmation_strength,
        anchor_value,
        availability_strength,
        social_strength
    )

    print_metrics(rational_beliefs, biased_beliefs)

    export_csv(rational_history, biased_history)

    plot_results(rational_history, biased_history)


if __name__ == "__main__":
    main()
