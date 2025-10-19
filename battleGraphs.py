import matplotlib.pyplot as plt

def battle_graph(unit,total_attacks, hits,suffered_wounds , saves, total_wounds):
    plt.figure()
    values = [total_attacks, hits,suffered_wounds , saves, total_wounds]
    labels = ['Total Attacks', 'Hits', 'Wounds', 'Saves', 'Total Wounds']

    plt.bar(labels, values, color=['blue', 'green', 'red', 'orange', 'purple'])
    plt.ylabel('Count')
    plt.title('Battle Simulation Results for '+unit.name)
    plt.ylim(0, 25)
    # Annotate each bar with its value
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center', va='bottom')