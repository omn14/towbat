import random

# Generate a list of random number pairs
num_pairs = 10000  # Change as needed
pairs = [(random.randint(1, 6), random.randint(1, 6)) for _ in range(num_pairs)]

# Discard the lowest number in each pair
highest_in_pairs = [min(a, b) for a, b in pairs]

print("Original pairs:", pairs)
print("Highest in each pair:", highest_in_pairs)

import matplotlib.pyplot as plt

plt.hist(highest_in_pairs, bins=[0.5,1.5,2.5,3.5,4.5,5.5,6.5], edgecolor='black', weights=[1/num_pairs]*len(highest_in_pairs), align='mid')
plt.xticks([1,2,3,4,5,6])
plt.title('Histogram, høyeste verdi av to terninger')
plt.xlabel('Høyeste terning verdi')
plt.ylabel('ca sannsynlighet')
plt.show()