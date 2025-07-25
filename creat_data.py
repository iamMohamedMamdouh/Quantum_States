import os
import matplotlib.pyplot as plt
from qutip import Bloch, basis, rand_ket, tensor, Qobj

base_dir = "quantum_states"
categories = ["entangled", "mixed", "pure"]
os.makedirs(base_dir, exist_ok=True)

for cat in categories:
    os.makedirs(os.path.join(base_dir, cat), exist_ok=True)

def random_mixed_state():
    psi1 = rand_ket(2)
    psi2 = rand_ket(2)
    rho = 0.5 * psi1.proj() + 0.5 * psi2.proj()
    return rho

def decohered_state(p=0.3):
    psi = rand_ket(2)
    rho = (1 - p) * psi.proj() + p * Qobj([[0.5, 0], [0, 0.5]])
    return rho

def save_bloch_sphere(state, filepath):
    b = Bloch()
    b.add_states(state)
    b.make_sphere()
    b.fig.savefig(filepath)
    plt.close(b.fig)

images_per_class = 1000

for i in range(images_per_class):
    # PURE STATE
    psi = rand_ket(2)
    save_bloch_sphere(psi, os.path.join(base_dir, "pure", f"pure_{i:04}.png"))

    # MIXED STATE
    rho = random_mixed_state()
    save_bloch_sphere(rho, os.path.join(base_dir, "mixed", f"mixed_{i:04}.png"))

    # ENTANGLED STATE
    bell = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
    rho_entangled = bell.proj().ptrace(0)
    save_bloch_sphere(rho_entangled, os.path.join(base_dir, "entangled", f"entangled_{i:04}.png"))

print("Done generating Bloch sphere images for all 3 quantum state types.")
