import os
import pennylane as qml
import numpy as np

[ds] = qml.data.load("op-t-mize")
file_name = "data"
idx = 0

os.makedirs("./circuit_ops", exist_ok=True)
os.makedirs("./circuit_images", exist_ok=True)

for dataset in ds.circuits:
    ops_list = dataset.circuit
    chunk_size = 5
    for i in range(0, len(ops_list) - 1, chunk_size):
        dataset_x = ops_list[i : i + chunk_size]
        np.savetxt(f"./circuit_ops/{file_name}{idx}.txt", [str(op) for op in dataset_x], fmt="%s")

        with qml.tape.QuantumTape() as tape:
            for gate in dataset_x:
                qml.apply(gate)
        fig, ax = qml.drawer.tape_mpl(tape)
        fig.savefig(f"./circuit_images/{file_name}{idx}.png")

        idx += 1
