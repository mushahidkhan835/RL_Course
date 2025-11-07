import pennylane as qml
import random
import numpy as np
import random


[ds] = qml.data.load("op-t-mize")
file_idx = 1
file_name = "data"
idx = 0

for dataset in ds.circuits:
  dataset_x = []
  dataset_y = []

  ops_list = dataset.circuit
  chunk_size = 5
  for i in range(0, len(ops_list) - 1, chunk_size):
    dataset_x = (ops_list[i : i + chunk_size]) 
    np.savetxt("./circuit_ops/" + file_name + str(idx) + ".txt", dataset_x)
    with qml.tape.QuantumTape() as tape:
      for gate in dataset_x:
        qml.apply(gate) 
    fig, ax = qml.drawer.tape_mpl(tape)

    np.savetxt("./circuit_images/" + file_name + str(idx) + ".png", dataset_x)
    idx += 1

    if idx == len(ops_list) - 4:
      break

