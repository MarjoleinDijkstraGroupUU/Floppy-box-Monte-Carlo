import sys
import time
import hoomd
import numpy as np

start_time = time.time()

class RTT:
    """
    Calculate the volume of a rounded truncated tetrahedron.
    """

    def __init__(self, cube_length, truncation, sweep_radius):
        self.side_length = 2**0.5 * cube_length
        self.truncation = truncation
        self.sweep_radius = sweep_radius

    def body_volume(self):
        return (2**0.5 / 12) * self.side_length**3 * (1 - self.truncation**3 / 2)

    def face_volume(self):
        return 3**0.5 * self.side_length**2 * self.sweep_radius * (1 - self.truncation**2 / 2)
        
    def line_volume(self):
        return 3 * self.side_length * self.sweep_radius**2 * (2 * self.truncation * np.arccos(1 / 3) - np.pi * self.truncation - np.arccos(1 / 3) + np.pi)
        
    def poin_volume(self):
        return 4 / 3 * np.pi * self.sweep_radius**3

    def total_volume(self):
        return self.body_volume() + self.face_volume() + self.line_volume() + self.poin_volume()


def truncated_tetrahedron_vertices(cube_length, truncation):
    """
    Generate truncated tetrahedron vertices.
    """
    original_vertices = np.array([( 0.5 * cube_length, -0.5 * cube_length, -0.5 * cube_length),
                                  (-0.5 * cube_length,  0.5 * cube_length, -0.5 * cube_length),
                                  (-0.5 * cube_length, -0.5 * cube_length,  0.5 * cube_length),
                                  ( 0.5 * cube_length,  0.5 * cube_length,  0.5 * cube_length)])
    
    if truncation == 0:
        return original_vertices.tolist()

    vertices = []
    for i in range(len(original_vertices)):
        for j in range(len(original_vertices)):
            if i != j:
                vertices.append(original_vertices[i] + 0.5 * truncation * (original_vertices[j] - original_vertices[i]))
    
    return vertices


def sweep_radius(cube_length, roundness):
    """
    Calculate the sweep radius for a rounded tetrahedron.
    """
    sweep_radius = 3**0.5 / 2 * cube_length * roundness / (1 - roundness)

    return sweep_radius


def place_cubes(num_cubes, cell_length):
    box_size = int(np.ceil(num_cubes ** (1 / 3)))
    center_offset = (box_size - 1) / 2
    positions = []
    for i in range(num_cubes):
        x = i % box_size
        y = (i // box_size) % box_size
        z = i // (box_size ** 2)
        cube_center = [(x - center_offset) * cell_length, (y - center_offset) * cell_length, (z - center_offset) * cell_length]
        positions.append(cube_center)
    return positions

def pressure(steps):
    return (initial_pressure + (final_pressure - initial_pressure) * (steps / total_steps)**gamma)

class Status():

    def __init__(self, sim):
        self.sim = sim
    
    @property
    def pressure(self):
        return pressure(self.sim.timestep)

    @property
    def packing_fraction(self):
        return self.sim.state.N_particles * ref_volume / self.sim.state.box.volume


# class BoxReduce(hoomd.custom.Action):

#     def act(self, timestep):
#         snap = self._state.get_snapshot()
#         if snap.communicator.rank == 0:
#             # Returns how much this current box shape resembles a cube
#             acubicity = snap.box.xy + snap.box.xz + snap.box.yz
#             if acubicity > 5:



# Shape parameters and the number of particles in unit cell
truncation = float(sys.argv[sys.argv.index("-t") + 1]) # truncation
roundness = float(sys.argv[sys.argv.index("-r") + 1]) # roundness
N_particles = int(sys.argv[sys.argv.index("-n") + 1]) # number of particles

# Construct a RTT with the volume of reference volume
test_cube_length = 1
test_sweep_radius = sweep_radius(test_cube_length, roundness)
ref_volume = 1
test_volume = RTT(test_cube_length, truncation, test_sweep_radius).total_volume()
real_cube_length = test_cube_length / (test_volume / ref_volume) ** (1 / 3)
real_sweep_radius = test_sweep_radius / (test_volume / ref_volume) ** (1 / 3)

# Build an initial cubic box with the number of particles
cell_length = (real_cube_length + 2 * real_sweep_radius) * 2
box_length = int(np.ceil(N_particles ** (1 / 3))) * cell_length
snapshot = hoomd.Snapshot()
snapshot.particles.N = N_particles
snapshot.particles.position[:] = place_cubes(N_particles, cell_length)
snapshot.particles.orientation[:] = [(1, 0, 0, 0)] * N_particles
snapshot.particles.types = ["RTT"]
snapshot.particles.typeid[:] = [0] * N_particles
snapshot.configuration.box = [box_length, box_length, box_length, 0, 0, 0]

# Create a hpmc integrater
mc = hoomd.hpmc.integrate.ConvexSpheropolyhedron()
mc.shape["RTT"] = dict(vertices=truncated_tetrahedron_vertices(real_cube_length, truncation),
                       sweep_radius=real_sweep_radius)
cpu = hoomd.device.CPU()
rng = np.random.default_rng()
sim = hoomd.Simulation(device=cpu, seed=rng.integers(0, 100))
sim.operations.integrator = mc
sim.create_state_from_snapshot(snapshot)

initial_pressure = 2.0
final_pressure = 10_0000
total_steps = 200_0000 # you should probably use a larger number
gamma = np.random.uniform(3, 9)
print(f"gamma is {gamma}")

# Table writer
logger_time = hoomd.logging.Logger(categories=['scalar'])
logger_time.add(sim, quantities=['timestep', 'tps'])
status = Status(sim)
logger_time[('Status', 'pressure')] = (status, 'pressure', 'scalar')
logger_time[('Status', 'packing_fraction')] = (status, 'packing_fraction', 'scalar')
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=20000),
                          logger=logger_time)
sim.operations.writers.append(table)

# GSD writer
logger_shape = hoomd.logging.Logger()
logger_shape.add(mc, quantities=["type_shapes"])
gsd_writer = hoomd.write.GSD(filename="compress.gsd",
                             trigger=hoomd.trigger.Periodic(20000),
                             mode="wb",
                             logger=logger_shape)
sim.operations.writers.append(gsd_writer)

# Initial delta values
volume_delta = 0.2
length_delta = (0.01, 0.01, 0.01)
shear_delta = (0.01, 0.01, 0.01)

while sim.timestep <= total_steps:
    boxmc = hoomd.hpmc.update.BoxMC(trigger=hoomd.trigger.Periodic(1), betaP=pressure(sim.timestep))
    boxmc.volume = dict(weight=7 / 10, mode="ln", delta=volume_delta)
    boxmc.length = dict(weight=2 / 10, delta=length_delta)
    boxmc.shear = dict(weight=1 / 10, delta=shear_delta, reduce=3)
    sim.operations.updaters.append(boxmc)

    tune_move_size = hoomd.hpmc.tune.MoveSize.scale_solver(trigger=hoomd.trigger.Periodic(10),
                                                           moves=["a", "d"],
                                                           target=0.30,
                                                           max_translation_move=mc.d['RTT'],
                                                           max_rotation_move=mc.a['RTT'],
                                                           gamma=0.5)
    sim.operations.tuners.append(tune_move_size)

    tune_box_size = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(trigger=hoomd.trigger.Periodic(10),
                                                               boxmc=boxmc,
                                                               moves=["volume", "length_x", "length_y", "length_z", "shear_x", "shear_y", "shear_z"],
                                                               target=0.15,
                                                               max_move_size=dict(volume=volume_delta, 
                                                                                  length_x=length_delta[0],
                                                                                  length_y=length_delta[1],
                                                                                  length_z=length_delta[2],
                                                                                  shear_x=shear_delta[0],
                                                                                  shear_y=shear_delta[1],
                                                                                  shear_z=shear_delta[2]),
                                                               gamma=0.5)
    sim.operations.tuners.append(tune_box_size)
    
    sim.run(2000, write_at_start=(True if sim.timestep==0 else False))
    # for volume
    volume_delta = (boxmc.volume["delta"] if boxmc.volume["delta"]!=0 else 0.01)
    # for length
    length_x=(boxmc.length["delta"][0] if boxmc.length["delta"][0]!=0 else 0.01)
    length_y=(boxmc.length["delta"][1] if boxmc.length["delta"][1]!=0 else 0.01)
    length_z=(boxmc.length["delta"][2] if boxmc.length["delta"][2]!=0 else 0.01)
    length_delta = (length_x, length_y, length_z)
    # for shear
    shear_x=(boxmc.shear["delta"][0] if boxmc.shear["delta"][0]!=0 else 0.01)
    shear_y=(boxmc.shear["delta"][1] if boxmc.shear["delta"][1]!=0 else 0.01)
    shear_z=(boxmc.shear["delta"][2] if boxmc.shear["delta"][2]!=0 else 0.01)
    shear_delta = (shear_x, shear_y, shear_z)

    sim.operations.updaters.remove(boxmc)
    sim.operations.tuners.remove(tune_move_size)
    sim.operations.tuners.remove(tune_box_size)


# hoomd.write.GSD.write(state=sim.state, mode='xb', filename='densest_packing.gsd')

end_time = time.time()

print(f"Total time: {(end_time - start_time) / 60} minutes")
print(f"Final packing fraction: {sim.state.N_particles * ref_volume / sim.state.box.volume}")