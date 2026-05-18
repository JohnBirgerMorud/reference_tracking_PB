import torch
from plants import CostumDataset
import matplotlib.pyplot as plt

class BumpercarDataset(CostumDataset):
    def __init__(self, random_seed, horizon,x_bar, x0, std_ini=0.3, n_agents=2):
        exp_name = 'bumpercar'
        file_name = 'data_T'+str(horizon)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'
        
        super().__init__(random_seed, horizon, exp_name, file_name)
        
        self.std_ini = std_ini
        self.n_agents = n_agents

        
        self.x0 = x0
        self.xbar = x_bar
    
    
    def generate_vector_with_min_distance(self,interval_x1=(-3, 3), interval_x2=(2, 3), min_distance=1.0):
        while True:
            # Generate the first vector (x1 in [-2, 2], x2 in [0, 1])

            vec1 = torch.tensor([
                torch.empty(1).uniform_(interval_x1[0], interval_x1[1]),
                torch.empty(1).uniform_(interval_x2[0], interval_x2[1])
            ]).flatten()

            # Generate the second vector (x1 in [-2, 2], x2 in [0, 1])
            vec2 = torch.tensor([
                torch.empty(1).uniform_(interval_x1[0], interval_x1[1]),
                torch.empty(1).uniform_(interval_x2[0], interval_x2[1])
            ]).flatten()
            # Check the Euclidean distance
            distance = torch.norm(vec1 - vec2)
            if distance >= min_distance:
                # Concatenate vec1 and vec2 to form a (4,) vector
                return torch.cat((vec1, vec2))
            
    def _generate_data(self, num_samples):
        # Initial-condition / disturbance block
        state_dim_x0 = 7 * self.n_agents

        # Full reference block, same dimension as state
        state_dim_ref = 7 * self.n_agents

        state_dim = state_dim_x0 + state_dim_ref
        data = torch.zeros(num_samples, self.horizon, state_dim)

        for rollout_num in range(num_samples):
            vecs = self.generate_vector_with_min_distance(
                interval_x1=(-1, 5),
                interval_x2=(4, 4.1),
                min_distance=2.0,
            )

            x0_sample = self.x0.clone()
            noise = torch.zeros_like(self.x0)

            for i in range(self.n_agents):
                base = 7 * i
                noise[base + 0] = self.std_ini * torch.randn(())
                noise[base + 1] = self.std_ini * torch.randn(())

            x0_sample = x0_sample + noise
            data[rollout_num, 0, :state_dim_x0] = x0_sample

            # Full-state reference block starts here
            ref_start = state_dim_x0

            # Agent 1 reference x, y
            data[rollout_num, 1:, ref_start + 0:ref_start + 2] = vecs[0:2]

            # Agent 2 reference x, y
            data[rollout_num, 1:, ref_start + 7:ref_start + 9] = vecs[2:4]

        assert data.shape[0] == num_samples
        return data


        