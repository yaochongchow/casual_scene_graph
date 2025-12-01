import torch
from src import utils

class QM9Noise:
    def __init__(self, node_marginals=None, edge_marginals=None):
        """
        Noise schedule for QM9 graphs using marginal distributions.
        
        Args:
            node_marginals (torch.Tensor, optional): Marginal distribution of node types.
                Defaults to QM9 stats: [C, N, O, F] -> [0.7230, 0.1151, 0.1593, 0.0026]
            edge_marginals (torch.Tensor, optional): Marginal distribution of edge types.
                Defaults to QM9 stats: [No Bond, Single, Double, Triple, Aromatic] 
                -> [0.7261, 0.2384, 0.0274, 0.0081, 0.0]
        """
        # Default QM9 marginals if not provided
        if node_marginals is None:
            self.node_marginals = torch.tensor([0.7230, 0.1151, 0.1593, 0.0026])
        else:
            self.node_marginals = node_marginals
            
        if edge_marginals is None:
            self.edge_marginals = torch.tensor([0.7261, 0.2384, 0.0274, 0.0081, 0.0])
        else:
            self.edge_marginals = edge_marginals
            
        self.X_classes = len(self.node_marginals)
        self.E_classes = len(self.edge_marginals)
        self.y_classes = 0 # QM9 usually has no y classes in this context, or handled separately
        
        # Create uniform transition matrices based on marginals
        # u_x: (1, dx, dx) - probability of transitioning to state j is marginal[j]
        self.u_x = self.node_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        
        # u_e: (1, de, de)
        self.u_e = self.edge_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        
        # u_y: placeholder
        self.u_y = torch.zeros(1, 0, 0)

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t * M
        where M is the marginal transition matrix.

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy). """
        beta_t = beta_t.view(-1, 1, 1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = torch.zeros(beta_t.size(0), 0, 0, device=device)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt_bar = alpha_bar_t * I + (1 - alpha_bar_t) * M

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = torch.zeros(alpha_bar_t.size(0), 0, 0, device=device)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)
