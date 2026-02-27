import torch
import torch.nn.functional as F

class CentinelState:
    def __init__(self, num_clients, tau=0.1, omega=0.7, Q_i=0.8, rho=0.4, eta=0.6, kappa=0.7, zeta=0.3):
        self.num_clients = num_clients
        self.tau = tau
        self.omega = omega
        
        # Subjective logic hyperparameters
        self.Q_i = Q_i
        self.rho = rho
        self.eta = eta
        self.kappa = kappa
        self.zeta = zeta
        
        # Each client's history of alpha and beta
        self.alpha_history = {i: [] for i in range(num_clients)}
        self.beta_history = {i: [] for i in range(num_clients)}
        
        # Reputation history (gamma)
        self.reputation_history = {i: [] for i in range(num_clients)}
        
        # Global centroids per label
        self.global_centroids = {}
        
    def get_accepted_clients(self):
        """Returns the set of client IDs whose current reputation >= omega."""
        accepted = set()
        for i in range(self.num_clients):
            if not self.reputation_history[i]:
                # If no history yet, accept by default
                accepted.add(i)
            elif self.reputation_history[i][-1] >= self.omega:
                accepted.add(i)
        return accepted

def compute_centroids(activations, labels):
    """
    Computes the mean activation vector for each label in the given batch.
    Args:
        activations (torch.Tensor): Smashed activations from client
        labels (torch.Tensor): Ground truth labels
    Returns:
        centroids (dict): Map from label (int) to mean activation tensor
        counts (dict): Map from label (int) to count of samples
    """
    centroids = {}
    counts = {}
    
    unique_labels = torch.unique(labels)
    for lbl in unique_labels:
        lbl_val = lbl.item()
        mask = (labels == lbl)
        acts_for_label = activations[mask]
        
        centroids[lbl_val] = acts_for_label.mean(dim=0).detach()
        counts[lbl_val] = acts_for_label.size(0)
        
    return centroids, counts

def run_sfl_centinel_round(clients, server, state, ref_loader, local_epochs=1, device='cpu'):
    """
    Executes a single communication round of the Centinel SplitFed defense.
    """
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    
    # ---------------------------------------------------------
    # Step A: Measure each client's centroid shift
    # ---------------------------------------------------------
    shifts = {}
    
    for client in clients:
        client.model.eval()
        
        client_loss = 0.0
        client_shifts = []
        client_counts = []
        
        # We need a forward pass to compute shift. 
        # Using the first batch of their dataset to calculate shift for efficiency context
        data, target = client.get_next_batch()
        if data is not None:
            with torch.no_grad():
                acts = client.model(data)
                
            c_centroids, c_counts = compute_centroids(acts, target)
            
            # Weighted average over labels of distance to global centroid
            weighted_dist_sum = 0.0
            total_count = 0
            
            for lbl, count in c_counts.items():
                if lbl in state.global_centroids:
                    dist = torch.norm(c_centroids[lbl] - state.global_centroids[lbl].to(device))
                    weighted_dist_sum += dist.item() * count
                    total_count += count
                    
            if total_count > 0:
                shifts[client.id] = weighted_dist_sum / total_count
            else:
                shifts[client.id] = 0.0
                
            # Put the data back basically by resetting iterator so training uses full data
            client.reset_iterator()
        else:
            shifts[client.id] = 0.0
            
    # Normalize shifts into scores summing to 1
    total_shift = sum(shifts.values())
    scores = {}
    for cid in shifts:
        scores[cid] = shifts[cid] / total_shift if total_shift > 0 else 0.0
        
    # ---------------------------------------------------------
    # Step B: Update subjective-logic reputation
    # ---------------------------------------------------------
    accepted_clients_ids = set()
    
    for cid in range(state.num_clients):
        score = scores.get(cid, 0.0)
        
        if score > state.tau:
            alpha_r, beta_r = 0.0, 1.0 # Negative interaction
        else:
            alpha_r, beta_r = 1.0, 0.0 # Positive interaction
            
        mean_alpha = sum(state.alpha_history[cid]) / len(state.alpha_history[cid]) if state.alpha_history[cid] else 1.0
        mean_beta = sum(state.beta_history[cid]) / len(state.beta_history[cid]) if state.beta_history[cid] else 0.0
        
        alpha_i = state.kappa * state.rho * alpha_r + state.zeta * state.rho * mean_alpha
        beta_i = state.kappa * state.eta * beta_r + state.zeta * state.eta * mean_beta
        
        # Add to history
        state.alpha_history[cid].append(alpha_r)
        state.beta_history[cid].append(beta_r)
        
        # Belief calculation b_i based on (alpha, beta, Q)
        W = 2.0 # Subjective logic uninformative prior weight
        
        # The typical conversion: b = alpha / (alpha + beta + W)
        # But pseudocode mentions Q_i constraint, commonly: b_i = alpha_i / (alpha_i + beta_i + W)
        b_i = alpha_i / (alpha_i + beta_i + W)
        
        gamma_i = b_i  # reputation score
        state.reputation_history[cid].append(gamma_i)
        
        if gamma_i >= state.omega:
            accepted_clients_ids.add(cid)
            
    # ---------------------------------------------------------
    # Step C, D: Train only accepted clients
    # ---------------------------------------------------------
    accepted_clients = [c for c in clients if c.id in accepted_clients_ids]
    
    if not accepted_clients:
        # No clients accepted, return zeros
        return 0.0, 0.0, list(shifts.values()), set()

    for epoch in range(local_epochs):
        for client in accepted_clients:
            client.reset_iterator()
            
        while True:
            smashed_activations_list = []
            labels_list = []
            active_clients = []
            
            for client in accepted_clients:
                result = client.forward_pass(global_round=epoch)
                if result[0] is not None:
                    smashed_data, labels = result
                    smashed_activations_list.append((client.id, smashed_data))
                    labels_list.append((client.id, labels))
                    active_clients.append(client)
            
            if not active_clients:
                break
                
            grad_to_clients_map = {}
            
            for client_id, smashed_data in smashed_activations_list:
                labels = next(l for cid, l in labels_list if cid == client_id)
                grad_to_client, loss, acc = server.train_step(smashed_data, labels, client_id=client_id)
                
                grad_to_clients_map[client_id] = grad_to_client
                total_loss += loss
                total_acc += acc
                total_batches += 1
                
            for client in active_clients:
                grad = grad_to_clients_map[client.id]
                client.backward_pass(grad)
                
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    avg_acc = total_acc / total_batches if total_batches > 0 else 0.0
    
    # ---------------------------------------------------------
    # Step E: FedAvg client-side and server-side weights
    # ---------------------------------------------------------
    client_weights = [c.get_weights() for c in accepted_clients]
    global_weights = server.aggregate_client_models(client_weights)
    
    # Broadcast to ALL clients (so rejected clients still get global update)
    for client in clients:
        client.set_weights(global_weights)
        
    # Aggregate server side models for accepted clients
    server.aggregate_server_models(active_client_indices=list(accepted_clients_ids))
    
    # ---------------------------------------------------------
    # Step F: Refresh reference centroids 
    # ---------------------------------------------------------
    # We use client_models[1] equivalent (here we just use the globally updated weights from clients[0])
    ref_client = clients[0]
    ref_client.model.eval()
    
    new_centroids = {}
    new_counts = {}
    
    with torch.no_grad():
        for data, target in ref_loader:
            acts = ref_client.model(data.to(device))
            c, counts = compute_centroids(acts, target.to(device))
            
            for lbl in c:
                if lbl in new_centroids:
                    new_centroids[lbl] = (new_centroids[lbl] * new_counts[lbl] + c[lbl] * counts[lbl]) / (new_counts[lbl] + counts[lbl])
                    new_counts[lbl] += counts[lbl]
                else:
                    new_centroids[lbl] = c[lbl]
                    new_counts[lbl] = counts[lbl]
                    
    state.global_centroids = new_centroids
        
    return avg_loss, avg_acc, scores, accepted_clients_ids
