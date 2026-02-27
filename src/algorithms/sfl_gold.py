def run_sfl_gold_round(clients, server, local_epochs=1):
    """
    Executes a single communication round of the SplitFed Gold algorithm.
    This oracle method perfectly filters out all malicious clients during the entire round,
    skipping their forward/backward passes and aggregation, providing an upper bound performance.
    """
    
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    
    honest_clients = [c for c in clients if not c.is_malicious]
    if not honest_clients:
        # If all clients are malicious, no training occurs.
        return 0.0, 0.0
    
    for epoch in range(local_epochs):
        for client in honest_clients:
            client.reset_iterator()
            
        while True:
            smashed_activations_list = []
            labels_list = []
            active_clients = []
            
            for client in honest_clients:
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
    
    client_weights = [c.get_weights() for c in honest_clients]
    global_weights = server.aggregate_client_models(client_weights)
    
    for client in clients:
        client.set_weights(global_weights)
        
    honest_client_indices = [c.id for c in honest_clients]
    server.aggregate_server_models(active_client_indices=honest_client_indices)
        
    return avg_loss, avg_acc
