def run_sfl_round(clients, server, local_epochs=1):
    """
    Matches the posted implementation's sequencing:
    - Train clients SEQUENTIALLY (one client finishes all its local batches for the epoch, then next client).
    - Server model is a single global model updated continuously as each client's batches are processed.
    - FedAvg is applied ONLY to client-side models after all clients finish local training.
    """

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    # Local training (sequential clients, sequential batches)
    for ep in range(local_epochs):
        for client in clients:
            client.reset_iterator()

            while True:
                # Client forward (one batch)
                smashed_data, labels = client.forward_pass(global_round=ep)

                # No more data for this client in this local epoch
                if smashed_data is None:
                    break

                # Server forward+backward+update (one global server model)
                grad_to_client, loss, acc = server.train_step(
                    smashed_data, labels, client_id=client.id
                )

                total_loss += float(loss)
                total_acc += float(acc)
                total_batches += 1

                # Client backward+update (one batch)
                client.backward_pass(grad_to_client)

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    avg_acc = total_acc / total_batches if total_batches > 0 else 0.0

    # FedAvg of Client Models ONLY (no server aggregation)
    client_weights = [c.get_weights() for c in clients]
    global_weights = server.aggregate_client_models(client_weights)

    # Broadcast updated client model back to all clients
    for client in clients:
        client.set_weights(global_weights)

    return avg_loss, avg_acc