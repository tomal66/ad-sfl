import json
import copy

with open('demo_setup.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

gold_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('run_sfl_gold_round' in line for line in cell['source']):
        gold_idx = i
        break

if gold_idx != -1:
    # 1. Add markdown cell
    md_cell = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            "### Step 5: Centinel Evaluation\n",
            "\n",
            "This section evaluates the Centinel defense method, which uses anomaly detection on activation centroids and Subjective Logic to build client reputations over time."
        ]
    }
    
    # Setup cell (hyperparams + reference dataset)
    setup_source = [
        "from src.algorithms.centinel import run_sfl_centinel_round, CentinelState\n",
        "from torch.utils.data import Subset\n",
        "import numpy as np\n",
        "\n",
        "# --- Centinel Hyperparameters ---\n",
        "NUM_REF_SAMPLES_PER_LABEL = 10\n",
        "TAU = 0.1\n",
        "OMEGA = 0.7\n",
        "Q_i = 0.8\n",
        "rho = 0.4\n",
        "eta = 0.6\n",
        "kappa = 0.7\n",
        "zeta = 0.3\n",
        "\n",
        "# Build a small, class-balanced reference dataset from the test set\n",
        "labels = np.array(test_data['label'])\n",
        "ref_indices = []\n",
        "for c in range(10): # CIFAR10 has 10 classes\n",
        "    c_indices = np.where(labels == c)[0]\n",
        "    ref_indices.extend(np.random.choice(c_indices, NUM_REF_SAMPLES_PER_LABEL, replace=False))\n",
        "\n",
        "ref_dataset = Subset(test_data, ref_indices)\n",
        "ref_loader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=False)\n",
        "\n",
        "state = CentinelState(num_clients, tau=TAU, omega=OMEGA, Q_i=Q_i, rho=rho, eta=eta, kappa=kappa, zeta=zeta)\n",
        "\n",
        "# Quick initial centroids generation using client 0\n",
        "print(\"Re-initializing models for Centinel...\")\n",
        "base_client_model, server_model = get_split_models(DATASET, weights=PRETRAINED_WEIGHTS)\n",
        "server = SplitFedServer(model=server_model, num_clients=num_clients, lr=learning_rate, device=device)\n",
        "clients = []\n",
        "for i in range(num_clients):\n",
        "    client_model = copy.deepcopy(base_client_model)\n",
        "    is_mal = i in malicious_clients_indices\n",
        "    c_dataset = client_datasets[i]\n",
        "    if is_mal and ATTACK_TYPE != 'none':\n",
        "        c_dataset = PoisonedDataset(c_dataset, attack_type=ATTACK_TYPE, attack_kwargs=ATTACK_KWARGS, dataset_name=DATASET, seed=GLOBAL_SEED+i)\n",
        "    client = SplitFedClient(client_id=i, model=client_model, dataset=c_dataset, batch_size=batch_size, lr=learning_rate, device=device, is_malicious=is_mal)\n",
        "    clients.append(client)\n",
        "\n",
        "from src.algorithms.centinel import compute_centroids\n",
        "client0 = clients[0]\n",
        "client0.model.eval()\n",
        "with torch.no_grad():\n",
        "    for data, target in ref_loader:\n",
        "        acts = client0.model(data.to(device))\n",
        "        c, counts = compute_centroids(acts, target.to(device))\n",
        "        for lbl in c:\n",
        "            if lbl in state.global_centroids:\n",
        "                state.global_centroids[lbl] = (state.global_centroids[lbl] + c[lbl])/2\n",
        "            else:\n",
        "                state.global_centroids[lbl] = c[lbl]\n"
    ]
    setup_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': setup_source
    }
    
    # 2. Add code cell for SFL Centinel
    code_cell = copy.deepcopy(nb['cells'][gold_idx])
    code_cell['execution_count'] = None
    code_cell['outputs'] = []
    
    source_lines = code_cell['source']
    new_source = []
    
    skip_next = False
    for line in source_lines:
        if "Re-initializing" in line or "base_client_model, server_model =" in line or "SplitFedServer(" in line or "clients = []" in line or "for i in range(" in line or "is_mal =" in line or "c_dataset =" in line or "if is_mal and " in line or "PoisonedDataset(" in line or "SplitFedClient(" in line or "clients.append(" in line:
            continue
            
        line = line.replace('run_sfl_gold_round(clients, server, local_epochs=1)', 'run_sfl_centinel_round(clients, server, state, ref_loader, local_epochs=1, device=device)')
        
        # Centinel returns 4 values, so we need to capture train_loss, train_acc, scores, accepted
        if 'train_loss, train_acc =' in line:
            line = line.replace('train_loss, train_acc =', 'train_loss, train_acc, scores, accepted_clients =')
            
        line = line.replace('gold_historical_', 'centinel_historical_')
        new_source.append(line)
        
    code_cell['source'] = new_source
    
    # 3. Add plot cell
    plot_idx = gold_idx + 1
    if plot_idx < len(nb['cells']) and 'Plotting' in ''.join(nb['cells'][plot_idx]['source']):
        plot_cell = copy.deepcopy(nb['cells'][plot_idx])
        plot_cell['execution_count'] = None
        plot_cell['outputs'] = []
        plot_source = [l.replace('gold_historical_', 'centinel_historical_').replace('SplitFed Gold', 'SplitFed Centinel') for l in plot_cell['source']]
        plot_cell['source'] = plot_source
        
        nb['cells'].insert(plot_idx + 1, md_cell)
        nb['cells'].insert(plot_idx + 2, setup_cell)
        nb['cells'].insert(plot_idx + 3, code_cell)
        nb['cells'].insert(plot_idx + 4, plot_cell)
    else:
        nb['cells'].insert(gold_idx + 1, md_cell)
        nb['cells'].insert(gold_idx + 2, setup_cell)
        nb['cells'].insert(gold_idx + 3, code_cell)
        
    with open('demo_setup.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Successfully added SFL Centinel cells to demo_setup.ipynb")
else:
    print("Could not find the gold cell in demo_setup.ipynb")
