import sys
import os
import torch
sys.path.append(os.path.join(os.getcwd(), 'ad-sfl-experiments'))
from src.models.registry import get_client_model, get_server_model
from src.models.split import split_model_output

try:
    print("Testing LeNet Split...")
    client_lenet = get_client_model('lenet')
    server_lenet = get_server_model('lenet', num_classes=10)
    
    dummy_mnist = torch.randn(2, 1, 28, 28) # Batch size 2
    client_out = client_lenet(dummy_mnist)
    print(f"LeNet Client Output Shape: {client_out.shape} (Expected: 2, 6, 14, 14)")
    
    orig_out, server_in = split_model_output(client_out)
    server_out = server_lenet(server_in)
    print(f"LeNet Server Output Shape: {server_out.shape} (Expected: 2, 10)")

    print("\nTesting ResNet18 Split...")
    client_resnet = get_client_model('resnet18')
    server_resnet = get_server_model('resnet18', num_classes=100)
    
    dummy_cifar = torch.randn(2, 3, 32, 32)
    client_out2 = client_resnet(dummy_cifar)
    print(f"ResNet18 Client Output Shape: {client_out2.shape} (Expected: 2, 64, 32, 32)")
    
    orig_out2, server_in2 = split_model_output(client_out2)
    server_out2 = server_resnet(server_in2)
    print(f"ResNet18 Server Output Shape: {server_out2.shape} (Expected: 2, 100)")

    print("\nAll model splits instantiated and passed dummy forward pass successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
