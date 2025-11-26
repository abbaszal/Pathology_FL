import flwr as fl


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,    
    min_fit_clients=2,   # Wait for 2 clients
    min_available_clients=2,
)

if __name__ == "__main__":
    print("Starting Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
