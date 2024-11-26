# Data
Nx_train = Ny_train = 200
Nx_b_train = Ny_b_train = 5_000
Nx_val = Ny_val = 100
Nx_test = Ny_test = 100

# Model
n_hidden_layers = 3
hidden_layer_size = 64
layers = [2] + n_hidden_layers * [hidden_layer_size] + [1]
activation = nn.Tanh()

# Training
scheduler_f = 0.33
scheduler_p = 100
N_epochs_adam = 5_000
print_every = 100

# Others
checkpoint_path = "model.pth"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32

# Create "PINN" object
pinn = PINN(Nx_train, Ny_train,
            Nx_b_train, Ny_b_train,
            Nx_val, Ny_val,
            Nx_test, Ny_test,
            layers, activation,
            scheduler_f, scheduler_p,
            checkpoint_path,
            device=device,
            dtype=dtype)

# Train
pinn.train(N_epochs_adam, print_every)