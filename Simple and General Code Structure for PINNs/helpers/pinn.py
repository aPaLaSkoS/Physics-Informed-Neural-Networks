class PINN():
    def __init__(self,
                 Nx_train, Ny_train,
                 Nx_b_train, Ny_b_train,
                 Nx_val, Ny_val,
                 Nx_test, Ny_test,
                 layers, activation,
                 scheduler_f, scheduler_p,
                 checkpoint_path,
                 device='cpu',
                 dtype=torch.float32):

        # Constants
        self.v = 16 * math.pi**2
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype

        # Some initializations
        self.epoch = 1

        # Create the data
        print("Creating the data...")
        self.data = Data(device, dtype)
        self.XY_train = self.data.sample_domain(Nx_train, Ny_train)
        self.XY_b_train, self.g_b_train = self.data.sample_boundary(Nx_b_train, Ny_b_train)
        self.XY_val = self.data.sample_domain(Nx_val, Ny_val)
        self.x_test, self.y_test, self.XY_test = self.data.sample_domain(Nx_test, Ny_test, is_test=True)

        # Define the model
        self.model = MLP(layers=layers,
                         activation=activation,
                         weight_init=lambda m: nn.init.xavier_normal_(m.data, nn.init.calculate_gain('tanh')),
                         bias_init=lambda m: nn.init.zeros_(m.data),
                         device=device)

        # Set the optimizers
        self.adam = torch.optim.Adam(self.model.parameters())

        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter = 50_000,
            max_eval = 50_000,
            history_size = 50,
            tolerance_grad = 1e-7,
            tolerance_change = 1.0* np.finfo(float).eps,
            line_search_fn ="strong_wolfe"
        )

        # Set the Loss function
        self.criterion = torch.nn.MSELoss()


    def calculate_g(self, XY):
        return self.model(XY)


    def grad(self, output, input):
        return torch.autograd.grad(
                    output, input,
                    grad_outputs=torch.ones_like(output),
                    retain_graph=True,
                    create_graph=True
                )[0]


    def calculate_pde_residual(self, XY):
        # Forward pass
        g_hat = self.calculate_g(XY)

        # Calculate 1st and 2nd derivatives
        dg_dX = self.grad(g_hat, XY)
        dg_dXX = self.grad(dg_dX, XY)

        # Retrieve the partial gradients
        dg_dxx = dg_dXX[:, 0]
        dg_dyy = dg_dXX[:, 1]

        return dg_dxx + dg_dyy + self.v*g_hat.squeeze() - \
               self.v*torch.cos(4*math.pi*(XY[:, 0] - XY[:, 1]))


    def calculate_train_loss(self):
        # Calculate the boundary loss
        loss_b = self.criterion(self.calculate_g(self.XY_b_train), self.g_b_train)

        # Calculate the in-domain loss
        pde_res = self.calculate_pde_residual(self.XY_train)
        loss_pde = self.criterion(pde_res, torch.zeros_like(pde_res, dtype=self.dtype, device=self.device))

        # Calculate total loss
        loss = loss_b + loss_pde

        return loss_b, loss_pde, loss


    def calculate_val_loss(self):
        pde_res = self.calculate_pde_residual(self.XY_val)
        val_loss = self.criterion(pde_res, torch.zeros_like(pde_res, dtype=self.dtype, device=self.device))
        return val_loss


    def train_step(self):
        # "Zero" the gradients
        self.optimizer.zero_grad()

        # Get losses
        loss_b, loss_pde, loss = self.calculate_train_loss()
        self.val_loss = self.calculate_val_loss().cpu().item()

        # # Update scheduler
        # self.scheduler.step(self.val_loss)

        # Backpropagate the loss
        loss.backward()

        # print losses
        self.flag = 0
        self.checkpoint_and_print_losses(loss_b.cpu().item(),
                                         loss_pde.cpu().item(),
                                         loss.cpu().item())

        # Update "epoch"
        self.epoch = self.epoch + 1

        return loss


    def train(self, N_epochs_adam, print_every):
        self.print_every = print_every

        # Set model in training mode
        self.model.train()

        # Start with the "Adam" optimizer
        self.optimizer = self.adam

        # # Setting the scheduler
        # self.scheduler = ReduceLROnPlateau(self.optimizer,
        #                                    mode='min',
        #                                    factor=scheduler_f,
        #                                    patience=scheduler_p,
        #                                    verbose=True)

        for i in range(N_epochs_adam):
            self.optimizer.step(self.train_step)

        # Switch to "LBFGS" optimizer
        self.optimizer = self.lbfgs

        # # Setting the scheduler
        # self.scheduler = ReduceLROnPlateau(self.optimizer,
        #                                    mode='min',
        #                                    factor=scheduler_f,
        #                                    patience=scheduler_p,
        #                                    verbose=True)

        self.optimizer.step(self.train_step)


    def checkpoint_and_print_losses(self, loss_b, loss_pde, loss):
        if self.epoch == 1:
            self.best_val_loss = self.val_loss
            best_epoch = -1
            self.checkpoint()
            self.flag = 1
            print(f"Epoch: {self.epoch} | loss_b: {loss_b} | loss_pde: {loss_pde} | loss: {loss} | val_loss: {self.val_loss} - *Checkpoint*")
        else:
            if self.val_loss < self.best_val_loss:
                self.best_val_loss = self.val_loss
                best_epoch = self.epoch
                self.checkpoint()
                self.flag = 1
                if self.epoch % print_every == 0:
                    print(f"Epoch: {self.epoch} | loss_b: {loss_b} | loss_pde: {loss_pde} | loss: {loss} | val_loss: {self.val_loss} - *Checkpoint*")
            # elif self.epoch - best_epoch > self.patience:
            #     if self.epoch % print_every == 0:
            #         print(f"\nEarly stopping applied at epoch {self.epoch}.")
            #     break

        if (self.flag == 0) and (self.epoch % print_every == 0):
            print(f"Epoch: {self.epoch} | loss_b: {loss_b} | loss_pde: {loss_pde} | loss: {loss} | val_loss: {self.val_loss} - *Checkpoint*")

    def checkpoint(self):
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.state_dict()
        }, self.checkpoint_path)