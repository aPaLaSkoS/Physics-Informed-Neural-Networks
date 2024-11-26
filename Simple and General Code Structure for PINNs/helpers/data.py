def generate_random_numbers(N, dtype):
    random_numbers = torch.rand(N - 1, dtype=dtype)
    random_numbers = torch.cat([random_numbers, torch.tensor([1.0], dtype=dtype)])
    return random_numbers


class Data():
  def __init__(self,
               x_min, x_max,
               y_min, y_max,
               Nc, Nl_BC, Nr_BC,
               Nx_test, Ny_test,
               device='cpu',
               dtype=torch.float32,
               EPS=1e-5):

    super().__init__()
    self.x_min = x_min
    self.x_max = x_max
    self.y_min = y_min
    self.y_max = y_max
    self.Nc = Nc
    self.Nl_BC = Nl_BC
    self.Nr_BC = Nr_BC
    self.Nx_test = Nx_test
    self.Ny_test = Ny_test
    self.device = device
    self.dtype = dtype
    self.EPS = EPS

  # ============================= TRAINING DATA ==============================

  # *** Create collocation points ***
  def sample_inside(self, Nc, x_min, x_max, y_min, y_max):
    # Random Grid
    XY_c = qmc.scale(qmc.LatinHypercube(2).random(Nc),
                      [x_min + self.EPS, y_min + self.EPS],
                      [x_max - self.EPS, y_max - self.EPS])
    return torch.tensor(XY_c, dtype=self.dtype, device=self.device)

  # *** Boundary Conditions ***
  # 1. u(x,0) = 0
  # 2. u(x,1) = 0
  # 3. u(0,y) = 0
  # 4. u(1,y) = 0
  def sample_boundary(self, Nl_BC, Nr_BC, x_min, x_max, y_min, y_max):
    xl_BC = np.linspace(x_min, x_max, Nl_BC).reshape(-1, 1)
    yl = np.zeros((Nl_BC, 1))
    XY_x0 = np.concatenate((xl_BC, yl), axis=1)
    XY_x0 = torch.tensor(XY_x0, dtype=self.dtype, device=self.device)

    xr_BC = np.linspace(x_min, x_max, Nr_BC).reshape(-1, 1)
    yr = np.ones((Nr_BC, 1))
    XY_x1 = np.concatenate((xr_BC, yr), axis=1)
    XY_x1 = torch.tensor(XY_x1, dtype=self.dtype, device=self.device)

    xl = np.zeros((Nl_BC, 1))
    yl_BC = np.linspace(y_min, y_max, Nl_BC).reshape(-1, 1)
    XY_0y = np.concatenate((xl, yl_BC), axis=1)
    XY_0y = torch.tensor(XY_0y, dtype=self.dtype, device=self.device)

    xr = np.ones((Nr_BC, 1))
    yr_BC = np.linspace(y_min, y_max, Nr_BC).reshape(-1, 1)
    XY_1y = np.concatenate((xr_BC, yr), axis=1)
    XY_1y = torch.tensor(XY_1y, dtype=self.dtype, device=self.device)

    u_xl = np.zeros(Nl_BC)
    u_xr = np.zeros(Nr_BC)
    u_yl = np.zeros(Nl_BC)
    u_yr = np.zeros(Nr_BC)
    u_xl = torch.tensor(u_xl, dtype=self.dtype, device=self.device)
    u_xr = torch.tensor(u_xr, dtype=self.dtype, device=self.device)
    u_yl = torch.tensor(u_yl, dtype=self.dtype, device=self.device)
    u_yr = torch.tensor(u_yr, dtype=self.dtype, device=self.device)

    return XY_x0, XY_x1, XY_0y, XY_1y, u_xl, u_xr, u_yl, u_yr

  # ============================ VALIDATION DATA =============================
  def sample_validation(self, N_val, x_min, x_max, y_min, y_max):
    # Random Grid
    XY_val = qmc.scale(qmc.LatinHypercube(2).random(N_val), [x_min, y_min], [x_max, y_max])
    return torch.tensor(XY_val, dtype=self.dtype, device=self.device)

  # =============================== TEST DATA ================================
  def sample_test(self, Nx_test, Ny_test, x_min, x_max, y_min, y_max):
    # Uniform Grid
    xc = np.linspace(x_min, x_max, Nx_test)
    yc = np.linspace(y_min, y_max, Ny_test)
    X_mesh, Y_mesh = np.meshgrid(xc, yc)
    XY_test = np.concatenate((X_mesh.flatten().reshape(-1, 1), Y_mesh.flatten().reshape(-1, 1)), axis=1)

    return xc, yc, torch.tensor(XY_test, dtype=self.dtype, device=self.device)