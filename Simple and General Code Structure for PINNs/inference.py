# Create Test Data
h = 0.01
k = 0.01

x_test = torch.arange(0, 1, h)
y_test = torch.arange(0, 1, k)

XY_test = torch.stack(torch.meshgrid(x_test, y_test)).reshape(2,-1).T
XY_test = XY_test.to(device)

# Load best model
print("Loading best model...")
checkpoint = torch.load(checkpoint_path)
pinn.model.load_state_dict(checkpoint['model'])

# Inference
pinn.model.eval()
with torch.inference_mode():
    g_pred = pinn.model(XY_test).cpu().numpy()
    # g_pred = g_pred.reshape(len(x_test), len(y_test))

# Real solution
g_real = np.zeros((len(x_test), len(y_test)))
for i, xi in enumerate(x_test):
    for j, yj in enumerate(y_test):
        g_real[i, j] = np.cos(4*np.pi*(xi - yj))

# Turn off 'requires_grad' and turn pytorch tensors into numpy arrays
x_test = x_test.cpu().numpy().flatten()
y_test = y_test.cpu().numpy().flatten()
g_pred = g_pred.flatten()
g_real = g_real.flatten()

# Calculate NMSE (Normalized Mean Square Error)
test_error = g_real - g_pred
test_nmse = np.linalg.norm(g_real - g_pred, 2) / np.linalg.norm(g_real, 2)
print(f'Test NMSE: {test_nmse}')