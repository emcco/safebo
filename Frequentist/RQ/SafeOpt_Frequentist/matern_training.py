'''
   model.train()
    likelihood = model.likelihood
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
   

    for i in range(100):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Access the Matern kernel and print the interpretable parameters
    kernel = model.covar_module.base_kernel

    # Print interpretable hyperparameters
    print(f"Lengthscale: {kernel.lengthscale.item()}")
    print(f"Outputscale (variance): {model.covar_module.outputscale.item()}")
    print(f"Nu: {kernel.nu}")  # This is the smoothness parameter, but it's not learned directly.




'''


# Results in nu=2.5