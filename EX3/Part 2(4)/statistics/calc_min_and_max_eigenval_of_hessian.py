from pyhessian import hessian

def calc_min_and_max_eigenval_of_hessian(net, loss_fn, train_dataloader, n_eigenvalues):
    hessian_comp = hessian(net,
                           loss_fn,
                           dataloader=train_dataloader,
                           cuda=False)

    eigenvalues = hessian_comp.eigenvalues(top_n=n_eigenvalues)
    max_eigenvalue = eigenvalues[0][0]
    min_eigenvalue = eigenvalues[0][-1]

    if max_eigenvalue < min_eigenvalue:
        print("WEIRED!")
        print("min eigenvalue: " + str(min_eigenvalue))
        print("max eigenvalue: " + str(max_eigenvalue))

    return min_eigenvalue, max_eigenvalue