from pyhessian import hessian

def count_num_weights(N, hidden_dim):
    count = 1*hidden_dim       # input_layer adds 1
    count += (N-2)*hidden_dim*hidden_dim
    count += 1*hidden_dim      # output_layer adds 1

    return count

def calc_min_and_max_eigenval_of_hessian(net, loss_fn, train_dataloader, N, hidden_dim):
    n_eigenvalues = count_num_weights(N, hidden_dim)
    hessian_comp = hessian(net,
                           loss_fn,
                           dataloader=train_dataloader,
                           cuda=False)

    eigenvalues = hessian_comp.eigenvalues(top_n=n_eigenvalues, maxIter=300, tol=1e-4)
    max_eigenvalue = eigenvalues[0][0]
    min_eigenvalue = eigenvalues[0][-1]

    if max_eigenvalue < min_eigenvalue:
        print("WEIRED!")
        print("min eigenvalue: " + str(min_eigenvalue))
        print("max eigenvalue: " + str(max_eigenvalue))

    return min_eigenvalue, max_eigenvalue