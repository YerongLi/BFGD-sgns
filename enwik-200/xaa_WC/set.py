# Train the model
start_time = time.time()
opt_experiment(model_enwik,
               mode='AM', 
               d=dimension,
               eta = 1e-5,
               MAX_ITER=50000,
               from_iter=189000,
               start_from='SVD',
               init=(True, None, None), display=True)
print("--- %s seconds ---" % (time.time() - start_time))

xaa


Iter #: 238400 loss 950890.4636540753
Iter #: 238500 loss 950889.3695294072
Iter #: 238600 loss 950888.2762317525
Iter #: 238700 loss 950887.1837551469
Iter #: 238800 loss 950886.0920885435
Iter #: 238900 loss 950885.0012409609
Iter #: 239000 loss 950883.9112154882
--- 614.1901476383209 seconds ---