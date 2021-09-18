from cs5600_6600_f21_hw02 import *

num_iters = 1000

wmats = train_3_layer_nn(num_iters, X1, y_and, build_231_nn)
save(wmats, 'and_3_layer_ann.pck')
wmats = train_4_layer_nn(num_iters, X1, y_and, build_2331_nn)
save(wmats, 'and_4_layer_ann.pck')

wmats = train_3_layer_nn(num_iters, X1, y_or, build_231_nn)
save(wmats, 'or_3_layer_ann.pck')
wmats = train_4_layer_nn(num_iters, X1, y_or, build_2331_nn)
save(wmats, 'or_4_layer_ann.pck')

wmats = train_3_layer_nn(num_iters, X2, y_not, build_121_nn)
save(wmats, 'not_3_layer_ann.pck')
wmats = train_4_layer_nn(num_iters, X2, y_not, build_1221_nn)
save(wmats, 'not_4_layer_ann.pck')

wmats = train_3_layer_nn(num_iters, X1, y_xor, build_231_nn)
save(wmats, 'xor_3_layer_ann.pck')
wmats = train_4_layer_nn(num_iters, X1, y_xor, build_2331_nn)
save(wmats, 'xor_4_layer_ann.pck')

wmats = train_3_layer_nn(num_iters, X3, bool_exp, build_421_nn)
save(wmats, 'bool_3_layer_ann.pck')
wmats = train_4_layer_nn(num_iters, X3, bool_exp, build_4221_nn)
save(wmats, 'bool_4_layer_ann.pck')
