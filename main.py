
from matplotlib import pyplot as plt
from data_loader import Loader as ld
from MLP import MLP


def plot3d():
    l = ld(dimentions=3, train_percent=75)
    train_inp = l.get_train_inp()
    train_out = l.get_train_out()
    t1 = [i[0] for i in train_inp]
    t2 = [i[1] for i in train_inp]
    o = [i[0] for i in train_out]
    test_inp = l.get_test_inp()
    test_out = l.get_test_out()
    mlp = MLP(l, ( 3, 2))
    mlp.learn()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(t1, t2, o, 'r')
    out = mlp.calc(train_inp)
    ax.plot3D(t1, t2, out, 'b')
    t1 = [i[0] for i in test_inp]
    t2 = [i[1] for i in test_inp]
    o = [i[0] for i in test_out]
    out = mlp.calc(test_inp)
    ax.plot3D(t1, t2, out, 'g')
    plt.show()

plot3d()