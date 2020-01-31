from LSSC.Parameters import *
def test_Parameters():
    params = Parameters(metric="l2",knn=20,accuracy=30,connections=43,num_threads=23, num_eig=25)
    assert params.metric == 'l2'
    assert params.knn ==20
    assert params.accuracy ==30
    assert params.connections==43
    assert params.num_threads == 23
    assert params.num_eig ==25
