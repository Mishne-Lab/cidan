from pick
def test_pickle_funcs():
    pickle_set_dir("test")
    os.mkdir("test")
    pickle_clear(trial_num=-1)
    assert not pickle_exist("test", trial_num=-1)
    obj = "pickle save"
    pickle_save(obj, "test",trial_num=-1)
    assert pickle_load("test", trial_num=-1)==obj
    assert pickle_exist("test", trial_num=-1)
    pickle_clear("test", trial_num=-1)
    assert not pickle_exist("test", trial_num=-1)
    assert len([f for f in os.listdir("test/-1") if f.endswith(".pickle")]) == 0

